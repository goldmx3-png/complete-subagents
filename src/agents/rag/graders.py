"""
Agentic RAG Graders - Document relevance, hallucination, and answer quality checking
Based on latest LangGraph and LlamaIndex patterns from Context7
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Pydantic Models for Structured Outputs
class DocumentRelevanceGrade(BaseModel):
    """Binary score for document relevance to query"""
    binary_score: Literal["yes", "no"] = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Brief explanation of why document is relevant or not"
    )


class HallucinationGrade(BaseModel):
    """Binary score for hallucination check"""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Brief explanation of grounding assessment"
    )


class AnswerQualityGrade(BaseModel):
    """Binary score for answer quality"""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Brief explanation of answer quality"
    )


class AnswerDepthGrade(BaseModel):
    """Score for answer depth and comprehensiveness"""
    binary_score: Literal["comprehensive", "shallow"] = Field(
        description="Answer is comprehensive or shallow"
    )
    reasoning: str = Field(
        description="Brief explanation of depth assessment"
    )
    aspects_covered: int = Field(
        description="Number of relevant aspects/scenarios covered in answer"
    )
    suggestions: str = Field(
        description="What aspects should be added for a comprehensive answer"
    )


class DocumentGrader:
    """
    Grades the relevance of retrieved documents to the user question

    Based on LangGraph adaptive RAG pattern:
    - Uses LLM with structured output
    - Binary grading (relevant/not relevant)
    - Routes to generate, rewrite, or fallback
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        # Use a smaller, faster model for grading
        self.llm = llm_client or OpenRouterClient(
            model=settings.classifier_model  # Use faster model for grading
        )

    async def grade_documents(
        self,
        question: str,
        documents: List[Dict],
        threshold: float = 0.6
    ) -> Dict:
        """
        Grade relevance of retrieved documents

        Args:
            question: User's question
            documents: List of retrieved document chunks
            threshold: Minimum fraction of relevant docs needed

        Returns:
            {
                "relevant_documents": List[Dict],
                "is_relevant": bool,
                "relevance_score": float,
                "reasoning": List[str]
            }
        """
        if not documents:
            return {
                "relevant_documents": [],
                "is_relevant": False,
                "relevance_score": 0.0,
                "reasoning": ["No documents to grade"]
            }

        logger.info(f"Grading {len(documents)} documents for relevance")

        relevant_docs = []
        reasoning_list = []

        # Grade each document
        for i, doc in enumerate(documents[:5]):  # Grade top 5 to save costs
            payload = doc.get("payload", {})
            doc_text = payload.get("text", "")

            # Use structured output for grading
            grade = await self._grade_single_document(question, doc_text)

            if grade.binary_score == "yes":
                relevant_docs.append(doc)
                logger.debug(f"Doc {i+1}: RELEVANT - {grade.reasoning}")
            else:
                logger.debug(f"Doc {i+1}: NOT RELEVANT - {grade.reasoning}")

            reasoning_list.append(f"Doc {i+1}: {grade.reasoning}")

        # Calculate relevance score
        relevance_score = len(relevant_docs) / len(documents[:5]) if documents else 0.0
        is_relevant = relevance_score >= threshold

        logger.info(
            f"Relevance grading complete: {len(relevant_docs)}/{len(documents[:5])} relevant "
            f"(score={relevance_score:.2f}, threshold={threshold})"
        )

        return {
            "relevant_documents": relevant_docs,
            "is_relevant": is_relevant,
            "relevance_score": relevance_score,
            "reasoning": reasoning_list
        }

    async def _grade_single_document(
        self,
        question: str,
        document: str
    ) -> DocumentRelevanceGrade:
        """Grade a single document using LLM with structured output"""

        prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.

Here is the retrieved document:
---
{document[:1000]}
---

Here is the user question:
---
{question}
---

If the document contains keywords or semantic meaning related to the question, grade it as relevant.
It does not need to be a complete answer, just related to the question.

Provide your assessment as:
- binary_score: "yes" if relevant, "no" if not
- reasoning: Brief explanation (1 sentence)
"""

        try:
            # Use structured output
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=DocumentRelevanceGrade,
                temperature=0
            )
            return result

        except Exception as e:
            logger.error(f"Document grading error: {e}")
            # Default to relevant on error to avoid blocking retrieval
            return DocumentRelevanceGrade(
                binary_score="yes",
                reasoning=f"Grading failed (error: {str(e)}), defaulting to relevant"
            )


class HallucinationGrader:
    """
    Grades whether an answer is grounded in the provided documents

    Based on LangGraph CRAG (Corrective RAG) pattern:
    - Checks if answer contains facts not in documents
    - Binary grading (grounded/hallucinated)
    - Triggers regeneration if hallucination detected
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(
            model=settings.classifier_model
        )

    async def grade_hallucination(
        self,
        documents: List[Dict],
        generation: str
    ) -> Dict:
        """
        Check if generation is grounded in documents

        Args:
            documents: List of source documents
            generation: Generated answer

        Returns:
            {
                "is_grounded": bool,
                "reasoning": str,
                "score": Literal["yes", "no"]
            }
        """
        if not documents:
            logger.warning("No documents provided for hallucination check")
            return {
                "is_grounded": True,  # Allow answer if no docs
                "reasoning": "No source documents to verify against",
                "score": "yes"
            }

        logger.info("Checking answer for hallucinations")

        # Combine document texts
        doc_texts = []
        for doc in documents[:5]:
            payload = doc.get("payload", {})
            text = payload.get("text", "")
            doc_texts.append(text)

        combined_docs = "\n\n".join(doc_texts)

        prompt = f"""You are a grader assessing whether an answer is grounded in a set of facts.

Here are the facts:
---
{combined_docs[:2000]}
---

Here is the answer:
---
{generation}
---

Task: Determine if the answer is grounded in the facts.
- If the answer contains statements that are NOT supported by the facts, grade it as "no" (hallucination)
- If all statements in the answer are supported by or reasonably inferred from the facts, grade it as "yes" (grounded)

Provide your assessment as:
- binary_score: "yes" if grounded, "no" if hallucination detected
- reasoning: Brief explanation (1-2 sentences)
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=HallucinationGrade,
                temperature=0
            )

            is_grounded = result.binary_score == "yes"

            if is_grounded:
                logger.info(f"✓ Answer is grounded: {result.reasoning}")
            else:
                logger.warning(f"✗ Hallucination detected: {result.reasoning}")

            return {
                "is_grounded": is_grounded,
                "reasoning": result.reasoning,
                "score": result.binary_score
            }

        except Exception as e:
            logger.error(f"Hallucination grading error: {e}")
            # Default to grounded on error to avoid blocking
            return {
                "is_grounded": True,
                "reasoning": f"Grading failed (error: {str(e)}), defaulting to grounded",
                "score": "yes"
            }


class AnswerGrader:
    """
    Grades whether an answer actually addresses the user's question

    Based on LangGraph adaptive RAG pattern:
    - Checks if answer is useful and addresses the question
    - Binary grading (useful/not useful)
    - Triggers regeneration or query rewrite if not useful
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(
            model=settings.classifier_model
        )

    async def grade_answer(
        self,
        question: str,
        generation: str
    ) -> Dict:
        """
        Check if answer addresses the question

        Args:
            question: User's question
            generation: Generated answer

        Returns:
            {
                "is_useful": bool,
                "reasoning": str,
                "score": Literal["yes", "no"]
            }
        """
        logger.info("Checking if answer addresses question")

        prompt = f"""You are a grader assessing whether an answer addresses a question.

Here is the question:
---
{question}
---

Here is the answer:
---
{generation}
---

Task: Determine if the answer addresses the question.
- If the answer resolves the question or provides useful information toward answering it, grade it as "yes"
- If the answer is off-topic, vague, or doesn't help answer the question, grade it as "no"

Provide your assessment as:
- binary_score: "yes" if useful, "no" if not useful
- reasoning: Brief explanation (1-2 sentences)
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=AnswerQualityGrade,
                temperature=0
            )

            is_useful = result.binary_score == "yes"

            if is_useful:
                logger.info(f"✓ Answer is useful: {result.reasoning}")
            else:
                logger.warning(f"✗ Answer not useful: {result.reasoning}")

            return {
                "is_useful": is_useful,
                "reasoning": result.reasoning,
                "score": result.binary_score
            }

        except Exception as e:
            logger.error(f"Answer grading error: {e}")
            # Default to useful on error to avoid blocking
            return {
                "is_useful": True,
                "reasoning": f"Grading failed (error: {str(e)}), defaulting to useful",
                "score": "yes"
            }


class AnswerDepthGrader:
    """
    Grades the depth and comprehensiveness of an answer

    Checks:
    - Does the answer cover multiple aspects/scenarios present in context?
    - Are workflows, variations, and use cases explained?
    - Is the answer sufficiently detailed vs shallow?
    - Does it use the majority of relevant information from context?
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(
            model=settings.classifier_model
        )

    async def grade_depth(
        self,
        question: str,
        answer: str,
        documents: List[Dict],
        min_word_count: int = 150
    ) -> Dict:
        """
        Check if answer is comprehensive enough

        Args:
            question: User's question
            answer: Generated answer
            documents: Source documents used
            min_word_count: Minimum word count for comprehensive answers

        Returns:
            {
                "is_comprehensive": bool,
                "reasoning": str,
                "aspects_covered": int,
                "suggestions": str,
                "word_count": int
            }
        """
        logger.info("Checking answer depth and comprehensiveness")

        # Quick word count check
        word_count = len(answer.split())

        # For "what is" or explanatory queries, enforce minimum depth
        question_lower = question.lower()
        is_explanatory = any(trigger in question_lower for trigger in
                            ["what is", "what are", "explain", "describe", "how does", "tell me about"])

        if is_explanatory and word_count < min_word_count:
            logger.warning(f"Answer too short: {word_count} words (min: {min_word_count})")
            return {
                "is_comprehensive": False,
                "reasoning": f"Answer is too brief ({word_count} words) for an explanatory question",
                "aspects_covered": 0,
                "suggestions": "Expand the answer to cover workflows, scenarios, and use cases",
                "word_count": word_count
            }

        # Extract context to understand available information
        doc_texts = []
        for doc in documents[:10]:  # Check first 10 docs
            payload = doc.get("payload", {})
            text = payload.get("text", "")
            doc_texts.append(text)

        combined_context = "\n\n".join(doc_texts)[:3000]  # Limit context size

        prompt = f"""You are evaluating the depth and comprehensiveness of an answer.

QUESTION:
{question}

AVAILABLE CONTEXT (what information is available):
{combined_context}

ANSWER PROVIDED:
{answer}

TASK: Assess if the answer is comprehensive or shallow.

A COMPREHENSIVE answer should:
- Cover multiple relevant aspects/scenarios present in the context
- Explain different workflows, types, or variations when applicable
- Provide context about how things work in different situations (e.g., bulk vs single, sequential vs non-sequential)
- Use specific details from the available information
- Cover both common cases and important variations

A SHALLOW answer:
- Only provides a basic definition without depth
- Misses important workflows or scenarios clearly present in context
- Doesn't explain variations or different types
- Lacks specific details that are available in context

Provide assessment as:
- binary_score: "comprehensive" if answer covers multiple aspects thoroughly, "shallow" if it's surface-level
- reasoning: Explanation of your assessment (2-3 sentences)
- aspects_covered: Count of distinct aspects/scenarios covered (e.g., workflows, use cases, variations)
- suggestions: What should be added to make it comprehensive (if shallow)
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=AnswerDepthGrade,
                temperature=0
            )

            is_comprehensive = result.binary_score == "comprehensive"

            if is_comprehensive:
                logger.info(f"✓ Answer is comprehensive: {result.reasoning}")
            else:
                logger.warning(f"✗ Answer is shallow: {result.reasoning}")
                logger.info(f"Suggestions: {result.suggestions}")

            return {
                "is_comprehensive": is_comprehensive,
                "reasoning": result.reasoning,
                "aspects_covered": result.aspects_covered,
                "suggestions": result.suggestions,
                "word_count": word_count
            }

        except Exception as e:
            logger.error(f"Depth grading error: {e}")
            # Default to comprehensive on error to avoid blocking
            return {
                "is_comprehensive": True,
                "reasoning": f"Grading failed (error: {str(e)}), defaulting to comprehensive",
                "aspects_covered": 1,
                "suggestions": "",
                "word_count": word_count
            }
