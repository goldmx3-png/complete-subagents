"""
Meta-Cognitive RAG Agent with Self-Improvement
Based on advanced agentic RAG patterns from research

Features:
- Recursive self-improvement loop
- Gap analysis and knowledge identification
- Quality assessment at each iteration
- Self-correction and retry strategies
- Meta-learning from successful retrievals
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import time
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGap(BaseModel):
    """Identified gap in current knowledge/response"""
    gap_type: str = Field(description="Type of gap: missing_info, unclear, incomplete, or inaccurate")
    description: str = Field(description="What information is missing or unclear")
    suggested_query: str = Field(description="Suggested refined query to fill this gap")
    priority: int = Field(description="Priority 1-5 (5=critical, 1=nice-to-have)")


class GapAnalysisResult(BaseModel):
    """Result of gap analysis"""
    has_gaps: bool = Field(description="Whether significant gaps were found")
    gaps: List[KnowledgeGap] = Field(description="List of identified gaps")
    overall_assessment: str = Field(description="Overall quality assessment")
    confidence_score: float = Field(description="Confidence in current answer (0-1)")
    recommended_action: str = Field(description="continue, refine_query, or accept_answer")


class IterativeRefinementResult(BaseModel):
    """Result of iterative refinement assessment"""
    should_continue: bool = Field(description="Whether to continue refining")
    missing_aspects: List[str] = Field(description="What aspects are still missing")
    suggested_improvements: str = Field(description="How to improve the answer")
    completeness_score: float = Field(description="Answer completeness (0-1)")


class GapAnalyzer:
    """
    Analyzes responses for knowledge gaps and missing information

    Based on meta-cognitive patterns from advanced agentic RAG research
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        """
        Initialize gap analyzer

        Args:
            llm_client: LLM client for gap analysis
        """
        self.llm = llm_client or OpenRouterClient(model=settings.classifier_model)

    async def analyze_gaps(
        self,
        question: str,
        current_answer: str,
        retrieved_context: str,
        previous_attempts: int = 0
    ) -> GapAnalysisResult:
        """
        Analyze current answer for knowledge gaps

        Args:
            question: Original user question
            current_answer: Current generated answer
            retrieved_context: Context that was retrieved
            previous_attempts: Number of previous attempts

        Returns:
            Gap analysis result with identified gaps
        """
        logger.info("Analyzing answer for knowledge gaps")

        prompt = f"""You are a meta-cognitive agent analyzing the quality and completeness of an answer.

ORIGINAL QUESTION:
{question}

RETRIEVED CONTEXT:
{retrieved_context[:2000]}

CURRENT ANSWER:
{current_answer}

PREVIOUS ATTEMPTS: {previous_attempts}

TASK: Identify gaps, missing information, or areas needing improvement.

Analyze:
1. Does the answer fully address the question?
2. Is there important information in the context that wasn't used?
3. Are there aspects of the question left unanswered?
4. Is the answer clear and comprehensive?

For each significant gap:
- gap_type: missing_info, unclear, incomplete, or inaccurate
- description: What's missing or unclear
- suggested_query: A refined query to fill this gap
- priority: 1-5 (5=critical, 1=nice-to-have)

Provide:
- has_gaps: true if significant gaps exist (priority >= 3)
- gaps: List of KnowledgeGap objects
- overall_assessment: Brief quality assessment
- confidence_score: 0-1 (how confident you are in the current answer)
- recommended_action: "continue" (refine more), "refine_query" (need better retrieval), or "accept_answer" (good enough)

IMPORTANT: If previous_attempts >= 2, be more lenient and lean towards "accept_answer" unless there are critical gaps.
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=GapAnalysisResult,
                temperature=0.3
            )

            logger.info(
                f"Gap analysis: has_gaps={result.has_gaps}, "
                f"confidence={result.confidence_score:.2f}, "
                f"action={result.recommended_action}"
            )

            if result.has_gaps:
                logger.info(f"Identified {len(result.gaps)} gaps:")
                for gap in result.gaps:
                    logger.info(f"  - {gap.gap_type}: {gap.description} (priority={gap.priority})")

            return result

        except Exception as e:
            logger.error(f"Gap analysis error: {e}")
            # Default to accepting answer on error
            return GapAnalysisResult(
                has_gaps=False,
                gaps=[],
                overall_assessment=f"Analysis failed: {str(e)}",
                confidence_score=0.7,
                recommended_action="accept_answer"
            )


class MetaCognitiveRAGAgent:
    """
    Meta-cognitive RAG agent with recursive self-improvement

    Based on the MetaCognitiveRAGAgent pattern from research:
    - Conducts research with recursive self-improvement
    - Assesses quality at each iteration
    - Identifies improvement opportunities
    - Self-corrects and refines
    """

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        retriever=None,
        max_iterations: int = 3,
        improvement_threshold: float = 0.1
    ):
        """
        Initialize meta-cognitive RAG agent

        Args:
            llm_client: LLM client for generation
            retriever: Retrieval system
            max_iterations: Maximum recursive iterations
            improvement_threshold: Minimum improvement needed to continue
        """
        self.llm = llm_client or OpenRouterClient(model=settings.main_model)
        self.retriever = retriever
        self.gap_analyzer = GapAnalyzer(llm_client=llm_client)
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

        # Meta-cognitive tracking
        self.session_history = []
        self.quality_scores = []

    async def conduct_research(
        self,
        question: str,
        user_id: str,
        initial_context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Conduct research with recursive self-improvement

        Args:
            question: Research question
            user_id: User ID
            initial_context: Optional initial retrieved context

        Returns:
            {
                "answer": str,
                "documents": List[Dict],
                "iterations": int,
                "improvement_history": List[Dict],
                "final_confidence": float,
                "meta_insights": Dict
            }
        """
        logger.info(f"[META-COG] Starting research with max {self.max_iterations} iterations")
        start_time = time.time()

        iteration = 0
        current_answer = ""
        current_documents = initial_context or []
        improvement_history = []

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[META-COG] === Iteration {iteration}/{self.max_iterations} ===")

            iteration_start = time.time()

            # Step 1: Retrieve or use provided context
            if iteration == 1 and initial_context:
                # Use provided context
                logger.info("[META-COG] Using provided initial context")
            else:
                # Retrieve fresh context (potentially with refined query)
                if self.retriever:
                    try:
                        result = await self.retriever.retrieve(
                            query=question,
                            user_id=user_id,
                            top_k=15  # Get more for comprehensive answers
                        )
                        current_documents = result.get("chunks", [])
                        logger.info(f"[META-COG] Retrieved {len(current_documents)} documents")
                    except Exception as e:
                        logger.error(f"[META-COG] Retrieval error: {e}")
                        if iteration == 1:
                            return {
                                "answer": "I encountered an error retrieving information.",
                                "documents": [],
                                "iterations": iteration,
                                "improvement_history": [],
                                "final_confidence": 0.0,
                                "meta_insights": {"error": str(e)}
                            }

            # Step 2: Generate answer
            context_str = self._format_context(current_documents)
            current_answer = await self._generate_answer(question, context_str, iteration)

            # Step 3: Meta-cognitive gap analysis
            gap_analysis = await self.gap_analyzer.analyze_gaps(
                question=question,
                current_answer=current_answer,
                retrieved_context=context_str,
                previous_attempts=iteration - 1
            )

            iteration_time = (time.time() - iteration_start) * 1000

            # Track iteration
            iteration_info = {
                "iteration": iteration,
                "answer_length": len(current_answer),
                "confidence": gap_analysis.confidence_score,
                "has_gaps": gap_analysis.has_gaps,
                "num_gaps": len(gap_analysis.gaps),
                "action": gap_analysis.recommended_action,
                "time_ms": iteration_time
            }
            improvement_history.append(iteration_info)
            self.quality_scores.append(gap_analysis.confidence_score)

            logger.info(
                f"[META-COG] Iteration {iteration}: confidence={gap_analysis.confidence_score:.2f}, "
                f"gaps={len(gap_analysis.gaps)}, action={gap_analysis.recommended_action}"
            )

            # Step 4: Decide whether to continue
            if gap_analysis.recommended_action == "accept_answer":
                logger.info("[META-COG] Answer accepted, stopping iterations")
                break

            elif gap_analysis.recommended_action == "refine_query" and iteration < self.max_iterations:
                # Use highest priority gap to refine query
                if gap_analysis.gaps:
                    high_priority_gaps = [g for g in gap_analysis.gaps if g.priority >= 4]
                    if high_priority_gaps:
                        refined_query = high_priority_gaps[0].suggested_query
                        logger.info(f"[META-COG] Refining query: {refined_query}")
                        question = refined_query  # Update question for next iteration
                    else:
                        logger.info("[META-COG] No high-priority gaps, accepting answer")
                        break
                else:
                    break

            elif gap_analysis.recommended_action == "continue" and iteration < self.max_iterations:
                # Continue with more context or refinement
                logger.info("[META-COG] Continuing refinement...")
                # Could implement additional strategies here
                continue

            else:
                # Max iterations or uncertain, stop
                break

        total_time = (time.time() - start_time) * 1000

        # Calculate improvement metrics
        improvement_gain = 0.0
        if len(self.quality_scores) > 1:
            improvement_gain = self.quality_scores[-1] - self.quality_scores[0]

        meta_insights = {
            "total_iterations": iteration,
            "improvement_gain": improvement_gain,
            "avg_confidence": sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0,
            "final_confidence": gap_analysis.confidence_score,
            "total_time_ms": total_time,
            "converged": gap_analysis.recommended_action == "accept_answer"
        }

        logger.info(
            f"[META-COG] Research complete: {iteration} iterations, "
            f"improvement={improvement_gain:.3f}, "
            f"final_confidence={gap_analysis.confidence_score:.2f}"
        )

        return {
            "answer": current_answer,
            "documents": current_documents,
            "iterations": iteration,
            "improvement_history": improvement_history,
            "final_confidence": gap_analysis.confidence_score,
            "meta_insights": meta_insights
        }

    async def _generate_answer(
        self,
        question: str,
        context: str,
        iteration: int
    ) -> str:
        """
        Generate answer with meta-cognitive awareness

        Args:
            question: Question to answer
            context: Retrieved context
            iteration: Current iteration number

        Returns:
            Generated answer
        """
        system_prompt = f"""You are a banking operations expert conducting iterative research (iteration {iteration}).

CRITICAL REQUIREMENTS:
1. Provide comprehensive, thorough answers covering ALL relevant aspects
2. Structure complex answers with clear organization (numbered sections, bullet points)
3. Cover different workflows, scenarios, and use cases when applicable
4. Include both common cases and important variations
5. Be definitive and authoritative - speak as an expert

If this is iteration {iteration} > 1:
- Build on previous attempts
- Fill in gaps that were identified
- Provide more depth and detail
"""

        if context:
            user_message = f"""Available knowledge:
{context}

Question: {question}

Provide a comprehensive, well-structured answer covering all relevant aspects."""
        else:
            user_message = question

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            answer = await self.llm.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            return answer.strip()

        except Exception as e:
            logger.error(f"[META-COG] Generation error: {e}")
            return "I encountered an error generating an answer."

    def _format_context(self, documents: List[Dict]) -> str:
        """Format documents into context string"""
        if not documents:
            return ""

        formatted = []
        for i, doc in enumerate(documents[:20], 1):
            payload = doc.get("payload", {})
            text = payload.get("text", "")
            formatted.append(f"[Doc {i}]\n{text}")

        return "\n\n".join(formatted)

    def get_session_insights(self) -> Dict:
        """Get insights from the current session"""
        if not self.quality_scores:
            return {}

        return {
            "total_queries": len(self.quality_scores),
            "avg_confidence": sum(self.quality_scores) / len(self.quality_scores),
            "max_confidence": max(self.quality_scores),
            "min_confidence": min(self.quality_scores),
            "improvement_trend": self.quality_scores[-1] - self.quality_scores[0] if len(self.quality_scores) > 1 else 0.0
        }

    def reset_session(self):
        """Reset session tracking"""
        self.session_history = []
        self.quality_scores = []
        logger.info("[META-COG] Session reset")
