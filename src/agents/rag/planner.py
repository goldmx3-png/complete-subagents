"""
Query Planner for Multi-Step RAG
Decomposes complex queries into sub-queries and aggregates results
Based on LlamaIndex QueryPlanningWorkflow pattern from Context7
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.llm.openrouter_client import OpenRouterClient
from src.retrieval.retriever import RAGRetriever
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryPlanItem(BaseModel):
    """A single step in a query execution plan"""
    step_number: int = Field(description="Step number in the plan")
    query: str = Field(description="Natural language search query for this step")
    purpose: str = Field(description="What this step aims to accomplish")
    depends_on: List[int] = Field(
        default=[],
        description="List of step numbers this step depends on"
    )


class QueryPlan(BaseModel):
    """Complete execution plan for a complex query"""
    items: List[QueryPlanItem] = Field(
        description="List of QueryPlanItem objects in execution order"
    )
    final_synthesis_needed: bool = Field(
        description="Whether results need to be synthesized at the end"
    )
    reasoning: str = Field(
        description="Explanation of the planning strategy"
    )


class QueryPlanner:
    """
    Plans and executes multi-step queries

    Based on LlamaIndex QueryPlanningWorkflow:
    - Breaks down complex queries into sub-queries
    - Executes sub-queries in sequence or parallel
    - Aggregates results intelligently
    - Handles dependencies between steps
    """

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        retriever: Optional[RAGRetriever] = None
    ):
        self.llm = llm_client or OpenRouterClient(model=settings.main_model)
        self.retriever = retriever or RAGRetriever()

    async def create_plan(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Create an execution plan for a complex query

        Args:
            query: Complex user query
            conversation_history: Optional conversation context

        Returns:
            {
                "plan": QueryPlan,
                "needs_planning": bool,
                "is_complex": bool
            }
        """
        logger.info(f"Creating query plan for: '{query[:100]}...'")

        # First, determine if planning is needed
        needs_planning = await self._needs_planning(query)

        if not needs_planning:
            logger.info("Query is simple, no planning needed")
            return {
                "plan": None,
                "needs_planning": False,
                "is_complex": False
            }

        # Create the plan
        prompt = self._build_planning_prompt(query, conversation_history)

        try:
            plan = await self.llm.structured_output(
                prompt=prompt,
                response_model=QueryPlan,
                temperature=0
            )

            logger.info(
                f"Query plan created: {len(plan.items)} steps, "
                f"synthesis_needed={plan.final_synthesis_needed}"
            )

            for item in plan.items:
                logger.debug(
                    f"Step {item.step_number}: {item.query} "
                    f"(depends on: {item.depends_on})"
                )

            return {
                "plan": plan,
                "needs_planning": True,
                "is_complex": True
            }

        except Exception as e:
            logger.error(f"Query planning error: {e}")
            # Fall back to no planning
            return {
                "plan": None,
                "needs_planning": False,
                "is_complex": False,
                "error": str(e)
            }

    async def execute_plan(
        self,
        plan: QueryPlan,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Execute a query plan

        Args:
            plan: QueryPlan to execute
            user_id: User ID
            conversation_history: Optional conversation context

        Returns:
            {
                "step_results": List[Dict],
                "all_chunks": List[Dict],
                "synthesis_context": str
            }
        """
        logger.info(f"Executing query plan with {len(plan.items)} steps")

        step_results = []
        all_chunks = []

        # Execute each step in order
        for item in plan.items:
            logger.info(f"Executing step {item.step_number}: {item.query}")

            # Check dependencies
            if item.depends_on:
                logger.debug(f"Step {item.step_number} depends on steps: {item.depends_on}")

            # Execute retrieval for this step
            try:
                result = await self.retriever.retrieve(
                    query=item.query,
                    user_id=user_id,
                    top_k=10  # Increased from 3 for more comprehensive coverage
                )

                chunks = result.get("chunks", [])
                logger.info(f"Step {item.step_number}: Retrieved {len(chunks)} chunks")

                step_results.append({
                    "step_number": item.step_number,
                    "query": item.query,
                    "purpose": item.purpose,
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                })

                # Collect all chunks
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Step {item.step_number} failed: {e}")
                step_results.append({
                    "step_number": item.step_number,
                    "query": item.query,
                    "purpose": item.purpose,
                    "chunks": [],
                    "chunk_count": 0,
                    "error": str(e)
                })

        # Create synthesis context
        synthesis_context = self._create_synthesis_context(
            step_results=step_results,
            plan=plan
        )

        logger.info(
            f"Plan execution complete: {len(step_results)} steps, "
            f"{len(all_chunks)} total chunks"
        )

        return {
            "step_results": step_results,
            "all_chunks": all_chunks,
            "synthesis_context": synthesis_context
        }

    async def _needs_planning(self, query: str) -> bool:
        """
        Determine if query needs multi-step planning

        Uses heuristics to quickly assess complexity
        """
        query_lower = query.lower()

        # Keywords indicating complexity
        complexity_indicators = [
            # Comparison
            "compare", "difference", "versus", "vs", "better", "worse",
            "similarities", "contrast",

            # Multiple parts
            " and ", " also ", " as well as ",

            # Aggregation
            "all", "every", "each", "list all", "what are the",

            # Multi-step
            "then", "after that", "first", "second", "finally",
            "steps", "process", "how to",

            # Complex reasoning
            "why", "explain how", "what causes", "relationship between"
        ]

        # Count indicators
        indicator_count = sum(
            1 for indicator in complexity_indicators
            if indicator in query_lower
        )

        # Also check length - very long queries often need planning
        is_long = len(query.split()) > 20

        needs_planning = indicator_count >= 2 or (indicator_count >= 1 and is_long)

        logger.info(
            f"Planning assessment: indicators={indicator_count}, "
            f"is_long={is_long}, needs_planning={needs_planning}"
        )

        return needs_planning

    def _build_planning_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for query planning"""

        prompt = f"""You are a query planning expert for a RAG system. Your task is to break down complex queries into executable steps.

Available data sources:
- Banking operations documentation
- Policy documents and procedures
- Technical specifications
- Transaction processing rules

Original query: {query}

"""

        if conversation_history:
            recent = conversation_history[-4:]
            context_lines = []
            for msg in recent:
                role = msg.get("role", "")
                content = msg.get("content", "")[:100]
                if role in ["user", "assistant"]:
                    context_lines.append(f"{role}: {content}")
            if context_lines:
                prompt += f"""Recent conversation:
{chr(10).join(context_lines)}

"""

        prompt += """Create a step-by-step plan to answer this query.

Guidelines:
1. Break the query into logical sub-queries
2. Each step should retrieve specific information
3. Later steps can build on earlier results
4. Keep steps focused and clear
5. If query involves comparison, create separate steps for each item
6. If query has multiple parts, address each part

Provide:
- items: List of QueryPlanItem objects with:
  - step_number: Sequential number
  - query: Specific search query for this step
  - purpose: What this step accomplishes
  - depends_on: List of step numbers this depends on (empty if independent)
- final_synthesis_needed: true if results need combining/comparison
- reasoning: Brief explanation of planning strategy

Example for "Compare transaction limits for USD and EUR":
```json
{
  "items": [
    {
      "step_number": 1,
      "query": "transaction limits for USD currency",
      "purpose": "Get USD transaction limits",
      "depends_on": []
    },
    {
      "step_number": 2,
      "query": "transaction limits for EUR currency",
      "purpose": "Get EUR transaction limits",
      "depends_on": []
    }
  ],
  "final_synthesis_needed": true,
  "reasoning": "Query requires comparison, so retrieve information for each currency separately, then compare"
}
```
"""

        return prompt

    def _create_synthesis_context(
        self,
        step_results: List[Dict],
        plan: QueryPlan
    ) -> str:
        """
        Create context string for final synthesis

        Organizes step results for easy LLM consumption
        """
        context_parts = []

        context_parts.append(f"Multi-step query execution results:")
        context_parts.append(f"Total steps: {len(step_results)}")
        context_parts.append("")

        for step_result in step_results:
            step_num = step_result["step_number"]
            query = step_result["query"]
            purpose = step_result["purpose"]
            chunks = step_result["chunks"]

            context_parts.append(f"--- Step {step_num} ---")
            context_parts.append(f"Query: {query}")
            context_parts.append(f"Purpose: {purpose}")
            context_parts.append(f"Results found: {len(chunks)}")
            context_parts.append("")

            # Add actual content from chunks
            if chunks:
                for i, chunk in enumerate(chunks[:2], 1):  # Top 2 chunks per step
                    payload = chunk.get("payload", {})
                    text = payload.get("text", "")[:300]
                    context_parts.append(f"Result {i}: {text}...")
                context_parts.append("")

        if plan.final_synthesis_needed:
            context_parts.append(
                "Note: Synthesize and compare/combine these results to answer the original query."
            )

        return "\n".join(context_parts)
