"""
Agentic RAG Workflow - Self-correcting adaptive RAG with LangGraph
Based on latest LangGraph adaptive RAG patterns from Context7

Key Features:
- Adaptive routing (vectorstore vs direct generation)
- Document relevance grading
- Query rewriting on poor retrieval
- Hallucination checking
- Answer quality validation
- Multi-step query planning
- Iterative refinement with retry logic
"""

from typing import List, Dict, Optional, Literal, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from src.agents.rag.graders import DocumentGrader, HallucinationGrader, AnswerGrader, AnswerDepthGrader
from src.agents.rag.routing import QueryRouter, RetrievalStrategyRouter
from src.agents.rag.tools import create_retriever_tool
from src.agents.rag.planner import QueryPlanner
from src.retrieval.retriever import RAGRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

# === Advanced Agentic RAG Components ===
from src.retrieval.multi_stage_pipeline import MultiStageRetrievalPipeline, AdaptiveMultiStagePipeline
from src.agents.rag.meta_cognitive_rag import MetaCognitiveRAGAgent, GapAnalyzer
from src.agents.rag.process_supervision import ProcessSupervisor, StageQualityMetrics

logger = get_logger(__name__)


# State Definition
class AgenticRAGState(TypedDict):
    """State for agentic RAG workflow"""
    # Input
    question: str
    user_id: str
    conversation_history: List[Dict]

    # Routing
    datasource: str  # "vectorstore" or "generate_direct"
    needs_planning: bool
    retrieval_strategy: str  # "simple", "multi_document", "multi_hop"

    # Query processing
    current_query: str
    reformulated_queries: Annotated[List[str], operator.add]  # Track all reformulations

    # Retrieval
    documents: List[Dict]
    all_documents: Annotated[List[Dict], operator.add]  # Accumulate from all attempts

    # Quality assessment
    documents_relevant: bool
    relevance_score: float
    answer_grounded: bool
    answer_useful: bool
    answer_comprehensive: bool

    # Generation
    generation: str
    final_answer: str

    # Control flow
    retry_count: int
    max_retries: int
    step: str  # Current step for logging
    error: Optional[str]


class AgenticRAGWorkflow:
    """
    Self-correcting adaptive RAG workflow using LangGraph

    Workflow:
    1. Route Query → Decide: vectorstore or direct generation
    2. Plan (if needed) → Break complex queries into steps
    3. Retrieve → Get documents from vectorstore
    4. Grade Documents → Check relevance
    5. If not relevant → Rewrite Query (retry)
    6. Generate Answer → Create response from documents
    7. Grade Answer → Check hallucination + usefulness
    8. If poor quality → Retry or fallback
    9. Return Final Answer
    """

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        retriever: Optional[RAGRetriever] = None
    ):
        """Initialize agentic RAG workflow with advanced components"""
        logger.info("Initializing AgenticRAGWorkflow with advanced features")

        # Core components
        self.llm = llm_client or OpenRouterClient(model=settings.main_model)
        self.base_retriever = retriever or RAGRetriever()

        # === Advanced Retrieval Pipeline ===
        if settings.use_multi_stage_pipeline:
            logger.info("Enabling multi-stage retrieval pipeline")
            if settings.enable_adaptive_routing:
                self.retriever = AdaptiveMultiStagePipeline(
                    base_retriever=self.base_retriever,
                    enable_query_enhancement=settings.multistage_enable_query_enhancement,
                    enable_hybrid_search=settings.multistage_enable_hybrid_search,
                    enable_reranking=settings.multistage_enable_reranking,
                    enable_compression=settings.multistage_enable_compression,
                    reranking_method=settings.reranking_method,
                    compression_method=settings.compression_method
                )
            else:
                self.retriever = MultiStageRetrievalPipeline(
                    base_retriever=self.base_retriever,
                    enable_query_enhancement=settings.multistage_enable_query_enhancement,
                    enable_hybrid_search=settings.multistage_enable_hybrid_search,
                    enable_reranking=settings.multistage_enable_reranking,
                    enable_compression=settings.multistage_enable_compression,
                    reranking_method=settings.reranking_method,
                    compression_method=settings.compression_method
                )
        else:
            logger.info("Using standard retriever")
            self.retriever = self.base_retriever

        # === Meta-Cognitive Components ===
        if settings.use_meta_cognitive_rag:
            logger.info("Enabling meta-cognitive RAG")
            self.meta_cognitive_agent = MetaCognitiveRAGAgent(
                llm_client=self.llm,
                retriever=self.retriever,
                max_iterations=settings.metacog_max_iterations,
                improvement_threshold=settings.metacog_improvement_threshold
            )
        else:
            self.meta_cognitive_agent = None

        # === Process Supervision ===
        if settings.use_process_supervision:
            logger.info("Enabling process supervision")
            self.supervisor = ProcessSupervisor(
                enable_fallbacks=settings.supervision_enable_fallbacks,
                track_metrics=settings.supervision_track_metrics
            )
        else:
            self.supervisor = None

        # Agentic components
        self.query_router = QueryRouter(llm_client=self.llm)
        self.strategy_router = RetrievalStrategyRouter(llm_client=self.llm)
        self.document_grader = DocumentGrader(llm_client=self.llm)
        self.hallucination_grader = HallucinationGrader(llm_client=self.llm)
        self.answer_grader = AnswerGrader(llm_client=self.llm)
        self.depth_grader = AnswerDepthGrader(llm_client=self.llm)
        self.query_rewriter = QueryRewriter(llm_client=self.llm)
        self.query_planner = QueryPlanner(llm_client=self.llm, retriever=self.base_retriever)

        # Gap analyzer for iterative refinement
        if settings.use_iterative_refinement:
            self.gap_analyzer = GapAnalyzer(llm_client=self.llm)
        else:
            self.gap_analyzer = None

        # Build workflow
        self.graph = self._build_workflow()

        # Metrics tracking
        self.workflow_metrics = []

        logger.info("AgenticRAGWorkflow initialized with advanced components")

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)

        # Add nodes
        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("plan_query", self._plan_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("grade_answer", self._grade_answer_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)

        # Set entry point
        workflow.set_entry_point("route_query")

        # Routing logic
        workflow.add_conditional_edges(
            "route_query",
            self._decide_after_routing,
            {
                "retrieve": "retrieve",
                "generate_direct": "generate",
                "plan": "plan_query"
            }
        )

        workflow.add_edge("plan_query", "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self._should_grade_documents,
            {
                "grade": "grade_documents",
                "generate": "generate"  # Skip grading if no docs
            }
        )

        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
                "end": END  # Give up after max retries
            }
        )

        workflow.add_edge("rewrite_query", "retrieve")

        workflow.add_conditional_edges(
            "generate",
            self._should_grade_answer,
            {
                "grade": "grade_answer",
                "end": END  # Skip grading for direct generation
            }
        )

        workflow.add_conditional_edges(
            "grade_answer",
            self._decide_after_answer_grading,
            {
                "end": END,
                "retry": "rewrite_query"
            }
        )

        return workflow.compile()

    # ============ Node Functions ============

    async def _route_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Route query to appropriate datasource"""
        logger.info(f"[ROUTE] Routing query: '{state['question'][:100]}...'")

        routing = await self.query_router.route_query(
            question=state["question"],
            conversation_history=state.get("conversation_history")
        )

        state["datasource"] = routing["datasource"]
        state["current_query"] = state["question"]
        state["step"] = "routed"

        logger.info(f"[ROUTE] Decision: {routing['datasource']} (confidence={routing['confidence']:.2f})")

        # Check if planning is needed
        if routing["datasource"] == "vectorstore":
            strategy = await self.strategy_router.determine_strategy(state["question"])
            state["retrieval_strategy"] = strategy["strategy"]
            state["needs_planning"] = strategy["requires_planning"]

            logger.info(
                f"[ROUTE] Strategy: {strategy['strategy']}, "
                f"planning_needed={strategy['requires_planning']}"
            )

        return state

    async def _plan_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Plan multi-step query execution"""
        logger.info("[PLAN] Creating query execution plan")

        plan_result = await self.query_planner.create_plan(
            query=state["question"],
            conversation_history=state.get("conversation_history")
        )

        if plan_result["needs_planning"] and plan_result["plan"]:
            # Execute plan
            plan = plan_result["plan"]
            execution_result = await self.query_planner.execute_plan(
                plan=plan,
                user_id=state["user_id"],
                conversation_history=state.get("conversation_history")
            )

            # Store all retrieved chunks
            state["documents"] = execution_result["all_chunks"]
            state["step"] = "planned_and_retrieved"

            logger.info(
                f"[PLAN] Executed {len(plan.items)} steps, "
                f"retrieved {len(execution_result['all_chunks'])} total chunks"
            )
        else:
            # Planning not needed or failed
            state["step"] = "plan_skipped"
            logger.info("[PLAN] Planning skipped or failed, proceeding with simple retrieval")

        return state

    async def _retrieve_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve documents using multi-stage pipeline"""
        query = state["current_query"]
        logger.info(f"[RETRIEVE] Query: '{query[:100]}...'")

        # Skip if already retrieved by planner
        if state.get("step") == "planned_and_retrieved":
            logger.info("[RETRIEVE] Using documents from planner")
            return state

        # Use process supervision if enabled
        if self.supervisor:
            async def retrieval_func():
                return await self._execute_retrieval(state)

            supervised_result = await self.supervisor.monitor_stage(
                stage_name="retrieval",
                execution_func=retrieval_func
            )

            if supervised_result["success"]:
                result = supervised_result["result"]
                state["documents"] = result.get("chunks", [])
                state["step"] = "retrieved"
            else:
                logger.error("[RETRIEVE] Supervised retrieval failed")
                state["documents"] = []
                state["step"] = "retrieval_failed"

        else:
            # Direct retrieval without supervision
            result = await self._execute_retrieval(state)
            state["documents"] = result.get("chunks", [])
            state["step"] = "retrieved"

        logger.info(f"[RETRIEVE] Retrieved {len(state['documents'])} documents")

        return state

    async def _execute_retrieval(self, state: AgenticRAGState) -> Dict:
        """Execute retrieval with appropriate retriever"""
        query = state["current_query"]
        conv_history = state.get("conversation_history")

        # Check if using multi-stage pipeline
        if isinstance(self.retriever, (MultiStageRetrievalPipeline, AdaptiveMultiStagePipeline)):
            # Multi-stage pipeline
            result = await self.retriever.retrieve(
                query=query,
                user_id=state["user_id"],
                conversation_history=conv_history
            )
            return result
        else:
            # Standard retriever
            if conv_history:
                result = await self.retriever.retrieve_with_context(
                    query=query,
                    user_id=state["user_id"],
                    conversation_history=conv_history
                )
            else:
                result = await self.retriever.retrieve(
                    query=query,
                    user_id=state["user_id"]
                )
            return result

    async def _grade_documents_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Grade retrieved documents for relevance"""
        logger.info(f"[GRADE DOCS] Grading {len(state['documents'])} documents")

        grading_result = await self.document_grader.grade_documents(
            question=state["question"],
            documents=state["documents"]
        )

        state["documents_relevant"] = grading_result["is_relevant"]
        state["relevance_score"] = grading_result["relevance_score"]

        # Replace documents with only relevant ones
        if grading_result["relevant_documents"]:
            state["documents"] = grading_result["relevant_documents"]
            logger.info(
                f"[GRADE DOCS] {len(state['documents'])} relevant documents "
                f"(score={state['relevance_score']:.2f})"
            )
        else:
            logger.warning("[GRADE DOCS] No relevant documents found")

        state["step"] = "graded"

        return state

    async def _generate_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate answer"""
        logger.info("[GENERATE] Generating answer")

        question = state["question"]
        documents = state.get("documents", [])
        conv_history = state.get("conversation_history", [])

        # Format context from documents
        context = self._format_context(documents)

        # Build messages
        system_prompt = """You are a banking operations expert. Answer questions directly and confidently.

CRITICAL RULES:
1. Answer using ONLY the provided information
2. Be definitive and authoritative - you are the expert
3. NEVER mention "context", "documents", or "provided information"
4. Synthesize information to give complete answers
5. If you don't have information, simply say "I don't have information about that"

COMPREHENSIVENESS REQUIREMENTS:
- Provide thorough, comprehensive explanations covering ALL relevant aspects
- When explaining features or concepts, describe different workflows, scenarios, and use cases
- Cover both common cases and important variations or edge cases
- If a topic has multiple types or variations (e.g., different workflows, different processing modes), explain each one
- When relevant, explain how features work in different contexts (e.g., bulk vs single, sequential vs non-sequential)
- Aim for complete understanding, not just surface-level definitions
- Structure complex answers with clear organization (numbered sections, bullet points)

Example of COMPREHENSIVE answer:
"The Authorization Matrix is a validation mechanism that... It supports two workflow types:
1. Sequential Workflow: where approvals happen in order...
2. Non-Sequential Workflow: where approvals can happen in any order...
For bulk payments specifically, there are 3 authorization options:...
Additionally, self-authorization is supported when..."

Example of SHALLOW answer (AVOID):
"The Authorization Matrix is a validation mechanism for approvals."
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conv_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")[:500]
                messages.append({"role": msg["role"], "content": content})

        # Add current query with context
        if context:
            user_message = f"""Available knowledge:
{context}

---

Question: {question}

Provide a clear, confident answer based on the available information."""
        else:
            user_message = question

        messages.append({"role": "user", "content": user_message})

        # Generate
        try:
            answer = await self.llm.chat(
                messages=messages,
                temperature=settings.temperature,
                stream=False
            )

            state["generation"] = answer.strip()
            state["step"] = "generated"

            logger.info(f"[GENERATE] Answer generated ({len(answer)} chars)")

        except Exception as e:
            logger.error(f"[GENERATE] Error: {e}")
            state["generation"] = "I encountered an error generating an answer."
            state["error"] = str(e)
            state["step"] = "generation_error"

        return state

    async def _grade_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Grade answer quality"""
        logger.info("[GRADE ANSWER] Checking answer quality")

        # Check hallucination
        hallucination_result = await self.hallucination_grader.grade_hallucination(
            documents=state.get("documents", []),
            generation=state["generation"]
        )

        state["answer_grounded"] = hallucination_result["is_grounded"]

        # Check if answer addresses question
        answer_result = await self.answer_grader.grade_answer(
            question=state["question"],
            generation=state["generation"]
        )

        state["answer_useful"] = answer_result["is_useful"]

        # Check answer depth and comprehensiveness
        depth_result = await self.depth_grader.grade_depth(
            question=state["question"],
            answer=state["generation"],
            documents=state.get("documents", [])
        )

        state["answer_comprehensive"] = depth_result["is_comprehensive"]

        logger.info(
            f"[GRADE ANSWER] Grounded: {state['answer_grounded']}, "
            f"Useful: {state['answer_useful']}, "
            f"Comprehensive: {state['answer_comprehensive']}"
        )

        if not depth_result["is_comprehensive"]:
            logger.info(f"[GRADE ANSWER] Depth suggestions: {depth_result['suggestions']}")

        state["step"] = "answer_graded"

        return state

    async def _rewrite_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Rewrite query for better retrieval"""
        logger.info(f"[REWRITE] Rewriting query (attempt {state['retry_count'] + 1})")

        # Rewrite
        rewritten = await self.query_rewriter.rewrite_query(state["current_query"])

        state["current_query"] = rewritten
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["step"] = "rewritten"

        logger.info(f"[REWRITE] New query: '{rewritten[:100]}...'")

        return state

    # ============ Decision Functions ============

    def _decide_after_routing(self, state: AgenticRAGState) -> str:
        """Decide next step after routing"""
        if state["datasource"] == "generate_direct":
            return "generate_direct"
        elif state.get("needs_planning", False):
            return "plan"
        else:
            return "retrieve"

    def _should_grade_documents(self, state: AgenticRAGState) -> str:
        """Decide whether to grade documents"""
        if not state.get("documents"):
            logger.info("[DECISION] No documents to grade, skipping to generation")
            return "generate"
        return "grade"

    def _decide_after_grading(self, state: AgenticRAGState) -> str:
        """Decide next step after document grading"""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        if state["documents_relevant"]:
            return "generate"
        elif retry_count >= max_retries:
            logger.warning(f"[DECISION] Max retries ({max_retries}) reached, proceeding with best effort")
            return "generate"  # Changed from "end" to "generate" - try anyway
        else:
            logger.info(f"[DECISION] Documents not relevant, rewriting query (retry {retry_count + 1}/{max_retries})")
            return "rewrite"

    def _should_grade_answer(self, state: AgenticRAGState) -> str:
        """Decide whether to grade answer"""
        # Skip grading for direct generation
        if state.get("datasource") == "generate_direct":
            state["final_answer"] = state["generation"]
            return "end"
        return "grade"

    def _decide_after_answer_grading(self, state: AgenticRAGState) -> str:
        """Decide next step after answer grading"""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        # Check all quality criteria: grounded, useful, and comprehensive
        is_grounded = state["answer_grounded"]
        is_useful = state["answer_useful"]
        is_comprehensive = state.get("answer_comprehensive", True)  # Default to True for backwards compatibility

        if is_grounded and is_useful and is_comprehensive:
            state["final_answer"] = state["generation"]
            return "end"
        elif retry_count >= max_retries:
            logger.warning("[DECISION] Max retries reached, using current answer")
            state["final_answer"] = state["generation"]
            return "end"
        else:
            if not is_comprehensive:
                logger.info("[DECISION] Answer lacks depth/comprehensiveness, retrying")
            else:
                logger.info("[DECISION] Answer quality poor (grounded/useful), retrying")
            return "retry"

    # ============ Helper Methods ============

    def _format_context(self, documents: List[Dict]) -> str:
        """Format documents into context string"""
        if not documents:
            return ""

        formatted = []
        for i, doc in enumerate(documents[:30], 1):  # Increased from 5 to 30 for comprehensive coverage
            payload = doc.get("payload", {})
            text = payload.get("text", "")
            formatted.append(f"[Document {i}]\n{text}")

        return "\n\n".join(formatted)

    # ============ Public Interface ============

    async def run_meta_cognitive(
        self,
        question: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Run meta-cognitive RAG with recursive self-improvement

        Args:
            question: User's question
            user_id: User ID
            conversation_history: Optional conversation context

        Returns:
            Enhanced result with iteration history and confidence scores
        """
        if not self.meta_cognitive_agent:
            logger.warning("Meta-cognitive agent not enabled, falling back to standard workflow")
            return await self.run(question, user_id, conversation_history)

        logger.info("[WORKFLOW] Running meta-cognitive RAG workflow")

        # Use meta-cognitive agent
        result = await self.meta_cognitive_agent.conduct_research(
            question=question,
            user_id=user_id,
            initial_context=None  # Let it retrieve fresh
        )

        return {
            "answer": result["answer"],
            "documents": result["documents"],
            "metadata": {
                "mode": "meta_cognitive",
                "iterations": result["iterations"],
                "improvement_history": result["improvement_history"],
                "final_confidence": result["final_confidence"],
                "meta_insights": result["meta_insights"]
            }
        }

    async def run(
        self,
        question: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        max_retries: int = 2,
        use_meta_cognitive: Optional[bool] = None
    ) -> Dict:
        """
        Run agentic RAG workflow

        Args:
            question: User's question
            user_id: User ID
            conversation_history: Optional conversation context
            max_retries: Maximum retry attempts
            use_meta_cognitive: Override to use meta-cognitive mode

        Returns:
            {
                "answer": str,
                "documents": List[Dict],
                "metadata": Dict (routing, grading results, etc.)
            }
        """
        # Choose workflow mode
        should_use_metacog = use_meta_cognitive if use_meta_cognitive is not None else settings.use_meta_cognitive_rag

        if should_use_metacog and self.meta_cognitive_agent:
            logger.info("[WORKFLOW] Using meta-cognitive workflow")
            return await self.run_meta_cognitive(question, user_id, conversation_history)

        logger.info(f"[WORKFLOW] Starting standard agentic RAG: '{question[:100]}...'")

        # Track workflow metrics
        import time
        workflow_start_time = time.time()

        # Initialize state
        initial_state = AgenticRAGState(
            question=question,
            user_id=user_id,
            conversation_history=conversation_history or [],
            datasource="",
            needs_planning=False,
            retrieval_strategy="simple",
            current_query=question,
            reformulated_queries=[],
            documents=[],
            all_documents=[],
            documents_relevant=False,
            relevance_score=0.0,
            answer_grounded=False,
            answer_useful=False,
            generation="",
            final_answer="",
            retry_count=0,
            max_retries=max_retries,
            step="init",
            error=None
        )

        # Run workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)

            # Calculate workflow metrics
            workflow_duration = (time.time() - workflow_start_time) * 1000

            logger.info(
                f"[WORKFLOW] Complete: "
                f"datasource={final_state.get('datasource')}, "
                f"retries={final_state.get('retry_count')}, "
                f"docs={len(final_state.get('documents', []))}, "
                f"time={workflow_duration:.0f}ms"
            )

            # Build comprehensive metadata
            metadata = {
                "mode": "standard_agentic",
                "datasource": final_state.get("datasource"),
                "retrieval_strategy": final_state.get("retrieval_strategy"),
                "retry_count": final_state.get("retry_count"),
                "documents_relevant": final_state.get("documents_relevant"),
                "relevance_score": final_state.get("relevance_score"),
                "answer_grounded": final_state.get("answer_grounded"),
                "answer_useful": final_state.get("answer_useful"),
                "answer_comprehensive": final_state.get("answer_comprehensive"),
                "reformulated_queries": final_state.get("reformulated_queries", []),
                "workflow_duration_ms": workflow_duration,
                "timestamp": time.time()
            }

            # Add multi-stage pipeline metrics if available
            if isinstance(self.retriever, (MultiStageRetrievalPipeline, AdaptiveMultiStagePipeline)):
                pipeline_metrics = self.retriever.get_metrics()
                metadata["pipeline_metrics"] = pipeline_metrics

            # Add supervision metrics if available
            if self.supervisor and settings.track_rag_metrics:
                supervision_metrics = self.supervisor.get_metrics_summary()
                metadata["supervision_metrics"] = supervision_metrics

            # Track workflow execution
            if settings.track_rag_metrics:
                self.workflow_metrics.append(metadata)
                # Keep only recent metrics
                if len(self.workflow_metrics) > settings.metrics_window_size:
                    self.workflow_metrics = self.workflow_metrics[-settings.metrics_window_size:]

            return {
                "answer": final_state.get("final_answer", final_state.get("generation", "")),
                "documents": final_state.get("documents", []),
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"[WORKFLOW] Error: {e}", exc_info=True)
            workflow_duration = (time.time() - workflow_start_time) * 1000
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "documents": [],
                "metadata": {
                    "error": str(e),
                    "workflow_duration_ms": workflow_duration
                }
            }

    def get_workflow_metrics(self) -> Dict:
        """Get comprehensive workflow metrics"""
        if not self.workflow_metrics:
            return {}

        total_queries = len(self.workflow_metrics)
        avg_duration = sum(m.get("workflow_duration_ms", 0) for m in self.workflow_metrics) / total_queries

        # Calculate success rates
        successful_queries = sum(1 for m in self.workflow_metrics if m.get("answer_useful", False))
        grounded_answers = sum(1 for m in self.workflow_metrics if m.get("answer_grounded", False))
        comprehensive_answers = sum(1 for m in self.workflow_metrics if m.get("answer_comprehensive", False))

        return {
            "total_queries": total_queries,
            "avg_duration_ms": avg_duration,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0.0,
            "grounded_rate": grounded_answers / total_queries if total_queries > 0 else 0.0,
            "comprehensive_rate": comprehensive_answers / total_queries if total_queries > 0 else 0.0,
            "avg_retries": sum(m.get("retry_count", 0) for m in self.workflow_metrics) / total_queries if total_queries > 0 else 0.0,
            "pipeline_metrics": self.retriever.get_metrics() if isinstance(self.retriever, (MultiStageRetrievalPipeline, AdaptiveMultiStagePipeline)) else {},
            "supervision_metrics": self.supervisor.get_metrics_summary() if self.supervisor else {}
        }
