"""
Process Supervision for Agentic RAG
Monitors and evaluates RAG pipeline quality at each step

Based on RAG-Gym paper: "Optimizing Reasoning and Search Agents with Process Supervision"
Key features:
- Quality checkpoints at each stage
- Fallback mechanisms on failures
- Performance metrics tracking
- Real-time monitoring and alerting
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StageQualityMetrics(BaseModel):
    """Quality metrics for a pipeline stage"""
    stage_name: str = Field(description="Name of the pipeline stage")
    success: bool = Field(description="Whether stage executed successfully")
    duration_ms: float = Field(description="Execution time in milliseconds")
    quality_score: float = Field(description="Quality score 0-1")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict = Field(default_factory=dict, description="Stage-specific metadata")


class PipelineQualityReport(BaseModel):
    """Comprehensive quality report for entire pipeline"""
    overall_success: bool = Field(description="Overall pipeline success")
    overall_quality: float = Field(description="Overall quality score 0-1")
    total_duration_ms: float = Field(description="Total execution time")
    stages: List[StageQualityMetrics] = Field(description="Metrics for each stage")
    warnings: List[str] = Field(default_factory=list, description="Warnings encountered")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


@dataclass
class QualityCheckpoint:
    """Quality checkpoint configuration"""
    name: str
    min_quality_score: float = 0.5
    required: bool = True
    fallback_strategy: Optional[str] = None
    retry_on_failure: bool = False
    max_retries: int = 2


class ProcessSupervisor:
    """
    Supervises RAG pipeline execution with quality checkpoints

    Features:
    - Stage-by-stage quality monitoring
    - Automatic fallback on failures
    - Performance tracking
    - Anomaly detection
    """

    def __init__(
        self,
        checkpoints: Optional[List[QualityCheckpoint]] = None,
        enable_fallbacks: bool = True,
        track_metrics: bool = True
    ):
        """
        Initialize process supervisor

        Args:
            checkpoints: Quality checkpoints to monitor
            enable_fallbacks: Enable automatic fallbacks
            track_metrics: Track performance metrics
        """
        # Default checkpoints for RAG pipeline
        self.checkpoints = checkpoints or self._default_checkpoints()
        self.enable_fallbacks = enable_fallbacks
        self.track_metrics = track_metrics

        # Metrics tracking
        self.stage_history = []
        self.failure_counts = {}
        self.avg_durations = {}

        logger.info(f"ProcessSupervisor initialized with {len(self.checkpoints)} checkpoints")

    def _default_checkpoints(self) -> List[QualityCheckpoint]:
        """Default quality checkpoints for RAG pipeline"""
        return [
            QualityCheckpoint(
                name="query_enhancement",
                min_quality_score=0.6,
                required=False,
                fallback_strategy="skip",
                retry_on_failure=False
            ),
            QualityCheckpoint(
                name="retrieval",
                min_quality_score=0.5,
                required=True,
                fallback_strategy="broad_search",
                retry_on_failure=True,
                max_retries=2
            ),
            QualityCheckpoint(
                name="document_grading",
                min_quality_score=0.4,
                required=False,
                fallback_strategy="accept_all",
                retry_on_failure=False
            ),
            QualityCheckpoint(
                name="reranking",
                min_quality_score=0.5,
                required=False,
                fallback_strategy="skip",
                retry_on_failure=False
            ),
            QualityCheckpoint(
                name="generation",
                min_quality_score=0.6,
                required=True,
                fallback_strategy="retry",
                retry_on_failure=True,
                max_retries=1
            ),
            QualityCheckpoint(
                name="answer_grading",
                min_quality_score=0.5,
                required=True,
                fallback_strategy="accept",
                retry_on_failure=False
            )
        ]

    async def monitor_stage(
        self,
        stage_name: str,
        execution_func,
        *args,
        quality_evaluator=None,
        **kwargs
    ) -> Dict:
        """
        Monitor execution of a pipeline stage

        Args:
            stage_name: Name of the stage
            execution_func: Function to execute
            *args, **kwargs: Arguments for execution_func
            quality_evaluator: Optional function to evaluate quality

        Returns:
            {
                "success": bool,
                "result": Any,
                "quality_score": float,
                "duration_ms": float,
                "metrics": StageQualityMetrics,
                "fallback_used": bool
            }
        """
        checkpoint = self._get_checkpoint(stage_name)

        if not checkpoint:
            # No checkpoint defined, execute without monitoring
            start_time = time.time()
            result = await execution_func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            return {
                "success": True,
                "result": result,
                "quality_score": 1.0,
                "duration_ms": duration,
                "metrics": None,
                "fallback_used": False
            }

        logger.info(f"[SUPERVISOR] Monitoring stage: {stage_name}")

        # Track attempts for retry
        attempt = 0
        max_attempts = checkpoint.max_retries + 1 if checkpoint.retry_on_failure else 1
        last_error = None

        while attempt < max_attempts:
            attempt += 1
            start_time = time.time()

            try:
                # Execute stage
                result = await execution_func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000

                # Evaluate quality
                if quality_evaluator:
                    quality_score = await quality_evaluator(result)
                else:
                    quality_score = self._default_quality_score(stage_name, result)

                # Check if quality meets threshold
                meets_threshold = quality_score >= checkpoint.min_quality_score

                if meets_threshold or attempt >= max_attempts:
                    # Success or final attempt
                    metrics = StageQualityMetrics(
                        stage_name=stage_name,
                        success=True,
                        duration_ms=duration,
                        quality_score=quality_score,
                        metadata={
                            "attempts": attempt,
                            "meets_threshold": meets_threshold
                        }
                    )

                    self._track_stage(metrics)

                    if not meets_threshold and checkpoint.required:
                        logger.warning(
                            f"[SUPERVISOR] Stage '{stage_name}' quality below threshold "
                            f"({quality_score:.2f} < {checkpoint.min_quality_score:.2f})"
                        )

                    return {
                        "success": True,
                        "result": result,
                        "quality_score": quality_score,
                        "duration_ms": duration,
                        "metrics": metrics,
                        "fallback_used": False
                    }
                else:
                    # Quality below threshold, retry
                    logger.warning(
                        f"[SUPERVISOR] Stage '{stage_name}' attempt {attempt}/{max_attempts}: "
                        f"quality {quality_score:.2f} < {checkpoint.min_quality_score:.2f}, retrying..."
                    )
                    continue

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                last_error = str(e)
                logger.error(
                    f"[SUPERVISOR] Stage '{stage_name}' attempt {attempt}/{max_attempts} failed: {e}"
                )

                if attempt >= max_attempts:
                    # All attempts failed
                    break

        # All attempts failed or quality never met threshold
        logger.error(f"[SUPERVISOR] Stage '{stage_name}' failed after {attempt} attempts")

        # Try fallback if enabled
        if self.enable_fallbacks and checkpoint.fallback_strategy:
            fallback_result = await self._execute_fallback(
                stage_name=stage_name,
                checkpoint=checkpoint,
                last_error=last_error,
                *args,
                **kwargs
            )
            if fallback_result["success"]:
                return fallback_result

        # Record failure
        metrics = StageQualityMetrics(
            stage_name=stage_name,
            success=False,
            duration_ms=duration,
            quality_score=0.0,
            error_message=last_error,
            metadata={"attempts": attempt}
        )

        self._track_stage(metrics)

        return {
            "success": False,
            "result": None,
            "quality_score": 0.0,
            "duration_ms": duration,
            "metrics": metrics,
            "fallback_used": False
        }

    async def _execute_fallback(
        self,
        stage_name: str,
        checkpoint: QualityCheckpoint,
        last_error: Optional[str],
        *args,
        **kwargs
    ) -> Dict:
        """Execute fallback strategy"""
        strategy = checkpoint.fallback_strategy
        logger.info(f"[SUPERVISOR] Executing fallback strategy: {strategy}")

        if strategy == "skip":
            # Skip this stage entirely
            return {
                "success": True,
                "result": None,
                "quality_score": 0.5,
                "duration_ms": 0.0,
                "metrics": StageQualityMetrics(
                    stage_name=stage_name,
                    success=True,
                    duration_ms=0.0,
                    quality_score=0.5,
                    metadata={"fallback": "skipped"}
                ),
                "fallback_used": True
            }

        elif strategy == "accept":
            # Accept current state
            return {
                "success": True,
                "result": kwargs.get("current_result"),
                "quality_score": 0.5,
                "duration_ms": 0.0,
                "metrics": StageQualityMetrics(
                    stage_name=stage_name,
                    success=True,
                    duration_ms=0.0,
                    quality_score=0.5,
                    metadata={"fallback": "accepted"}
                ),
                "fallback_used": True
            }

        elif strategy == "accept_all":
            # Accept all inputs without filtering
            return {
                "success": True,
                "result": kwargs.get("documents", []),
                "quality_score": 0.5,
                "duration_ms": 0.0,
                "metrics": StageQualityMetrics(
                    stage_name=stage_name,
                    success=True,
                    duration_ms=0.0,
                    quality_score=0.5,
                    metadata={"fallback": "accept_all"}
                ),
                "fallback_used": True
            }

        else:
            # Unknown strategy
            logger.warning(f"Unknown fallback strategy: {strategy}")
            return {"success": False, "fallback_used": False}

    def _get_checkpoint(self, stage_name: str) -> Optional[QualityCheckpoint]:
        """Get checkpoint for stage"""
        for checkpoint in self.checkpoints:
            if checkpoint.name == stage_name:
                return checkpoint
        return None

    def _default_quality_score(self, stage_name: str, result) -> float:
        """Default quality scoring based on result"""
        if result is None:
            return 0.0

        if isinstance(result, dict):
            # Check for common quality indicators
            if "chunks" in result:
                num_chunks = len(result["chunks"])
                return min(1.0, num_chunks / 5.0)  # 5+ chunks = quality 1.0

            if "score" in result:
                return result["score"]

            if "quality_score" in result:
                return result["quality_score"]

        # Default to moderate quality
        return 0.7

    def _track_stage(self, metrics: StageQualityMetrics):
        """Track stage metrics"""
        if not self.track_metrics:
            return

        self.stage_history.append(metrics)

        # Update failure counts
        if not metrics.success:
            self.failure_counts[metrics.stage_name] = self.failure_counts.get(metrics.stage_name, 0) + 1

        # Update average durations
        if metrics.stage_name not in self.avg_durations:
            self.avg_durations[metrics.stage_name] = []

        self.avg_durations[metrics.stage_name].append(metrics.duration_ms)

        # Keep only recent history (last 100 entries per stage)
        if len(self.avg_durations[metrics.stage_name]) > 100:
            self.avg_durations[metrics.stage_name] = self.avg_durations[metrics.stage_name][-100:]

    def generate_report(self, stages: List[StageQualityMetrics]) -> PipelineQualityReport:
        """
        Generate comprehensive quality report

        Args:
            stages: Metrics from all stages

        Returns:
            PipelineQualityReport
        """
        overall_success = all(stage.success for stage in stages)
        overall_quality = sum(stage.quality_score for stage in stages) / len(stages) if stages else 0.0
        total_duration = sum(stage.duration_ms for stage in stages)

        warnings = []
        recommendations = []

        # Analyze stages for warnings and recommendations
        for stage in stages:
            if not stage.success:
                warnings.append(f"Stage '{stage.stage_name}' failed: {stage.error_message}")
                recommendations.append(f"Review and fix '{stage.stage_name}' stage")

            checkpoint = self._get_checkpoint(stage.stage_name)
            if checkpoint and stage.quality_score < checkpoint.min_quality_score:
                warnings.append(
                    f"Stage '{stage.stage_name}' quality below threshold "
                    f"({stage.quality_score:.2f} < {checkpoint.min_quality_score:.2f})"
                )

            # Performance warnings
            if stage.stage_name in self.avg_durations:
                avg_duration = sum(self.avg_durations[stage.stage_name]) / len(self.avg_durations[stage.stage_name])
                if stage.duration_ms > avg_duration * 2:
                    warnings.append(
                        f"Stage '{stage.stage_name}' took {stage.duration_ms:.0f}ms "
                        f"(2x average of {avg_duration:.0f}ms)"
                    )
                    recommendations.append(f"Optimize '{stage.stage_name}' stage performance")

        return PipelineQualityReport(
            overall_success=overall_success,
            overall_quality=overall_quality,
            total_duration_ms=total_duration,
            stages=stages,
            warnings=warnings,
            recommendations=recommendations
        )

    def get_metrics_summary(self) -> Dict:
        """Get summary of tracked metrics"""
        if not self.track_metrics:
            return {}

        summary = {
            "total_stages_executed": len(self.stage_history),
            "failure_counts": self.failure_counts.copy(),
            "avg_durations_ms": {}
        }

        for stage_name, durations in self.avg_durations.items():
            summary["avg_durations_ms"][stage_name] = sum(durations) / len(durations) if durations else 0.0

        return summary

    def reset_metrics(self):
        """Reset all tracked metrics"""
        self.stage_history = []
        self.failure_counts = {}
        self.avg_durations = {}
        logger.info("[SUPERVISOR] Metrics reset")
