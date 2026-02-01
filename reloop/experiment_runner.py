"""
ReLoop Experiment Runner

Process dataset format and run complete experiment pipeline.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .data_extraction import DataExtractor
from .pipeline import ReLoopPipeline, PipelineResult
from .verification import Severity


@dataclass
class ExperimentRecord:
    """Single experiment record."""
    id: int
    difficulty: str

    # Input
    problem_description: str
    extracted_data: Dict[str, Any]
    ground_truth: float

    # Output
    generated_code: str
    final_code: str
    predicted_objective: Optional[float]

    # Verification results
    initial_status: str
    final_status: str
    iterations: int

    # Evaluation
    objective_error: Optional[float]
    error_detected: bool
    error_repaired: bool

    # Timing
    extraction_time: float
    pipeline_time: float


@dataclass
class ExperimentSummary:
    """Experiment summary statistics."""
    total_problems: int

    # By status
    verified_count: int
    warnings_count: int
    errors_count: int
    failed_count: int

    # Evaluation metrics
    detection_rate: float       # Rate of detecting problems
    false_positive_rate: float  # Rate of false alarms
    repair_success_rate: float  # Rate of successful repairs
    avg_objective_error: float  # Average objective error

    # By difficulty
    by_difficulty: Dict[str, Dict[str, float]]

    # By layer
    layer_detection_rate: Dict[str, float]


class ExperimentRunner:
    """Experiment runner for ReLoop."""

    def __init__(
        self,
        llm_client,
        output_dir: str = "results",
        enable_cpt: bool = True,
        max_repair_iterations: int = 3,
        verbose: bool = False
    ):
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.extractor = DataExtractor(llm_client)
        self.pipeline = ReLoopPipeline(
            llm_client,
            max_repair_iterations=max_repair_iterations,
            enable_cpt=enable_cpt,
            verbose=verbose
        )
        self.verbose = verbose

    def run_dataset(self, dataset_path: str) -> ExperimentSummary:
        """
        Run experiment on entire dataset.

        Args:
            dataset_path: Path to dataset JSON file

        Returns:
            Experiment summary
        """
        # Load dataset
        dataset = self._load_dataset(dataset_path)

        records = []

        for i, item in enumerate(dataset):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Problem {i+1}/{len(dataset)}: ID={item.get('id', i+1)}")
                print(f"{'='*60}")

            record = self._run_single(item)
            records.append(record)

            # Save intermediate results
            self._save_record(record)

        # Compute summary
        summary = self._compute_summary(records)
        self._save_summary(summary)

        return summary

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from file."""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Try JSONL format first (one JSON per line)
        if '\n' in content:
            lines = content.split('\n')
            dataset = []
            for line in lines:
                line = line.strip()
                if not line or line in ['[', ']']:
                    continue
                # Remove trailing comma if present
                if line.endswith(','):
                    line = line[:-1]
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        dataset.append(item)
                except json.JSONDecodeError:
                    continue
            if dataset:
                return dataset

        # Try regular JSON array or object
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("problems", [data])
        except json.JSONDecodeError:
            pass

        return []

    def _run_single(self, item: Dict) -> ExperimentRecord:
        """Run single problem."""
        problem_id = item.get("id", 0)
        difficulty = item.get("difficulty", "Unknown")
        problem = item.get("en_question", "")
        ground_truth_str = item.get("en_answer", "0")

        # Parse ground truth
        try:
            ground_truth = float(ground_truth_str)
        except (ValueError, TypeError):
            ground_truth = 0.0

        # Step 1: Extract data
        extract_start = time.time()
        data = self.extractor.extract(problem)
        extraction_time = time.time() - extract_start

        if self.verbose:
            print(f"[Extraction] Extracted {len(data)} parameters in {extraction_time:.2f}s")
            print(f"[Data] {list(data.keys())}")

        # Step 2-4: Pipeline (generate -> verify -> repair)
        pipeline_start = time.time()
        result = self.pipeline.run(problem, data)
        pipeline_time = time.time() - pipeline_start

        if self.verbose:
            print(f"[Pipeline] {result.iterations} iterations in {pipeline_time:.2f}s")
            initial_status = result.history[0][1].status if result.history else "UNKNOWN"
            print(f"[Status] {initial_status} -> {result.final_report.status}")

        # Step 5: Evaluate
        predicted = result.final_report.objective

        if predicted is not None and ground_truth != 0:
            objective_error = abs(predicted - ground_truth) / abs(ground_truth)
        else:
            objective_error = None

        # Check if error was detected
        initial_status = result.history[0][1].status if result.history else "UNKNOWN"
        error_detected = initial_status in ['WARNINGS', 'ERRORS']

        # Check if error was repaired
        error_repaired = (
            error_detected and
            result.final_report.status == 'VERIFIED' and
            objective_error is not None and
            objective_error < 0.01  # Error less than 1%
        )

        if self.verbose:
            print(f"[Objective] Predicted={predicted}, Ground Truth={ground_truth}")
            if objective_error is not None:
                print(f"[Error] {objective_error:.2%}")

        return ExperimentRecord(
            id=problem_id,
            difficulty=difficulty,
            problem_description=problem,
            extracted_data=data,
            ground_truth=ground_truth,
            generated_code=result.history[0][0] if result.history else "",
            final_code=result.final_code,
            predicted_objective=predicted,
            initial_status=initial_status,
            final_status=result.final_report.status,
            iterations=result.iterations,
            objective_error=objective_error,
            error_detected=error_detected,
            error_repaired=error_repaired,
            extraction_time=extraction_time,
            pipeline_time=pipeline_time
        )

    def _compute_summary(self, records: List[ExperimentRecord]) -> ExperimentSummary:
        """Compute summary statistics."""
        total = len(records)

        # Count by status
        verified = sum(1 for r in records if r.final_status == 'VERIFIED')
        warnings = sum(1 for r in records if r.final_status == 'WARNINGS')
        errors = sum(1 for r in records if r.final_status == 'ERRORS')
        failed = sum(1 for r in records if r.final_status == 'FAILED')

        # Detection rate: problems with error > 1% that were detected
        has_error = [r for r in records if r.objective_error is not None and r.objective_error > 0.01]
        detected_errors = [r for r in has_error if r.error_detected]
        detection_rate = len(detected_errors) / len(has_error) if has_error else 0.0

        # False positive rate: problems with error <= 1% that were flagged
        no_error = [r for r in records if r.objective_error is not None and r.objective_error <= 0.01]
        false_positives = [r for r in no_error if r.error_detected]
        false_positive_rate = len(false_positives) / len(no_error) if no_error else 0.0

        # Repair success rate
        detected = [r for r in records if r.error_detected]
        repaired = [r for r in detected if r.error_repaired]
        repair_success_rate = len(repaired) / len(detected) if detected else 0.0

        # Average objective error
        errors_list = [r.objective_error for r in records if r.objective_error is not None]
        avg_objective_error = sum(errors_list) / len(errors_list) if errors_list else 0.0

        # By difficulty
        by_difficulty = {}
        for diff in set(r.difficulty for r in records):
            diff_records = [r for r in records if r.difficulty == diff]
            diff_errors = [r.objective_error for r in diff_records if r.objective_error is not None]
            by_difficulty[diff] = {
                "count": len(diff_records),
                "verified_rate": sum(1 for r in diff_records if r.final_status == 'VERIFIED') / len(diff_records) if diff_records else 0.0,
                "avg_error": sum(diff_errors) / len(diff_errors) if diff_errors else 0.0
            }

        return ExperimentSummary(
            total_problems=total,
            verified_count=verified,
            warnings_count=warnings,
            errors_count=errors,
            failed_count=failed,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            repair_success_rate=repair_success_rate,
            avg_objective_error=avg_objective_error,
            by_difficulty=by_difficulty,
            layer_detection_rate={}  # TODO: compute from layer_results
        )

    def _save_record(self, record: ExperimentRecord):
        """Save single record."""
        record_file = self.output_dir / f"record_{record.id}.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump({
                "id": record.id,
                "difficulty": record.difficulty,
                "ground_truth": record.ground_truth,
                "predicted_objective": record.predicted_objective,
                "objective_error": record.objective_error,
                "initial_status": record.initial_status,
                "final_status": record.final_status,
                "iterations": record.iterations,
                "error_detected": record.error_detected,
                "error_repaired": record.error_repaired,
                "extraction_time": record.extraction_time,
                "pipeline_time": record.pipeline_time
            }, f, indent=2)

    def _save_summary(self, summary: ExperimentSummary):
        """Save summary."""
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_problems": summary.total_problems,
                "verified_count": summary.verified_count,
                "warnings_count": summary.warnings_count,
                "errors_count": summary.errors_count,
                "failed_count": summary.failed_count,
                "detection_rate": summary.detection_rate,
                "false_positive_rate": summary.false_positive_rate,
                "repair_success_rate": summary.repair_success_rate,
                "avg_objective_error": summary.avg_objective_error,
                "by_difficulty": summary.by_difficulty
            }, f, indent=2)


def run_experiment(
    dataset_path: str,
    llm_client,
    output_dir: str = "results",
    verbose: bool = True
) -> ExperimentSummary:
    """
    Convenience function to run experiment.

    Args:
        dataset_path: Path to dataset file
        llm_client: LLM client with generate() method
        output_dir: Output directory for results
        verbose: Print progress

    Returns:
        Experiment summary
    """
    runner = ExperimentRunner(llm_client, output_dir, verbose=verbose)
    return runner.run_dataset(dataset_path)
