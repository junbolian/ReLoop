# ==============================================================================
# FILE: evaluate_with_probes.py
# DESCRIPTION: 集成Semantic Probes的评估流程
# ==============================================================================

import json
import os
import sys
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# 假设semantic_probes.py在同目录
from semantic_probes import ProbeRunner, ProbeResult, ProbeReport


@dataclass
class EvaluationResult:
    """单个模型代码的完整评估结果"""
    scenario_id: str
    
    # Phase 1: 基础检查
    syntax_valid: bool
    can_execute: bool
    execution_error: Optional[str]
    
    # Phase 2: Semantic Probes
    probes_total: int
    probes_passed: int
    probes_failed: int
    failed_probes: List[str]
    probe_diagnoses: Dict[str, str]
    
    # Phase 3: Objective对比
    model_status: str
    model_objective: Optional[float]
    ground_truth_objective: Optional[float]
    objective_error_pct: Optional[float]
    objective_match: bool  # 误差<5%
    
    # 综合判断
    final_verdict: str  # CORRECT, SILENT_FAILURE, CRASH, INFEASIBLE


class BenchmarkEvaluator:
    """带Semantic Probe的Benchmark评估器"""
    
    def __init__(self, ground_truth_path: str):
        """
        Args:
            ground_truth_path: benchmark_results.csv的路径
        """
        self.probe_runner = ProbeRunner()
        self.ground_truth = self._load_ground_truth(ground_truth_path)
    
    def _load_ground_truth(self, path: str) -> Dict[str, float]:
        """加载ground truth objectives"""
        gt = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scenario = row['scenario']
                    obj = row.get('objective_numeric', '')
                    if obj:
                        try:
                            gt[scenario] = float(obj)
                        except:
                            pass
        return gt
    
    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """检查Python语法"""
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def evaluate_single(self, 
                        scenario_id: str, 
                        model_code: str,
                        scenario_data: dict,
                        run_probes: bool = True) -> EvaluationResult:
        """评估单个生成的代码"""
        
        result = EvaluationResult(
            scenario_id=scenario_id,
            syntax_valid=False,
            can_execute=False,
            execution_error=None,
            probes_total=0,
            probes_passed=0,
            probes_failed=0,
            failed_probes=[],
            probe_diagnoses={},
            model_status="UNKNOWN",
            model_objective=None,
            ground_truth_objective=self.ground_truth.get(scenario_id),
            objective_error_pct=None,
            objective_match=False,
            final_verdict="CRASH"
        )
        
        # Phase 1: 语法检查
        syntax_ok, syntax_err = self.check_syntax(model_code)
        result.syntax_valid = syntax_ok
        if not syntax_ok:
            result.execution_error = syntax_err
            result.final_verdict = "CRASH"
            return result
        
        # Phase 2: Semantic Probes (可选)
        if run_probes:
            probe_reports = self.probe_runner.run_all_probes(model_code)
            summary = self.probe_runner.get_summary(probe_reports)
            
            result.probes_total = summary['total']
            result.probes_passed = summary['passed']
            result.probes_failed = summary['failed']
            result.failed_probes = summary['failed_probes']
            result.probe_diagnoses = summary['diagnoses']
        
        # Phase 3: 在实际场景上运行
        actual_result = self.probe_runner.run_model_code(model_code, scenario_data, timeout=60)
        
        result.can_execute = actual_result.get('returncode', 1) == 0 or actual_result.get('status') in ['OPTIMAL', 'INFEASIBLE', 'UNBOUNDED']
        result.model_status = actual_result.get('status', 'UNKNOWN')
        result.model_objective = actual_result.get('objective')
        
        if actual_result.get('stderr'):
            result.execution_error = actual_result['stderr'][:500]
        
        # 计算objective误差
        if result.model_objective is not None and result.ground_truth_objective is not None:
            if result.ground_truth_objective != 0:
                error_pct = abs(result.model_objective - result.ground_truth_objective) / abs(result.ground_truth_objective) * 100
            else:
                error_pct = 0 if result.model_objective == 0 else 100
            
            result.objective_error_pct = round(error_pct, 2)
            result.objective_match = error_pct < 5.0
        
        # 综合判断
        result.final_verdict = self._determine_verdict(result)
        
        return result
    
    def _determine_verdict(self, result: EvaluationResult) -> str:
        """综合判断最终结果"""
        
        if not result.syntax_valid:
            return "CRASH"
        
        if result.model_status == "INFEASIBLE":
            return "INFEASIBLE"
        
        if result.model_status == "UNBOUNDED":
            return "UNBOUNDED"
        
        if result.model_status != "OPTIMAL":
            return "CRASH"
        
        if result.objective_match:
            return "CORRECT"
        
        # 能运行，但答案错误 = Silent Failure
        return "SILENT_FAILURE"
    
    def evaluate_batch(self, 
                       submissions: List[Tuple[str, str, dict]],
                       run_probes: bool = True) -> List[EvaluationResult]:
        """
        批量评估
        
        Args:
            submissions: [(scenario_id, model_code, scenario_data), ...]
        """
        results = []
        
        for scenario_id, model_code, scenario_data in submissions:
            print(f"Evaluating {scenario_id}...")
            result = self.evaluate_single(scenario_id, model_code, scenario_data, run_probes)
            results.append(result)
            print(f"  Verdict: {result.final_verdict}")
            
            if result.failed_probes:
                print(f"  Failed probes: {result.failed_probes}")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> dict:
        """生成汇总报告"""
        
        total = len(results)
        
        verdicts = {}
        for r in results:
            verdicts[r.final_verdict] = verdicts.get(r.final_verdict, 0) + 1
        
        # Probe统计
        total_probes = sum(r.probes_total for r in results)
        passed_probes = sum(r.probes_passed for r in results)
        
        # 失败的probe分布
        probe_failures = {}
        for r in results:
            for probe_name in r.failed_probes:
                probe_failures[probe_name] = probe_failures.get(probe_name, 0) + 1
        
        # Objective误差统计
        obj_errors = [r.objective_error_pct for r in results if r.objective_error_pct is not None]
        
        report = {
            "total_scenarios": total,
            "verdicts": verdicts,
            "verdict_rates": {k: v/total*100 for k, v in verdicts.items()},
            
            "execution_rate": sum(1 for r in results if r.can_execute) / total * 100,
            "objective_match_rate": sum(1 for r in results if r.objective_match) / total * 100,
            "silent_failure_rate": verdicts.get("SILENT_FAILURE", 0) / total * 100,
            
            "probe_pass_rate": passed_probes / total_probes * 100 if total_probes > 0 else 0,
            "probe_failure_distribution": probe_failures,
            
            "objective_error_stats": {
                "mean": sum(obj_errors) / len(obj_errors) if obj_errors else None,
                "max": max(obj_errors) if obj_errors else None,
                "min": min(obj_errors) if obj_errors else None,
            }
        }
        
        return report
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """保存详细结果到CSV"""
        
        fieldnames = [
            'scenario_id', 'final_verdict', 
            'syntax_valid', 'can_execute', 'model_status',
            'probes_passed', 'probes_failed', 'failed_probes',
            'model_objective', 'ground_truth_objective', 'objective_error_pct', 'objective_match',
            'execution_error'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                row = {
                    'scenario_id': r.scenario_id,
                    'final_verdict': r.final_verdict,
                    'syntax_valid': r.syntax_valid,
                    'can_execute': r.can_execute,
                    'model_status': r.model_status,
                    'probes_passed': r.probes_passed,
                    'probes_failed': r.probes_failed,
                    'failed_probes': ','.join(r.failed_probes),
                    'model_objective': r.model_objective,
                    'ground_truth_objective': r.ground_truth_objective,
                    'objective_error_pct': r.objective_error_pct,
                    'objective_match': r.objective_match,
                    'execution_error': (r.execution_error or '')[:200]
                }
                writer.writerow(row)


# ==============================================================================
# QUICK PROBE CHECK (for repair loop)
# ==============================================================================

def quick_probe_check(model_code: str, mechanisms: List[str] = None) -> Dict[str, bool]:
    """
    快速检查特定机制是否正确实现。
    用于repair loop中快速定位问题。
    
    Args:
        model_code: 生成的代码
        mechanisms: 要检查的机制列表，如 ['substitution', 'aging']
                   如果为None，检查所有
    
    Returns:
        {mechanism: passed}
    """
    runner = ProbeRunner()
    
    # 机制到probe的映射
    mechanism_probes = {
        'substitution': ['substitution_basic', 'demand_route_constraint', 'no_substitution'],
        'aging': ['aging_dynamics'],
        'capacity': ['production_capacity', 'storage_capacity'],
        'lead_time': ['lead_time'],
        'costs': ['holding_cost']
    }
    
    if mechanisms is None:
        mechanisms = list(mechanism_probes.keys())
    
    results = {}
    
    for mechanism in mechanisms:
        probe_names = mechanism_probes.get(mechanism, [])
        if not probe_names:
            continue
        
        reports = runner.run_selected_probes(model_code, probe_names)
        
        # 只要有一个fail就认为该机制有问题
        all_passed = all(r.result == ProbeResult.PASS for r in reports)
        results[mechanism] = all_passed
    
    return results


def get_probe_diagnosis(model_code: str) -> str:
    """
    获取probe诊断信息，用于repair提示。
    
    Returns:
        诊断字符串，可直接放入repair prompt
    """
    runner = ProbeRunner()
    reports = runner.run_all_probes(model_code)
    
    failed = [r for r in reports if r.result != ProbeResult.PASS]
    
    if not failed:
        return "All semantic probes passed."
    
    lines = ["Semantic probe failures detected:"]
    for r in failed:
        lines.append(f"- {r.probe_name}: {r.diagnosis}")
    
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Benchmark Evaluator with Semantic Probes")
    print("=" * 50)
    
    # 示例用法
    evaluator = BenchmarkEvaluator("benchmark_results.csv")
    
    print(f"Loaded {len(evaluator.ground_truth)} ground truth objectives")
    print(f"Available probes: {[p.name for p in evaluator.probe_runner.probes]}")
    
    # 示例：快速检查
    example_code = "# placeholder"
    # diagnosis = get_probe_diagnosis(example_code)
    # print(diagnosis)
