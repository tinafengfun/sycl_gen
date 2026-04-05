#!/usr/bin/env python3
"""
JSON Report Generator
JSON报告生成器

生成结构化JSON报告，支持高层Agent决策
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class TestIssue:
    """测试中发现的问题"""
    severity: str  # "critical", "warning", "info"
    test_id: Optional[str]
    category: str
    message: str
    details: Optional[Dict] = None
    recommendation: Optional[str] = None

@dataclass
class TestReport:
    """完整测试报告"""
    # 元信息
    kernel_id: str
    kernel_name: str
    test_timestamp: str
    total_duration_seconds: float
    framework_version: str = "1.0.0"
    
    # 平台信息
    platform_info: Dict = field(default_factory=dict)
    
    # 测试配置
    test_configurations: List[Dict] = field(default_factory=list)
    
    # 测试结果
    test_results: List[Dict] = field(default_factory=list)
    
    # 聚合统计
    summary: Dict = field(default_factory=dict)
    
    # LLM使用统计
    llm_usage: Dict = field(default_factory=dict)
    
    # 问题和建议
    issues: List[TestIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # 决策支持
    decision_support: Dict = field(default_factory=dict)
    
    # 执行trace
    execution_trace: List[Dict] = field(default_factory=list)


class JSONReportGenerator:
    """JSON报告生成器"""
    
    def __init__(self, kernel_id: str, kernel_name: str):
        self.kernel_id = kernel_id
        self.kernel_name = kernel_name
        self.start_time = time.time()
        self.trace = []
        self.issues = []
        self.recommendations = []
    
    def add_trace(self, event: str, test_id: str = None, details: Dict = None):
        """添加执行trace"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "test_id": test_id,
            "details": details or {}
        }
        self.trace.append(trace_entry)
    
    def add_issue(self, severity: str, category: str, message: str,
                  test_id: str = None, details: Dict = None, 
                  recommendation: str = None):
        """添加问题"""
        issue = TestIssue(
            severity=severity,
            test_id=test_id,
            category=category,
            message=message,
            details=details,
            recommendation=recommendation
        )
        self.issues.append(issue)
    
    def add_recommendation(self, recommendation: str):
        """添加建议"""
        self.recommendations.append(recommendation)
    
    def generate_report(
        self,
        platform_info: Dict,
        test_configs: List[Dict],
        test_results: List[Dict],
        llm_usage: Dict
    ) -> Dict:
        """
        生成完整JSON报告
        
        Returns:
            结构化JSON字典
        """
        # 计算统计
        summary = self._calculate_summary(test_results, test_configs)
        
        # 生成决策支持信息
        decision_support = self._generate_decision_support(summary, test_results)
        
        # 构建完整报告
        report = {
            "metadata": {
                "kernel_id": self.kernel_id,
                "kernel_name": self.kernel_name,
                "test_timestamp": datetime.now().isoformat(),
                "total_duration_seconds": time.time() - self.start_time,
                "framework_version": "1.0.0"
            },
            
            "platform": platform_info,
            
            "test_configurations": test_configs,
            
            "test_results": test_results,
            
            "summary": summary,
            
            "llm_usage": llm_usage,
            
            "issues": [
                {
                    "severity": issue.severity,
                    "test_id": issue.test_id,
                    "category": issue.category,
                    "message": issue.message,
                    "details": issue.details,
                    "recommendation": issue.recommendation
                }
                for issue in self.issues
            ],
            
            "recommendations": self.recommendations,
            
            "decision_support": decision_support,
            
            "execution_trace": self.trace
        }
        
        return report
    
    def _calculate_summary(self, test_results: List[Dict], 
                          test_configs: List[Dict]) -> Dict:
        """计算测试摘要"""
        total = len(test_results)
        passed = sum(1 for r in test_results if r.get("status") == "PASSED")
        failed = sum(1 for r in test_results if r.get("status") == "FAILED")
        skipped = sum(1 for r in test_results if r.get("status") == "SKIPPED")
        warnings = sum(1 for r in test_results if r.get("status") == "WARNING")
        
        # 计算通过率
        pass_rate = passed / total if total > 0 else 0
        
        # 计算误差统计
        errors = []
        for result in test_results:
            if "comparison" in result:
                comp = result["comparison"]
                errors.append({
                    "max_abs": comp.get("max_abs_error", 0),
                    "max_rel": comp.get("max_rel_error", 0),
                    "mean_abs": comp.get("mean_abs_error", 0)
                })
        
        if errors:
            avg_max_abs = sum(e["max_abs"] for e in errors) / len(errors)
            avg_max_rel = sum(e["max_rel"] for e in errors) / len(errors)
            avg_mean_abs = sum(e["mean_abs"] for e in errors) / len(errors)
        else:
            avg_max_abs = avg_max_rel = avg_mean_abs = 0
        
        # 计算覆盖率
        data_types_tested = set()
        dimensions_tested = set()
        extreme_values_tested = set()
        
        for config in test_configs:
            data_types_tested.add(config.get("dtype", "unknown"))
            
            dim_key = f"{config.get('N',1)}x{config.get('C',1)}x{config.get('H',1)}x{config.get('W',1)}"
            dimensions_tested.add(dim_key)
            
            if "boundary" in config.get("data_gen", ""):
                extreme_values_tested.add("boundary")
            if "special" in config.get("data_gen", ""):
                extreme_values_tested.add("special")
        
        # 计算平均测试时间
        durations = [
            sum(p.get("duration", 0) for p in r.get("phases", {}).values())
            for r in test_results
        ]
        avg_test_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "warnings": warnings,
            "pass_rate": pass_rate,
            "consistency_rate": 1.0 if failed == 0 else (passed / (passed + failed)),
            "error_statistics": {
                "average_max_abs_error": avg_max_abs,
                "average_max_rel_error": avg_max_rel,
                "average_mean_abs_error": avg_mean_abs
            },
            "coverage": {
                "data_types": {
                    "tested": len(data_types_tested),
                    "list": list(data_types_tested)
                },
                "dimensions": {
                    "tested": len(dimensions_tested),
                    "count": len(dimensions_tested)
                },
                "extreme_values": {
                    "tested": len(extreme_values_tested),
                    "categories": list(extreme_values_tested)
                }
            },
            "performance": {
                "average_test_duration_seconds": avg_test_duration,
                "total_tests_duration_seconds": sum(durations),
                "tests_per_minute": 60.0 / avg_test_duration if avg_test_duration > 0 else 0
            }
        }
    
    def _generate_decision_support(self, summary: Dict, 
                                   test_results: List[Dict]) -> Dict:
        """生成决策支持信息"""
        
        # 质量评估
        if summary["pass_rate"] >= 0.99 and summary["failed"] == 0:
            quality_score = "A"
            quality_verdict = "EXCELLENT"
            quality_desc = "Conversion is highly accurate across all test scenarios"
        elif summary["pass_rate"] >= 0.95:
            quality_score = "B"
            quality_verdict = "GOOD"
            quality_desc = "Conversion is accurate with minor issues"
        elif summary["pass_rate"] >= 0.80:
            quality_score = "C"
            quality_verdict = "ACCEPTABLE"
            quality_desc = "Conversion has some issues but may be usable"
        else:
            quality_score = "D"
            quality_verdict = "POOR"
            quality_desc = "Conversion has significant issues and needs review"
        
        # 部署准备度
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        blocker_messages = [i.message for i in critical_issues]
        
        if len(critical_issues) == 0 and summary["pass_rate"] >= 0.95:
            ready = True
            confidence = "HIGH"
        elif len(critical_issues) == 0 and summary["pass_rate"] >= 0.80:
            ready = True
            confidence = "MEDIUM"
        else:
            ready = False
            confidence = "LOW"
        
        # 风险评估
        risks = []
        
        if summary["pass_rate"] < 0.95:
            risks.append({
                "level": "high",
                "category": "accuracy",
                "description": f"Low pass rate ({summary['pass_rate']*100:.1f}%) may indicate conversion issues"
            })
        
        if any(i.category == "nan_behavior" for i in self.issues):
            risks.append({
                "level": "medium",
                "category": "portability",
                "description": "NaN handling differences between CUDA and SYCL may cause issues"
            })
        
        # 下一步建议
        next_steps = []
        
        if summary["failed"] > 0:
            next_steps.append("Review failed tests and fix conversion issues")
        
        if any(i.category == "platform_not_supported" for i in self.issues):
            next_steps.append("Consider testing on additional platforms")
        
        if summary["pass_rate"] >= 0.99:
            next_steps.append("Conversion validated - ready for integration")
        
        return {
            "conversion_quality": {
                "score": quality_score,
                "verdict": quality_verdict,
                "description": quality_desc,
                "metrics": {
                    "pass_rate": summary["pass_rate"],
                    "consistency_rate": summary["consistency_rate"],
                    "total_tests": summary["total_tests"]
                }
            },
            "deployment_readiness": {
                "ready": ready,
                "confidence": confidence,
                "blockers": blocker_messages,
                "requirements_met": summary["pass_rate"] >= 0.95 and len(critical_issues) == 0
            },
            "risks": risks,
            "next_steps": next_steps,
            "estimated_effort": {
                "review_time_minutes": 30 if summary["failed"] > 0 else 5,
                "fixes_needed": summary["failed"],
                "recommendation": "Minor fixes" if summary["failed"] <= 2 else "Significant work"
            }
        }
    
    def save_report(self, report: Dict, output_dir: str = None) -> str:
        """保存报告到文件"""
        if output_dir is None:
            output_dir = str(self.base_dir / "results" / "reports")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.kernel_id}_test_report_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(filepath)
    
    def print_summary(self, report: Dict):
        """打印报告摘要到控制台"""
        print("\n" + "="*70)
        print("📊 TEST REPORT SUMMARY")
        print("="*70)
        
        meta = report["metadata"]
        print(f"\nKernel: {meta['kernel_name']} ({meta['kernel_id']})")
        print(f"Duration: {meta['total_duration_seconds']:.1f}s")
        print(f"Timestamp: {meta['test_timestamp']}")
        
        summary = report["summary"]
        print(f"\n📈 Test Results:")
        print(f"   Total: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ✅")
        print(f"   Failed: {summary['failed']} ❌")
        print(f"   Skipped: {summary['skipped']} ⏭️")
        print(f"   Pass Rate: {summary['pass_rate']*100:.1f}%")
        
        ds = report["decision_support"]
        print(f"\n🎯 Quality Assessment:")
        print(f"   Score: {ds['conversion_quality']['score']}")
        print(f"   Verdict: {ds['conversion_quality']['verdict']}")
        print(f"   Description: {ds['conversion_quality']['description']}")
        
        print(f"\n🚀 Deployment Readiness:")
        print(f"   Ready: {'YES ✅' if ds['deployment_readiness']['ready'] else 'NO ❌'}")
        print(f"   Confidence: {ds['deployment_readiness']['confidence']}")
        
        if ds["deployment_readiness"]["blockers"]:
            print(f"\n   Blockers:")
            for blocker in ds["deployment_readiness"]["blockers"]:
                print(f"     - {blocker}")
        
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   - {rec}")
        
        if ds["next_steps"]:
            print(f"\n📋 Next Steps:")
            for step in ds["next_steps"]:
                print(f"   - {step}")
        
        print("\n" + "="*70)


# 使用示例
def example_report_generation():
    """示例：生成报告"""
    generator = JSONReportGenerator("test_kernel", "testKernel")
    
    # 添加trace
    generator.add_trace("test_started")
    generator.add_trace("harness_generated", "test_001", {"llm_calls": 2})
    
    # 添加问题
    generator.add_issue(
        severity="warning",
        category="nan_behavior",
        message="NaN handling differs for inf - inf",
        test_id="f32_special",
        recommendation="Consider explicit NaN checks"
    )
    
    # 添加建议
    generator.add_recommendation("All core tests passed - ready for integration")
    
    # 模拟测试结果
    test_results = [
        {
            "test_id": "f32_small_random",
            "status": "PASSED",
            "phases": {
                "harness_generation": {"status": "completed", "duration": 15.2},
                "cuda_compilation": {"status": "completed", "duration": 3.1},
                "sycl_compilation": {"status": "completed", "duration": 8.5},
                "cuda_execution": {"status": "completed", "duration": 0.5},
                "sycl_execution": {"status": "completed", "duration": 0.8},
                "result_comparison": {"status": "completed", "duration": 0.01}
            },
            "comparison": {
                "pass": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "mean_abs_error": 0.0,
                "violations": 0,
                "violation_rate": 0.0
            }
        }
    ]
    
    test_configs = [
        {
            "test_id": "f32_small_random",
            "name": "Float32 Small Random",
            "dtype": "float32",
            "N": 1, "C": 64, "H": 8, "W": 8,
            "data_gen": "random_uniform",
            "tolerance": {"abs": 1e-5, "rel": 1e-4}
        }
    ]
    
    platform_info = {
        "sycl": {
            "device": "Intel Graphics",
            "vendor": "Intel",
            "supports_float32": True,
            "supports_float16": True,
            "supports_bfloat16": False
        },
        "cuda": {
            "device": "NVIDIA L20",
            "sm_version": 89,
            "supports_float32": True,
            "supports_float16": True,
            "supports_bfloat16": True
        }
    }
    
    llm_usage = {
        "total_calls": 5,
        "total_input_tokens": 10000,
        "total_output_tokens": 8000,
        "estimated_cost_usd": 0.15
    }
    
    # 生成报告
    report = generator.generate_report(
        platform_info=platform_info,
        test_configs=test_configs,
        test_results=test_results,
        llm_usage=llm_usage
    )
    
    # 打印摘要
    generator.print_summary(report)
    
    # 保存报告
    filepath = generator.save_report(report, "/tmp")
    print(f"\n💾 Report saved to: {filepath}")
    
    return report


if __name__ == "__main__":
    report = example_report_generation()
    print("\n📄 Full Report (first 2000 chars):")
    print(json.dumps(report, indent=2)[:2000])
