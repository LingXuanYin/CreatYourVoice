"""基准测试结果分析和报告生成

这个模块负责分析性能测试结果，生成详细报告，并提供性能趋势分析。
设计原则：
1. 结果分析 - 深入分析性能数据，识别瓶颈
2. 趋势监控 - 跟踪性能变化，检测回归
3. 报告生成 - 生成可读性强的性能报告
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pandas as pd

from test_performance import PerformanceMetrics, BenchmarkResult, generate_performance_report


@dataclass
class PerformanceTrend:
    """性能趋势数据"""
    metric_name: str
    timestamps: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    baseline_value: Optional[float] = None
    trend_direction: str = "stable"  # improving, degrading, stable
    regression_detected: bool = False

    def add_data_point(self, timestamp: str, value: float):
        """添加数据点"""
        self.timestamps.append(timestamp)
        self.values.append(value)
        self._analyze_trend()

    def _analyze_trend(self):
        """分析趋势"""
        if len(self.values) < 3:
            return

        # 计算最近3个点的趋势
        recent_values = self.values[-3:]
        if recent_values[-1] > recent_values[0] * 1.1:  # 增长超过10%
            self.trend_direction = "degrading"
        elif recent_values[-1] < recent_values[0] * 0.9:  # 减少超过10%
            self.trend_direction = "improving"
        else:
            self.trend_direction = "stable"

        # 检测回归
        if self.baseline_value and recent_values[-1] > self.baseline_value * 1.2:
            self.regression_detected = True


class BenchmarkAnalyzer:
    """基准测试分析器"""

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path("tests/performance/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trends = {}

    def analyze_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_performance": self._analyze_overall_performance(results),
            "bottlenecks": self._identify_bottlenecks(results),
            "memory_analysis": self._analyze_memory_usage(results),
            "recommendations": self._generate_recommendations(results),
            "detailed_analysis": self._detailed_analysis(results)
        }

        return analysis

    def _analyze_overall_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """分析整体性能"""
        all_metrics = []
        for result in results:
            all_metrics.extend(result.metrics)

        if not all_metrics:
            return {"status": "no_data"}

        execution_times = [m.execution_time for m in all_metrics]
        memory_usages = [m.memory_usage_mb for m in all_metrics]
        cpu_usages = [m.cpu_usage_percent for m in all_metrics]

        return {
            "total_operations": len(all_metrics),
            "execution_time": {
                "total": round(sum(execution_times), 3),
                "average": round(statistics.mean(execution_times), 3),
                "median": round(statistics.median(execution_times), 3),
                "max": round(max(execution_times), 3),
                "min": round(min(execution_times), 3)
            },
            "memory_usage": {
                "average_mb": round(statistics.mean(memory_usages), 2),
                "peak_mb": round(max(memory_usages), 2),
                "min_mb": round(min(memory_usages), 2)
            },
            "cpu_usage": {
                "average_percent": round(statistics.mean(cpu_usages), 2),
                "peak_percent": round(max(cpu_usages), 2)
            }
        }

    def _identify_bottlenecks(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []

        for result in results:
            for metric in result.metrics:
                # 识别执行时间瓶颈
                if metric.execution_time > 5.0:  # 超过5秒
                    bottlenecks.append({
                        "type": "execution_time",
                        "operation": metric.operation_name,
                        "value": metric.execution_time,
                        "severity": "high" if metric.execution_time > 10.0 else "medium",
                        "description": f"操作 {metric.operation_name} 执行时间过长: {metric.execution_time:.3f}s"
                    })

                # 识别内存使用瓶颈
                if metric.peak_memory_mb > 500.0:  # 超过500MB
                    bottlenecks.append({
                        "type": "memory_usage",
                        "operation": metric.operation_name,
                        "value": metric.peak_memory_mb,
                        "severity": "high" if metric.peak_memory_mb > 1000.0 else "medium",
                        "description": f"操作 {metric.operation_name} 内存使用过高: {metric.peak_memory_mb:.2f}MB"
                    })

                # 识别CPU使用瓶颈
                if metric.cpu_usage_percent > 80.0:  # 超过80%
                    bottlenecks.append({
                        "type": "cpu_usage",
                        "operation": metric.operation_name,
                        "value": metric.cpu_usage_percent,
                        "severity": "medium",
                        "description": f"操作 {metric.operation_name} CPU使用率过高: {metric.cpu_usage_percent:.2f}%"
                    })

        # 按严重程度排序
        bottlenecks.sort(key=lambda x: (x["severity"] == "high", x["value"]), reverse=True)

        return bottlenecks

    def _analyze_memory_usage(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """分析内存使用模式"""
        memory_data = []

        for result in results:
            for metric in result.metrics:
                memory_data.append({
                    "operation": metric.operation_name,
                    "memory_usage": metric.memory_usage_mb,
                    "peak_memory": metric.peak_memory_mb,
                    "items_processed": metric.items_processed
                })

        if not memory_data:
            return {"status": "no_data"}

        # 分析内存使用模式
        memory_usages = [d["memory_usage"] for d in memory_data]
        peak_memories = [d["peak_memory"] for d in memory_data]

        # 计算内存效率
        memory_efficiency = []
        for data in memory_data:
            if data["items_processed"] > 0:
                efficiency = data["items_processed"] / data["peak_memory"]
                memory_efficiency.append(efficiency)

        analysis = {
            "average_usage_mb": round(statistics.mean(memory_usages), 2),
            "peak_usage_mb": round(max(peak_memories), 2),
            "memory_variance": round(statistics.variance(memory_usages), 2),
            "memory_growth_pattern": self._analyze_memory_growth(memory_data)
        }

        if memory_efficiency:
            analysis["memory_efficiency"] = {
                "average_items_per_mb": round(statistics.mean(memory_efficiency), 2),
                "best_efficiency": round(max(memory_efficiency), 2),
                "worst_efficiency": round(min(memory_efficiency), 2)
            }

        return analysis

    def _analyze_memory_growth(self, memory_data: List[Dict]) -> str:
        """分析内存增长模式"""
        # 按处理项目数量排序
        sorted_data = sorted(memory_data, key=lambda x: x["items_processed"])

        if len(sorted_data) < 3:
            return "insufficient_data"

        # 检查内存使用是否与处理项目数量成正比
        items = [d["items_processed"] for d in sorted_data if d["items_processed"] > 0]
        memories = [d["peak_memory"] for d in sorted_data if d["items_processed"] > 0]

        if len(items) < 3:
            return "insufficient_data"

        # 简单的线性相关性检查
        correlation = self._calculate_correlation(items, memories)

        if correlation > 0.8:
            return "linear_growth"
        elif correlation > 0.5:
            return "moderate_growth"
        else:
            return "irregular_growth"

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """计算相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[Dict[str, str]]:
        """生成优化建议"""
        recommendations = []

        # 分析所有指标
        all_metrics = []
        for result in results:
            all_metrics.extend(result.metrics)

        if not all_metrics:
            return recommendations

        # 执行时间建议
        slow_operations = [m for m in all_metrics if m.execution_time > 2.0]
        if slow_operations:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "title": "优化慢速操作",
                "description": f"发现 {len(slow_operations)} 个执行时间超过2秒的操作，建议进行性能优化",
                "actions": [
                    "分析算法复杂度，寻找更高效的实现",
                    "考虑使用缓存机制减少重复计算",
                    "评估是否可以并行化处理"
                ]
            })

        # 内存使用建议
        high_memory_operations = [m for m in all_metrics if m.peak_memory_mb > 200.0]
        if high_memory_operations:
            recommendations.append({
                "category": "memory",
                "priority": "medium",
                "title": "优化内存使用",
                "description": f"发现 {len(high_memory_operations)} 个内存使用超过200MB的操作",
                "actions": [
                    "检查是否存在内存泄漏",
                    "优化数据结构，减少内存占用",
                    "实现流式处理，避免一次性加载大量数据"
                ]
            })

        # 并发性建议
        cpu_intensive_operations = [m for m in all_metrics if m.cpu_usage_percent > 70.0]
        if cpu_intensive_operations:
            recommendations.append({
                "category": "concurrency",
                "priority": "medium",
                "title": "考虑并发优化",
                "description": f"发现 {len(cpu_intensive_operations)} 个CPU密集型操作",
                "actions": [
                    "评估多线程或多进程处理的可行性",
                    "考虑使用异步处理提高响应性",
                    "优化算法减少CPU使用"
                ]
            })

        return recommendations

    def _detailed_analysis(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """详细分析"""
        detailed = {}

        for result in results:
            test_analysis = {
                "test_name": result.test_name,
                "metrics_count": len(result.metrics),
                "performance_issues": [],
                "strengths": [],
                "operation_breakdown": {}
            }

            # 分析每个操作
            for metric in result.metrics:
                op_name = metric.operation_name
                test_analysis["operation_breakdown"][op_name] = {
                    "execution_time": metric.execution_time,
                    "memory_usage": metric.memory_usage_mb,
                    "throughput": metric.throughput,
                    "efficiency_score": self._calculate_efficiency_score(metric)
                }

                # 识别问题和优点
                if metric.execution_time > 1.0:
                    test_analysis["performance_issues"].append(f"{op_name}: 执行时间较长")
                elif metric.execution_time < 0.1:
                    test_analysis["strengths"].append(f"{op_name}: 执行速度快")

                if metric.throughput and metric.throughput > 100:
                    test_analysis["strengths"].append(f"{op_name}: 高吞吐量")

            detailed[result.test_name] = test_analysis

        return detailed

    def _calculate_efficiency_score(self, metric: PerformanceMetrics) -> float:
        """计算效率分数 (0-100)"""
        # 基于执行时间、内存使用和吞吐量的综合评分
        time_score = max(0, 100 - metric.execution_time * 10)  # 执行时间越短分数越高
        memory_score = max(0, 100 - metric.memory_usage_mb / 10)  # 内存使用越少分数越高

        if metric.throughput:
            throughput_score = min(100, metric.throughput)  # 吞吐量越高分数越高
            return (time_score + memory_score + throughput_score) / 3
        else:
            return (time_score + memory_score) / 2

    def save_results(self, results: List[BenchmarkResult], analysis: Dict[str, Any]):
        """保存测试结果和分析"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存原始结果
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(generate_performance_report(results), f, indent=2, ensure_ascii=False)

        # 保存分析结果
        analysis_file = self.results_dir / f"benchmark_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # 生成HTML报告
        self._generate_html_report(results, analysis, timestamp)

    def _generate_html_report(self, results: List[BenchmarkResult], analysis: Dict[str, Any], timestamp: str):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CreatYourVoice 性能测试报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .warning {{ background: #fff3cd; border-color: #ffeaa7; }}
        .success {{ background: #d4edda; border-color: #c3e6cb; }}
        .error {{ background: #f8d7da; border-color: #f5c6cb; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .bottleneck {{ color: #d73527; font-weight: bold; }}
        .recommendation {{ background: #e7f3ff; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CreatYourVoice 性能测试报告</h1>
        <p>生成时间: {analysis['timestamp']}</p>
        <p>测试时间戳: {timestamp}</p>
    </div>

    <div class="section">
        <h2>整体性能概览</h2>
        <div class="metric">
            <h3>执行时间统计</h3>
            <p>总执行时间: {analysis['overall_performance']['execution_time']['total']}秒</p>
            <p>平均执行时间: {analysis['overall_performance']['execution_time']['average']}秒</p>
            <p>最长执行时间: {analysis['overall_performance']['execution_time']['max']}秒</p>
        </div>

        <div class="metric">
            <h3>内存使用统计</h3>
            <p>平均内存使用: {analysis['overall_performance']['memory_usage']['average_mb']}MB</p>
            <p>峰值内存使用: {analysis['overall_performance']['memory_usage']['peak_mb']}MB</p>
        </div>
    </div>

    <div class="section">
        <h2>性能瓶颈</h2>
        {self._format_bottlenecks_html(analysis['bottlenecks'])}
    </div>

    <div class="section">
        <h2>优化建议</h2>
        {self._format_recommendations_html(analysis['recommendations'])}
    </div>

    <div class="section">
        <h2>详细测试结果</h2>
        {self._format_detailed_results_html(results)}
    </div>
</body>
</html>
        """

        html_file = self.results_dir / f"benchmark_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _format_bottlenecks_html(self, bottlenecks: List[Dict]) -> str:
        """格式化瓶颈信息为HTML"""
        if not bottlenecks:
            return '<div class="metric success">未发现明显的性能瓶颈</div>'

        html = ""
        for bottleneck in bottlenecks:
            severity_class = "error" if bottleneck["severity"] == "high" else "warning"
            html += f'''
            <div class="metric {severity_class}">
                <h4>{bottleneck["operation"]} - {bottleneck["type"]}</h4>
                <p>{bottleneck["description"]}</p>
                <p>严重程度: {bottleneck["severity"]}</p>
            </div>
            '''

        return html

    def _format_recommendations_html(self, recommendations: List[Dict]) -> str:
        """格式化建议信息为HTML"""
        if not recommendations:
            return '<div class="metric success">系统性能良好，暂无优化建议</div>'

        html = ""
        for rec in recommendations:
            html += f'''
            <div class="recommendation">
                <h4>{rec["title"]} (优先级: {rec["priority"]})</h4>
                <p>{rec["description"]}</p>
                <ul>
                    {"".join(f"<li>{action}</li>" for action in rec["actions"])}
                </ul>
            </div>
            '''

        return html

    def _format_detailed_results_html(self, results: List[BenchmarkResult]) -> str:
        """格式化详细结果为HTML"""
        html = ""
        for result in results:
            html += f'''
            <div class="metric">
                <h3>{result.test_name}</h3>
                <table>
                    <tr>
                        <th>操作名称</th>
                        <th>执行时间(s)</th>
                        <th>内存使用(MB)</th>
                        <th>CPU使用(%)</th>
                        <th>吞吐量(ops/s)</th>
                    </tr>
            '''

            for metric in result.metrics:
                throughput_str = f"{metric.throughput:.2f}" if metric.throughput else "N/A"
                html += f'''
                    <tr>
                        <td>{metric.operation_name}</td>
                        <td>{metric.execution_time:.3f}</td>
                        <td>{metric.memory_usage_mb:.2f}</td>
                        <td>{metric.cpu_usage_percent:.2f}</td>
                        <td>{throughput_str}</td>
                    </tr>
                '''

            html += '''
                </table>
            </div>
            '''

        return html
