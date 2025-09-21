"""完整测试执行器

这个模块负责执行所有测试套件并生成综合报告。
设计原则：
1. 全面覆盖 - 执行所有类型的测试
2. 结果汇总 - 生成统一的测试报告
3. 问题识别 - 自动识别和分类问题
"""

import unittest
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import subprocess
import importlib

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@dataclass
class TestResult:
    """测试结果"""
    test_suite: str
    success: bool
    tests_run: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    execution_time: float = 0.0
    error_details: List[str] = field(default_factory=list)
    failure_details: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.tests_run == 0:
            return 0.0
        return (self.tests_run - self.failures - self.errors) / self.tests_run * 100


@dataclass
class ComprehensiveTestReport:
    """综合测试报告"""
    timestamp: str
    total_execution_time: float
    test_results: List[TestResult] = field(default_factory=list)
    overall_success: bool = True
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def add_result(self, result: TestResult):
        """添加测试结果"""
        self.test_results.append(result)
        if not result.success:
            self.overall_success = False

    def generate_summary(self):
        """生成摘要"""
        total_tests = sum(r.tests_run for r in self.test_results)
        total_failures = sum(r.failures for r in self.test_results)
        total_errors = sum(r.errors for r in self.test_results)
        total_skipped = sum(r.skipped for r in self.test_results)

        successful_suites = sum(1 for r in self.test_results if r.success)

        self.summary = {
            "total_test_suites": len(self.test_results),
            "successful_suites": successful_suites,
            "failed_suites": len(self.test_results) - successful_suites,
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "total_skipped": total_skipped,
            "overall_success_rate": (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0,
            "execution_time": self.total_execution_time
        }


class TestSuiteRunner:
    """测试套件运行器"""

    def __init__(self):
        self.test_modules = {
            "端到端测试": "tests.integration.test_end_to_end",
            "音色创建测试": "tests.integration.test_voice_creation",
            "音色合成测试": "tests.integration.test_voice_synthesis",
            "音色融合测试": "tests.integration.test_voice_fusion",
            "性能测试": "tests.performance.test_performance",
            "界面测试": "tests.ui.test_gradio_interface",
            "安装测试": "tests.deployment.test_installation",
            "配置测试": "tests.deployment.test_configuration"
        }

    def run_test_module(self, module_name: str, display_name: str) -> TestResult:
        """运行单个测试模块"""
        logger.info(f"开始执行 {display_name}")

        start_time = time.time()
        result = TestResult(test_suite=display_name, success=False)

        try:
            # 动态导入测试模块
            test_module = importlib.import_module(module_name)

            # 查找测试类
            test_loader = unittest.TestLoader()
            test_suite = unittest.TestSuite()

            # 加载模块中的所有测试
            module_tests = test_loader.loadTestsFromModule(test_module)
            test_suite.addTests(module_tests)

            # 运行测试
            test_runner = unittest.TextTestRunner(
                verbosity=0,  # 减少输出
                stream=open('/dev/null', 'w') if sys.platform != 'win32' else open('nul', 'w')
            )

            test_result = test_runner.run(test_suite)

            # 收集结果
            result.tests_run = test_result.testsRun
            result.failures = len(test_result.failures)
            result.errors = len(test_result.errors)
            result.skipped = len(test_result.skipped)
            result.success = test_result.wasSuccessful()

            # 收集错误详情
            for test, error in test_result.failures:
                result.failure_details.append(f"{test}: {error}")

            for test, error in test_result.errors:
                result.error_details.append(f"{test}: {error}")

            logger.info(f"✓ {display_name} 完成: {result.tests_run} 测试, {result.failures} 失败, {result.errors} 错误")

        except ImportError as e:
            result.error_details.append(f"模块导入失败: {e}")
            logger.error(f"✗ {display_name} 导入失败: {e}")
        except Exception as e:
            result.error_details.append(f"执行异常: {e}")
            logger.error(f"✗ {display_name} 执行异常: {e}")

        result.execution_time = time.time() - start_time
        return result

    def run_all_tests(self) -> ComprehensiveTestReport:
        """运行所有测试"""
        logger.info("开始执行完整测试套件")

        start_time = time.time()
        report = ComprehensiveTestReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_execution_time=0.0
        )

        # 执行每个测试模块
        for display_name, module_name in self.test_modules.items():
            try:
                result = self.run_test_module(module_name, display_name)
                report.add_result(result)
            except Exception as e:
                # 如果测试模块完全失败，创建失败结果
                failed_result = TestResult(
                    test_suite=display_name,
                    success=False,
                    error_details=[f"测试套件执行失败: {e}"]
                )
                report.add_result(failed_result)
                logger.error(f"测试套件 {display_name} 执行失败: {e}")

        report.total_execution_time = time.time() - start_time
        report.generate_summary()

        logger.info(f"完整测试套件执行完成，耗时: {report.total_execution_time:.2f}秒")

        return report


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROJECT_ROOT / "test_reports"
        self.output_dir.mkdir(exist_ok=True)

    def generate_json_report(self, report: ComprehensiveTestReport) -> Path:
        """生成JSON报告"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"test_report_{timestamp}.json"

        report_data = {
            "timestamp": report.timestamp,
            "total_execution_time": report.total_execution_time,
            "overall_success": report.overall_success,
            "summary": report.summary,
            "test_results": [
                {
                    "test_suite": r.test_suite,
                    "success": r.success,
                    "tests_run": r.tests_run,
                    "failures": r.failures,
                    "errors": r.errors,
                    "skipped": r.skipped,
                    "success_rate": r.success_rate,
                    "execution_time": r.execution_time,
                    "error_details": r.error_details,
                    "failure_details": r.failure_details
                }
                for r in report.test_results
            ],
            "recommendations": report.recommendations
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return json_file

    def generate_html_report(self, report: ComprehensiveTestReport) -> Path:
        """生成HTML报告"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        html_file = self.output_dir / f"test_report_{timestamp}.html"

        # 生成测试结果表格
        results_html = ""
        for result in report.test_results:
            status_class = "success" if result.success else "failure"
            status_text = "✓ 通过" if result.success else "✗ 失败"

            results_html += f"""
            <tr class="{status_class}">
                <td>{result.test_suite}</td>
                <td>{status_text}</td>
                <td>{result.tests_run}</td>
                <td>{result.failures}</td>
                <td>{result.errors}</td>
                <td>{result.skipped}</td>
                <td>{result.success_rate:.1f}%</td>
                <td>{result.execution_time:.2f}s</td>
            </tr>
            """

        # 生成错误详情
        error_details_html = ""
        for result in report.test_results:
            if result.error_details or result.failure_details:
                error_details_html += f"""
                <div class="error-section">
                    <h3>{result.test_suite} - 错误详情</h3>
                """

                for error in result.failure_details:
                    error_details_html += f"<div class='error-item failure'>失败: {error}</div>"

                for error in result.error_details:
                    error_details_html += f"<div class='error-item error'>错误: {error}</div>"

                error_details_html += "</div>"

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CreatYourVoice 测试报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .summary-card.success {{ border-left-color: #28a745; }}
        .summary-card.warning {{ border-left-color: #ffc107; }}
        .summary-card.danger {{ border-left-color: #dc3545; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #333; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .summary-card.success .value {{ color: #28a745; }}
        .summary-card.warning .value {{ color: #ffc107; }}
        .summary-card.danger .value {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .success {{ background-color: #d4edda; }}
        .failure {{ background-color: #f8d7da; }}
        .error-section {{ margin: 20px 0; padding: 20px; background: #fff3cd; border-radius: 8px; }}
        .error-item {{ margin: 10px 0; padding: 10px; border-radius: 4px; }}
        .error-item.failure {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .error-item.error {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .recommendations {{ background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .recommendations h3 {{ color: #0066cc; margin-top: 0; }}
        .recommendations ul {{ margin: 10px 0; }}
        .recommendations li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 CreatYourVoice 测试报告</h1>
            <p>生成时间: {report.timestamp}</p>
            <p>总执行时间: {report.total_execution_time:.2f} 秒</p>
        </div>

        <div class="summary">
            <div class="summary-card {'success' if report.overall_success else 'danger'}">
                <h3>总体状态</h3>
                <div class="value">{'✓ 通过' if report.overall_success else '✗ 失败'}</div>
            </div>
            <div class="summary-card">
                <h3>测试套件</h3>
                <div class="value">{report.summary['successful_suites']}/{report.summary['total_test_suites']}</div>
            </div>
            <div class="summary-card">
                <h3>测试用例</h3>
                <div class="value">{report.summary['total_tests']}</div>
            </div>
            <div class="summary-card {'success' if report.summary['overall_success_rate'] > 90 else 'warning' if report.summary['overall_success_rate'] > 70 else 'danger'}">
                <h3>成功率</h3>
                <div class="value">{report.summary['overall_success_rate']:.1f}%</div>
            </div>
        </div>

        <h2>测试结果详情</h2>
        <table>
            <thead>
                <tr>
                    <th>测试套件</th>
                    <th>状态</th>
                    <th>测试数</th>
                    <th>失败</th>
                    <th>错误</th>
                    <th>跳过</th>
                    <th>成功率</th>
                    <th>执行时间</th>
                </tr>
            </thead>
            <tbody>
                {results_html}
            </tbody>
        </table>

        {error_details_html}

        <div class="recommendations">
            <h3>📋 建议和改进</h3>
            <ul>
                <li>定期运行完整测试套件以确保系统稳定性</li>
                <li>关注失败的测试用例，及时修复发现的问题</li>
                <li>监控性能测试结果，优化系统性能</li>
                <li>保持测试覆盖率，添加新功能时同步更新测试</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_file

    def generate_console_report(self, report: ComprehensiveTestReport):
        """生成控制台报告"""
        print("\n" + "="*80)
        print("🎵 CreatYourVoice 测试报告")
        print("="*80)
        print(f"生成时间: {report.timestamp}")
        print(f"总执行时间: {report.total_execution_time:.2f} 秒")
        print(f"总体状态: {'✓ 通过' if report.overall_success else '✗ 失败'}")
        print()

        print("📊 测试摘要:")
        print(f"  测试套件: {report.summary['successful_suites']}/{report.summary['total_test_suites']} 通过")
        print(f"  测试用例: {report.summary['total_tests']} 个")
        print(f"  失败: {report.summary['total_failures']} 个")
        print(f"  错误: {report.summary['total_errors']} 个")
        print(f"  跳过: {report.summary['total_skipped']} 个")
        print(f"  成功率: {report.summary['overall_success_rate']:.1f}%")
        print()

        print("📋 详细结果:")
        for result in report.test_results:
            status = "✓" if result.success else "✗"
            print(f"  {status} {result.test_suite}: {result.tests_run} 测试, {result.failures} 失败, {result.errors} 错误 ({result.execution_time:.2f}s)")

        # 显示错误详情
        failed_suites = [r for r in report.test_results if not r.success]
        if failed_suites:
            print("\n❌ 失败详情:")
            for result in failed_suites:
                print(f"\n  {result.test_suite}:")
                for error in result.failure_details[:3]:  # 只显示前3个错误
                    print(f"    - {error[:100]}...")
                for error in result.error_details[:3]:
                    print(f"    - {error[:100]}...")

        print("\n" + "="*80)


def main():
    """主函数"""
    setup_logging(log_level="INFO")

    print("🎵 CreatYourVoice 完整测试套件")
    print("开始执行所有测试...")

    # 运行测试
    runner = TestSuiteRunner()
    report = runner.run_all_tests()

    # 生成报告
    generator = ReportGenerator()

    # 生成控制台报告
    generator.generate_console_report(report)

    # 生成文件报告
    json_file = generator.generate_json_report(report)
    html_file = generator.generate_html_report(report)

    print(f"\n📄 报告已生成:")
    print(f"  JSON报告: {json_file}")
    print(f"  HTML报告: {html_file}")

    # 返回退出码
    return 0 if report.overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
