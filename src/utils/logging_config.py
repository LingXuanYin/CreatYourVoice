"""日志配置模块

这个模块提供统一的日志配置和管理。
设计原则：
1. 统一配置 - 所有模块使用相同的日志格式
2. 分级记录 - 不同级别的日志记录到不同位置
3. 性能优化 - 避免过度日志记录影响性能
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }

    def format(self, record):
        # 添加颜色
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)


class LoggingManager:
    """日志管理器

    负责配置和管理整个应用的日志系统。
    """

    def __init__(self):
        self._configured = False
        self._loggers: Dict[str, logging.Logger] = {}

    def setup_logging(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_dir: str = "logs",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True,
        colored_output: bool = True
    ) -> None:
        """设置日志配置

        Args:
            log_level: 日志级别
            log_file: 日志文件名
            log_dir: 日志目录
            max_file_size: 单个日志文件最大大小
            backup_count: 备份文件数量
            console_output: 是否输出到控制台
            colored_output: 控制台输出是否使用颜色
        """
        if self._configured:
            return

        # 创建日志目录
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # 获取根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # 清除现有处理器
        root_logger.handlers.clear()

        # 日志格式
        detailed_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        simple_format = "%(asctime)s - %(levelname)s - %(message)s"

        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))

            if colored_output and sys.stdout.isatty():
                console_formatter = ColoredFormatter(simple_format)
            else:
                console_formatter = logging.Formatter(simple_format)

            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            log_file_path = log_dir_path / log_file

            # 使用RotatingFileHandler进行日志轮转
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别

            file_formatter = logging.Formatter(detailed_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # 错误日志文件（单独记录ERROR及以上级别）
        if log_file:
            error_log_path = log_dir_path / f"error_{log_file}"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(detailed_format))
            root_logger.addHandler(error_handler)

        self._configured = True

        # 记录启动信息
        logger = self.get_logger("LoggingManager")
        logger.info("日志系统初始化完成")
        logger.info(f"日志级别: {log_level}")
        logger.info(f"日志目录: {log_dir_path.absolute()}")
        if log_file:
            logger.info(f"日志文件: {log_file}")

    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志器

        Args:
            name: 日志器名称

        Returns:
            日志器实例
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

    def set_level(self, name: str, level: str) -> None:
        """设置指定日志器的级别

        Args:
            name: 日志器名称
            level: 日志级别
        """
        logger = self.get_logger(name)
        logger.setLevel(getattr(logging, level.upper()))

    def add_file_handler(
        self,
        name: str,
        file_path: str,
        level: str = "INFO",
        format_string: Optional[str] = None
    ) -> None:
        """为指定日志器添加文件处理器

        Args:
            name: 日志器名称
            file_path: 文件路径
            level: 日志级别
            format_string: 格式字符串
        """
        logger = self.get_logger(name)

        # 创建文件目录
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建文件处理器
        handler = logging.FileHandler(file_path, encoding='utf-8')
        handler.setLevel(getattr(logging, level.upper()))

        # 设置格式
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    def log_system_info(self) -> None:
        """记录系统信息"""
        logger = self.get_logger("SystemInfo")

        logger.info("=== 系统信息 ===")
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"平台: {sys.platform}")
        logger.info(f"工作目录: {os.getcwd()}")
        logger.info(f"启动时间: {datetime.now().isoformat()}")

        # 记录环境变量（敏感信息除外）
        sensitive_keys = {'password', 'token', 'key', 'secret', 'api'}
        env_info = {}
        for key, value in os.environ.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                env_info[key] = "***"
            else:
                env_info[key] = value

        logger.debug(f"环境变量: {env_info}")


# 全局日志管理器实例
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """获取全局日志管理器实例"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def setup_logging(**kwargs) -> None:
    """设置日志配置（便捷函数）"""
    get_logging_manager().setup_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """获取日志器（便捷函数）"""
    return get_logging_manager().get_logger(name)


def log_exception(logger: logging.Logger, message: str = "发生异常") -> None:
    """记录异常信息

    Args:
        logger: 日志器
        message: 异常消息
    """
    logger.exception(message)


def log_performance(logger: logging.Logger, operation: str, duration: float) -> None:
    """记录性能信息

    Args:
        logger: 日志器
        operation: 操作名称
        duration: 持续时间（秒）
    """
    if duration > 1.0:
        logger.warning(f"性能警告: {operation} 耗时 {duration:.2f}秒")
    else:
        logger.debug(f"性能: {operation} 耗时 {duration:.3f}秒")


class PerformanceTimer:
    """性能计时器上下文管理器"""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"开始: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            log_performance(self.logger, self.operation, duration)

        if exc_type is not None:
            self.logger.error(f"操作失败: {self.operation}, 异常: {exc_val}")


def performance_timer(logger: logging.Logger, operation: str) -> PerformanceTimer:
    """创建性能计时器

    Args:
        logger: 日志器
        operation: 操作名称

    Returns:
        性能计时器上下文管理器
    """
    return PerformanceTimer(logger, operation)


# 装饰器
def log_function_call(logger: Optional[logging.Logger] = None):
    """函数调用日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            logger.debug(f"调用函数: {func.__name__}")
            try:
                with performance_timer(logger, func.__name__):
                    result = func(*args, **kwargs)
                logger.debug(f"函数完成: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"函数异常: {func.__name__}, 错误: {e}")
                raise

        return wrapper
    return decorator
