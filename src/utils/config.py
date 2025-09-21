"""配置管理模块

这个模块提供系统配置的管理功能。
设计原则：
1. 单一配置源 - 所有配置都在一个地方
2. 类型安全 - 使用类型注解和验证
3. 环境感知 - 支持不同环境的配置
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """配置异常"""
    pass


@dataclass
class DDSPSVCSettings:
    """DDSP-SVC设置"""
    model_dir: str = "DDSP-SVC/exp"
    default_f0_predictor: str = "rmvpe"
    default_f0_min: float = 50.0
    default_f0_max: float = 1100.0
    default_threshold: float = -60.0
    cache_dir: str = "DDSP-SVC/cache"
    max_audio_length: float = 30.0  # 最大音频长度（秒）


@dataclass
class IndexTTSSettings:
    """IndexTTS设置"""
    model_dir: str = "index-tts/checkpoints"
    config_file: str = "config.yaml"
    use_fp16: bool = False
    use_cuda_kernel: bool = False
    use_deepspeed: bool = False
    max_text_length: int = 500
    default_emotion_weight: float = 0.65
    default_max_tokens_per_segment: int = 120


@dataclass
class AudioSettings:
    """音频处理设置"""
    default_sample_rate: int = 22050
    supported_formats: List[str] = field(default_factory=lambda: ["wav", "mp3", "flac", "ogg"])
    max_file_size_mb: float = 100.0
    normalize_audio: bool = True
    trim_silence: bool = True
    silence_threshold_db: float = -40.0


@dataclass
class UISettings:
    """界面设置"""
    theme: str = "default"
    language: str = "zh_CN"
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    debug: bool = False
    max_concurrent_users: int = 10


@dataclass
class GPUSettings:
    """GPU设置"""
    auto_cleanup_enabled: bool = True
    cleanup_threshold_percent: float = 85.0
    idle_timeout_minutes: int = 30
    memory_monitoring_enabled: bool = True
    memory_monitoring_interval: float = 2.0
    max_memory_usage_percent: float = 90.0


@dataclass
class SystemSettings:
    """系统设置"""
    device: str = "auto"  # auto, cpu, cuda, mps
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/app.log"
    cache_dir: str = "cache"
    temp_dir: str = "temp"
    voices_dir: str = "voices"
    outputs_dir: str = "outputs"


@dataclass
class Config:
    """主配置类"""
    ddsp_svc: DDSPSVCSettings = field(default_factory=DDSPSVCSettings)
    index_tts: IndexTTSSettings = field(default_factory=IndexTTSSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    ui: UISettings = field(default_factory=UISettings)
    system: SystemSettings = field(default_factory=SystemSettings)
    gpu: GPUSettings = field(default_factory=GPUSettings)

    # 元数据
    version: str = "1.0.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        from datetime import datetime
        now = datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        self.updated_at = now

    def validate(self) -> List[str]:
        """验证配置

        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []

        # 验证目录路径
        required_dirs = [
            self.ddsp_svc.model_dir,
            self.index_tts.model_dir,
            self.system.cache_dir,
            self.system.temp_dir,
            self.system.voices_dir,
            self.system.outputs_dir
        ]

        for dir_path in required_dirs:
            if not dir_path or not isinstance(dir_path, str):
                errors.append(f"无效的目录路径: {dir_path}")

        # 验证数值范围
        if not (0 < self.ui.port < 65536):
            errors.append(f"端口号超出范围: {self.ui.port}")

        if not (0.0 < self.audio.max_file_size_mb <= 1000.0):
            errors.append(f"最大文件大小超出范围: {self.audio.max_file_size_mb}MB")

        if not (0.0 <= self.index_tts.default_emotion_weight <= 1.0):
            errors.append(f"情感权重超出范围: {self.index_tts.default_emotion_weight}")

        # 验证设备设置
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.system.device not in valid_devices:
            errors.append(f"无效的设备设置: {self.system.device}")

        # 验证日志级别
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.system.log_level not in valid_log_levels:
            errors.append(f"无效的日志级别: {self.system.log_level}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        # 创建子配置对象
        ddsp_svc = DDSPSVCSettings(**data.get("ddsp_svc", {}))
        index_tts = IndexTTSSettings(**data.get("index_tts", {}))
        audio = AudioSettings(**data.get("audio", {}))
        ui = UISettings(**data.get("ui", {}))
        system = SystemSettings(**data.get("system", {}))

        return cls(
            ddsp_svc=ddsp_svc,
            index_tts=index_tts,
            audio=audio,
            ui=ui,
            system=system,
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )


class ConfigManager:
    """配置管理器

    负责配置的加载、保存和管理。
    设计原则：
    1. 配置文件优先级：命令行参数 > 环境变量 > 配置文件 > 默认值
    2. 自动创建缺失的配置文件
    3. 配置验证和错误恢复
    """

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Optional[Config] = None

        logger.info(f"配置管理器初始化，配置文件: {self.config_path}")

    def load_config(self, create_if_missing: bool = True) -> Config:
        """加载配置

        Args:
            create_if_missing: 如果配置文件不存在是否创建默认配置

        Returns:
            配置对象

        Raises:
            ConfigError: 配置加载失败
        """
        try:
            if self.config_path.exists():
                logger.info(f"加载配置文件: {self.config_path}")

                # 根据文件扩展名选择解析器
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    raise ConfigError(f"不支持的配置文件格式: {self.config_path.suffix}")

                config = Config.from_dict(data or {})

            else:
                if create_if_missing:
                    logger.info("配置文件不存在，创建默认配置")
                    config = Config()
                    self.save_config(config)
                else:
                    raise ConfigError(f"配置文件不存在: {self.config_path}")

            # 验证配置
            errors = config.validate()
            if errors:
                logger.warning(f"配置验证发现问题: {errors}")
                # 可以选择是否抛出异常或使用默认值

            # 应用环境变量覆盖
            config = self._apply_env_overrides(config)

            self._config = config
            logger.info("配置加载成功")
            return config

        except Exception as e:
            raise ConfigError(f"加载配置失败: {e}")

    def save_config(self, config: Optional[Config] = None) -> None:
        """保存配置

        Args:
            config: 要保存的配置，None表示保存当前配置

        Raises:
            ConfigError: 配置保存失败
        """
        if config is None:
            config = self._config

        if config is None:
            raise ConfigError("没有可保存的配置")

        try:
            # 创建目录
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # 更新时间戳
            from datetime import datetime
            config.updated_at = datetime.now().isoformat()

            # 保存配置
            data = config.to_dict()

            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ConfigError(f"不支持的配置文件格式: {self.config_path.suffix}")

            logger.info(f"配置保存成功: {self.config_path}")

        except Exception as e:
            raise ConfigError(f"保存配置失败: {e}")

    def get_config(self) -> Config:
        """获取当前配置

        Returns:
            当前配置对象
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置

        Args:
            updates: 要更新的配置项
        """
        config = self.get_config()

        # 深度更新配置
        self._deep_update(config.to_dict(), updates)

        # 重新创建配置对象
        updated_config = Config.from_dict(config.to_dict())

        # 验证更新后的配置
        errors = updated_config.validate()
        if errors:
            raise ConfigError(f"配置更新后验证失败: {errors}")

        self._config = updated_config
        self.save_config()

    def _apply_env_overrides(self, config: Config) -> Config:
        """应用环境变量覆盖

        Args:
            config: 原始配置

        Returns:
            应用环境变量后的配置
        """
        # 定义环境变量映射
        env_mappings = {
            "CREATYOURVOICE_HOST": ("ui", "host"),
            "CREATYOURVOICE_PORT": ("ui", "port"),
            "CREATYOURVOICE_DEVICE": ("system", "device"),
            "CREATYOURVOICE_LOG_LEVEL": ("system", "log_level"),
            "CREATYOURVOICE_DEBUG": ("ui", "debug"),
        }

        config_dict = config.to_dict()

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 类型转换
                if key == "port":
                    value = int(value)
                elif key == "debug":
                    value = value.lower() in ("true", "1", "yes", "on")

                config_dict[section][key] = value
                logger.debug(f"环境变量覆盖: {env_var} -> {section}.{key} = {value}")

        return Config.from_dict(config_dict)

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def create_directories(self) -> None:
        """创建配置中指定的目录"""
        config = self.get_config()

        directories = [
            config.system.cache_dir,
            config.system.temp_dir,
            config.system.voices_dir,
            config.system.outputs_dir,
        ]

        # 添加日志目录
        if config.system.log_file:
            log_dir = Path(config.system.log_file).parent
            directories.append(str(log_dir))

        for dir_path in directories:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"目录创建成功: {dir_path}")
            except Exception as e:
                logger.warning(f"创建目录失败: {dir_path}, 错误: {e}")


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Union[str, Path] = "config.yaml") -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> Config:
    """获取当前配置"""
    return get_config_manager().get_config()


def save_config(config: Optional[Config] = None) -> None:
    """保存配置"""
    get_config_manager().save_config(config)


def update_config(updates: Dict[str, Any]) -> None:
    """更新配置"""
    get_config_manager().update_config(updates)
