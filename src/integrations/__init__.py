"""集成模块 - DDSP-SVC和IndexTTS集成"""

from .ddsp_svc import DDSPSVCIntegration
from .index_tts import IndexTTSIntegration

__all__ = [
    "DDSPSVCIntegration",
    "IndexTTSIntegration",
]
