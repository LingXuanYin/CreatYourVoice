"""合成历史管理

这个模块负责语音合成历史记录的保存、加载和管理。
设计原则：
1. 完整记录 - 保存所有合成参数，支持重现
2. 高效查询 - 支持按时间、音色、文本等条件搜索
3. 存储优化 - 情感参考音频转换为向量后保存
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging

from .voice_synthesizer import SynthesisResult, SynthesisParams
from .emotion_manager import EmotionVector, EmotionManager

logger = logging.getLogger(__name__)


@dataclass
class HistoryRecord:
    """历史记录条目"""
    # 基础信息
    record_id: str
    synthesis_id: str
    created_at: datetime

    # 合成参数（情感参考音频已转换为向量）
    text: str
    voice_id: str
    voice_name: str
    emotion_mode: str
    emotion_vector: List[float]  # 最终使用的情感向量
    emotion_preset: Optional[str] = None
    emotion_text: Optional[str] = None

    # 生成参数
    speed: float = 1.0
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30

    # 结果信息
    success: bool = False
    error_message: Optional[str] = None
    audio_path: Optional[str] = None
    processing_time: float = 0.0
    segments_count: int = 0
    text_length: int = 0

    # 元数据
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    favorite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryRecord":
        """从字典创建记录"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class SynthesisHistoryError(Exception):
    """合成历史异常基类"""
    pass


class SynthesisHistory:
    """合成历史管理器

    负责合成历史的持久化存储和管理。
    存储结构：
    history_dir/
    ├── records/
    │   ├── 2024/
    │   │   ├── 01/
    │   │   │   ├── record_001.json
    │   │   │   └── record_002.json
    │   │   └── 02/
    │   └── index.json  # 索引文件
    └── audio/
        ├── 2024/
        │   ├── 01/
        │   │   ├── synthesis_001.wav
        │   │   └── synthesis_002.wav
        │   └── 02/
    """

    def __init__(
        self,
        history_dir: Union[str, Path] = "data/synthesis_history",
        emotion_manager: Optional[EmotionManager] = None
    ):
        """初始化历史管理器

        Args:
            history_dir: 历史记录存储目录
            emotion_manager: 情感管理器，用于处理情感参考音频
        """
        self.history_dir = Path(history_dir)
        self.records_dir = self.history_dir / "records"
        self.audio_dir = self.history_dir / "audio"
        self.index_file = self.history_dir / "index.json"

        self.emotion_manager = emotion_manager or EmotionManager()

        # 创建目录
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        # 内存索引
        self._index: Dict[str, Dict[str, Any]] = {}
        self._index_dirty = True

        logger.info(f"合成历史管理器初始化完成，存储目录: {self.history_dir}")

    def save_record(self, result: SynthesisResult) -> HistoryRecord:
        """保存合成记录

        Args:
            result: 合成结果

        Returns:
            历史记录
        """
        if not result.synthesis_params:
            raise SynthesisHistoryError("合成结果缺少参数信息")

        try:
            # 处理情感参考音频转向量
            emotion_vector = self._process_emotion_for_storage(result.synthesis_params)

            # 创建历史记录
            record = HistoryRecord(
                record_id=self._generate_record_id(),
                synthesis_id=result.synthesis_id,
                created_at=result.created_at,
                text=result.synthesis_params.text,
                voice_id=result.synthesis_params.voice_id,
                voice_name=result.voice_config.name if result.voice_config else "未知",
                emotion_mode=result.synthesis_params.emotion_mode,
                emotion_vector=emotion_vector.to_list(),
                emotion_preset=result.synthesis_params.emotion_preset,
                emotion_text=result.synthesis_params.emotion_text,
                speed=result.synthesis_params.speed,
                temperature=result.synthesis_params.temperature,
                top_p=result.synthesis_params.top_p,
                top_k=result.synthesis_params.top_k,
                success=result.success,
                error_message=result.error_message,
                processing_time=result.processing_time,
                segments_count=result.segments_count,
                text_length=result.text_length
            )

            # 保存音频文件
            if result.audio_path and Path(result.audio_path).exists():
                audio_path = self._save_audio_file(result.audio_path, record.record_id)
                record.audio_path = str(audio_path)

            # 保存记录文件
            self._save_record_file(record)

            # 更新索引
            self._update_index(record)

            logger.info(f"合成记录保存成功: {record.record_id}")
            return record

        except Exception as e:
            raise SynthesisHistoryError(f"保存合成记录失败: {e}")

    def load_record(self, record_id: str) -> Optional[HistoryRecord]:
        """加载历史记录

        Args:
            record_id: 记录ID

        Returns:
            历史记录，如果不存在则返回None
        """
        try:
            record_path = self._get_record_path(record_id)
            if not record_path.exists():
                return None

            with open(record_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return HistoryRecord.from_dict(data)

        except Exception as e:
            logger.error(f"加载历史记录失败: {record_id}, 错误: {e}")
            return None

    def delete_record(self, record_id: str) -> bool:
        """删除历史记录

        Args:
            record_id: 记录ID

        Returns:
            是否成功删除
        """
        try:
            # 加载记录
            record = self.load_record(record_id)
            if not record:
                return False

            # 删除音频文件
            if record.audio_path and Path(record.audio_path).exists():
                Path(record.audio_path).unlink()

            # 删除记录文件
            record_path = self._get_record_path(record_id)
            if record_path.exists():
                record_path.unlink()

            # 更新索引
            self._remove_from_index(record_id)

            logger.info(f"历史记录删除成功: {record_id}")
            return True

        except Exception as e:
            logger.error(f"删除历史记录失败: {record_id}, 错误: {e}")
            return False

    def search_records(
        self,
        text_pattern: Optional[str] = None,
        voice_id: Optional[str] = None,
        emotion_mode: Optional[str] = None,
        success_only: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        favorite_only: bool = False,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[HistoryRecord]:
        """搜索历史记录

        Args:
            text_pattern: 文本模式匹配
            voice_id: 音色ID
            emotion_mode: 情感模式
            success_only: 只返回成功的记录
            start_date: 开始日期
            end_date: 结束日期
            tags: 标签列表
            favorite_only: 只返回收藏的记录
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            匹配的历史记录列表
        """
        try:
            self._refresh_index()

            results = []
            for record_info in self._index.values():
                # 加载完整记录
                record = self.load_record(record_info['record_id'])
                if not record:
                    continue

                # 应用过滤条件
                if text_pattern and text_pattern.lower() not in record.text.lower():
                    continue

                if voice_id and record.voice_id != voice_id:
                    continue

                if emotion_mode and record.emotion_mode != emotion_mode:
                    continue

                if success_only and not record.success:
                    continue

                if start_date and record.created_at < start_date:
                    continue

                if end_date and record.created_at > end_date:
                    continue

                if tags and not any(tag in record.tags for tag in tags):
                    continue

                if favorite_only and not record.favorite:
                    continue

                results.append(record)

            # 按时间倒序排序
            results.sort(key=lambda x: x.created_at, reverse=True)

            # 应用分页
            if offset > 0:
                results = results[offset:]
            if limit is not None:
                results = results[:limit]

            return results

        except Exception as e:
            logger.error(f"搜索历史记录失败: {e}")
            return []

    def get_recent_records(self, days: int = 7, limit: int = 50) -> List[HistoryRecord]:
        """获取最近的记录

        Args:
            days: 天数
            limit: 数量限制

        Returns:
            最近的历史记录列表
        """
        start_date = datetime.now() - timedelta(days=days)
        return self.search_records(
            start_date=start_date,
            limit=limit
        )

    def get_favorites(self, limit: Optional[int] = None) -> List[HistoryRecord]:
        """获取收藏的记录

        Args:
            limit: 数量限制

        Returns:
            收藏的历史记录列表
        """
        return self.search_records(favorite_only=True, limit=limit)

    def update_record_metadata(
        self,
        record_id: str,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        favorite: Optional[bool] = None
    ) -> bool:
        """更新记录元数据

        Args:
            record_id: 记录ID
            tags: 标签列表
            notes: 备注
            favorite: 是否收藏

        Returns:
            是否成功更新
        """
        try:
            record = self.load_record(record_id)
            if not record:
                return False

            # 更新元数据
            if tags is not None:
                record.tags = tags
            if notes is not None:
                record.notes = notes
            if favorite is not None:
                record.favorite = favorite

            # 保存记录
            self._save_record_file(record)
            self._update_index(record)

            logger.info(f"记录元数据更新成功: {record_id}")
            return True

        except Exception as e:
            logger.error(f"更新记录元数据失败: {record_id}, 错误: {e}")
            return False

    def recreate_synthesis_params(self, record: HistoryRecord) -> SynthesisParams:
        """从历史记录重建合成参数

        Args:
            record: 历史记录

        Returns:
            合成参数
        """
        return SynthesisParams(
            text=record.text,
            voice_id=record.voice_id,
            emotion_mode=record.emotion_mode,
            emotion_vector=record.emotion_vector,
            emotion_preset=record.emotion_preset,
            emotion_text=record.emotion_text,
            speed=record.speed,
            temperature=record.temperature,
            top_p=record.top_p,
            top_k=record.top_k
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        try:
            self._refresh_index()

            total_records = len(self._index)
            if total_records == 0:
                return {
                    "total_records": 0,
                    "success_rate": 0.0,
                    "avg_processing_time": 0.0,
                    "most_used_voices": [],
                    "emotion_mode_distribution": {},
                    "recent_activity": {}
                }

            # 加载所有记录进行统计
            records = []
            for record_info in self._index.values():
                record = self.load_record(record_info['record_id'])
                if record:
                    records.append(record)

            # 计算统计信息
            success_count = sum(1 for r in records if r.success)
            success_rate = success_count / len(records) if records else 0.0

            processing_times = [r.processing_time for r in records if r.success]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0

            # 音色使用统计
            voice_counts = {}
            for record in records:
                voice_name = record.voice_name
                voice_counts[voice_name] = voice_counts.get(voice_name, 0) + 1

            most_used_voices = sorted(voice_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # 情感模式分布
            emotion_counts = {}
            for record in records:
                mode = record.emotion_mode
                emotion_counts[mode] = emotion_counts.get(mode, 0) + 1

            # 最近活动
            now = datetime.now()
            recent_activity = {}
            for i in range(7):
                date = now - timedelta(days=i)
                date_key = date.strftime("%Y-%m-%d")
                count = sum(1 for r in records if r.created_at.date() == date.date())
                recent_activity[date_key] = count

            return {
                "total_records": total_records,
                "success_rate": success_rate,
                "avg_processing_time": avg_processing_time,
                "most_used_voices": most_used_voices,
                "emotion_mode_distribution": emotion_counts,
                "recent_activity": recent_activity
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def cleanup_old_records(self, days: int = 90) -> int:
        """清理旧记录

        Args:
            days: 保留天数

        Returns:
            清理的记录数量
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            old_records = self.search_records(end_date=cutoff_date)

            cleaned_count = 0
            for record in old_records:
                if self.delete_record(record.record_id):
                    cleaned_count += 1

            logger.info(f"清理旧记录完成: {cleaned_count} 条")
            return cleaned_count

        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            return 0

    def _process_emotion_for_storage(self, params: SynthesisParams) -> EmotionVector:
        """处理情感参数用于存储（将参考音频转换为向量）"""
        if params.emotion_mode == "vector" and params.emotion_vector:
            return EmotionVector.from_list(params.emotion_vector)

        elif params.emotion_mode == "reference" and params.emotion_reference_audio:
            # 将情感参考音频转换为向量
            return self.emotion_manager.extract_emotion_from_audio(params.emotion_reference_audio)

        elif params.emotion_mode == "text" and params.emotion_text:
            return self.emotion_manager.analyze_emotion_from_text(params.emotion_text)

        elif params.emotion_mode == "preset" and params.emotion_preset:
            preset = self.emotion_manager.get_preset(params.emotion_preset)
            if preset:
                return preset.emotion_vector

        # 默认平静状态
        return EmotionVector(calm=1.0)

    def _generate_record_id(self) -> str:
        """生成记录ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        short_uuid = str(uuid.uuid4())[:8]
        return f"record_{timestamp}_{short_uuid}"

    def _get_record_path(self, record_id: str) -> Path:
        """获取记录文件路径"""
        # 按年月组织目录结构
        now = datetime.now()
        year_dir = self.records_dir / str(now.year)
        month_dir = year_dir / f"{now.month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)

        return month_dir / f"{record_id}.json"

    def _save_record_file(self, record: HistoryRecord) -> None:
        """保存记录文件"""
        record_path = self._get_record_path(record.record_id)
        record_path.parent.mkdir(parents=True, exist_ok=True)

        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

    def _save_audio_file(self, source_path: Union[str, Path], record_id: str) -> Path:
        """保存音频文件"""
        source_path = Path(source_path)

        # 按年月组织目录结构
        now = datetime.now()
        year_dir = self.audio_dir / str(now.year)
        month_dir = year_dir / f"{now.month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)

        # 保持原始扩展名
        target_path = month_dir / f"{record_id}{source_path.suffix}"

        # 复制文件
        shutil.copy2(source_path, target_path)

        return target_path

    def _refresh_index(self) -> None:
        """刷新索引"""
        if not self._index_dirty:
            return

        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            else:
                self._index = {}

            self._index_dirty = False

        except Exception as e:
            logger.error(f"刷新索引失败: {e}")
            self._index = {}
            self._index_dirty = False

    def _update_index(self, record: HistoryRecord) -> None:
        """更新索引"""
        self._refresh_index()

        self._index[record.record_id] = {
            "record_id": record.record_id,
            "created_at": record.created_at.isoformat(),
            "voice_id": record.voice_id,
            "voice_name": record.voice_name,
            "emotion_mode": record.emotion_mode,
            "success": record.success,
            "favorite": record.favorite,
            "text_length": record.text_length
        }

        self._save_index()

    def _remove_from_index(self, record_id: str) -> None:
        """从索引中移除"""
        self._refresh_index()

        if record_id in self._index:
            del self._index[record_id]
            self._save_index()

    def _save_index(self) -> None:
        """保存索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存索引失败: {e}")


# 便捷函数
def save_synthesis_result(result: SynthesisResult, history_dir: Optional[Union[str, Path]] = None) -> HistoryRecord:
    """便捷的合成结果保存函数"""
    if history_dir is None:
        history_dir = "data/synthesis_history"
    history = SynthesisHistory(history_dir=history_dir)
    return history.save_record(result)


def search_synthesis_history(
    text_pattern: Optional[str] = None,
    voice_id: Optional[str] = None,
    limit: int = 20,
    history_dir: Optional[Union[str, Path]] = None
) -> List[HistoryRecord]:
    """便捷的历史搜索函数"""
    if history_dir is None:
        history_dir = "data/synthesis_history"
    history = SynthesisHistory(history_dir=history_dir)
    return history.search_records(
        text_pattern=text_pattern,
        voice_id=voice_id,
        limit=limit
    )
