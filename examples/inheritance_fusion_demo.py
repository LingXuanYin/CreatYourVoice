#!/usr/bin/env python3
"""音色继承和融合功能演示

这个脚本演示如何使用音色继承和融合功能。
包含以下示例：
1. 基础音色继承
2. 多音色融合
3. 继承链创建
4. 融合链创建
5. 复杂的继承-融合组合
"""

import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.core.voice_manager import VoiceManager
from src.core.voice_inheritance import VoiceInheritor, InheritanceConfig, InheritancePresetManager
from src.core.voice_fusion import VoiceFuser, FusionConfig, FusionSource, FusionPresetManager
from src.core.advanced_weight_calc import AdvancedWeightCalculator


def create_sample_voices(voice_manager: VoiceManager):
    """创建示例音色"""
    print("🎨 创建示例音色...")

    # 音色1：女性角色
    voice1 = VoiceConfig(
        voice_id="female_character_001",
        name="温柔女声",
        description="温柔甜美的女性角色音色",
        ddsp_config=DDSPSVCConfig(
            model_path="models/ddsp/female_model.pth",
            config_path="models/ddsp/female_config.yaml",
            speaker_id=0,
            f0_predictor="rmvpe",
            f0_min=80.0,
            f0_max=800.0,
            threhold=-60.0,
            spk_mix_dict={
                "female_speaker_1": 0.7,
                "female_speaker_2": 0.3
            }
        ),
        index_tts_config=IndexTTSConfig(
            model_path="models/index/female_model",
            config_path="models/index/female_config.yaml",
            speaker_name="gentle_female",
            emotion_strength=1.2,
            speed=0.9,
            temperature=0.6
        ),
        tags=["女性", "温柔", "甜美"]
    )

    # 音色2：男性角色
    voice2 = VoiceConfig(
        voice_id="male_character_001",
        name="成熟男声",
        description="成熟稳重的男性角色音色",
        ddsp_config=DDSPSVCConfig(
            model_path="models/ddsp/male_model.pth",
            config_path="models/ddsp/male_config.yaml",
            speaker_id=1,
            f0_predictor="fcpe",
            f0_min=60.0,
            f0_max=400.0,
            threhold=-55.0,
            spk_mix_dict={
                "male_speaker_1": 0.8,
                "male_speaker_2": 0.2
            }
        ),
        index_tts_config=IndexTTSConfig(
            model_path="models/index/male_model",
            config_path="models/index/male_config.yaml",
            speaker_name="mature_male",
            emotion_strength=0.8,
            speed=1.1,
            temperature=0.7
        ),
        tags=["男性", "成熟", "稳重"]
    )

    # 音色3：少女角色
    voice3 = VoiceConfig(
        voice_id="young_female_001",
        name="活泼少女",
        description="活泼可爱的少女角色音色",
        ddsp_config=DDSPSVCConfig(
            model_path="models/ddsp/young_female_model.pth",
            config_path="models/ddsp/young_female_config.yaml",
            speaker_id=2,
            f0_predictor="crepe",
            f0_min=100.0,
            f0_max=1000.0,
            threhold=-65.0,
            spk_mix_dict={
                "young_female_1": 0.6,
                "young_female_2": 0.4
            }
        ),
        index_tts_config=IndexTTSConfig(
            model_path="models/index/young_female_model",
            config_path="models/index/young_female_config.yaml",
            speaker_name="lively_girl",
            emotion_strength=1.5,
            speed=1.2,
            temperature=0.8
        ),
        tags=["少女", "活泼", "可爱"]
    )

    # 保存音色
    voices = [voice1, voice2, voice3]
    for voice in voices:
        voice_manager.save_voice(voice)
        print(f"  ✓ 已保存: {voice.name} ({voice.voice_id})")

    return voices


def demo_basic_inheritance(voice_manager: VoiceManager, parent_voice: VoiceConfig):
    """演示基础音色继承"""
    print("\n🧬 演示基础音色继承...")

    inheritor = VoiceInheritor(voice_manager)

    # 创建新的配置参数
    new_ddsp_config = DDSPSVCConfig(
        model_path="models/ddsp/inherited_model.pth",
        config_path="models/ddsp/inherited_config.yaml",
        speaker_id=10,
        f0_predictor="rmvpe",
        f0_min=70.0,
        f0_max=900.0,
        threhold=-58.0,
        spk_mix_dict={
            "new_speaker_1": 0.5,
            "new_speaker_2": 0.5
        }
    )

    new_index_config = IndexTTSConfig(
        model_path="models/index/inherited_model",
        config_path="models/index/inherited_config.yaml",
        speaker_name="inherited_speaker",
        emotion_strength=1.0,
        speed=1.0,
        temperature=0.75
    )

    # 使用平衡继承预设
    inheritance_config = InheritancePresetManager.get_balanced_preset()

    print(f"  📋 父音色: {parent_voice.name}")
    print(f"  📊 继承比例: {inheritance_config.inheritance_ratio:.1%}")

    # 执行继承
    result = inheritor.inherit_from_voice(
        parent_voice.voice_id,
        "继承音色示例",
        new_ddsp_config,
        new_index_config,
        inheritance_config
    )

    print(f"  ✅ 继承完成: {result.new_voice_config.name}")
    print(f"  ⏱️  处理时间: {result.processing_time:.3f}s")
    print(f"  🎯 DDSP权重: {len(result.inheritance_weights.ddsp_weights)} 个说话人")
    print(f"  ⚠️  警告数量: {len(result.warnings)}")

    # 保存继承结果
    voice_manager.save_voice(result.new_voice_config)

    return result.new_voice_config


def demo_multi_voice_fusion(voice_manager: VoiceManager, voices: list):
    """演示多音色融合"""
    print("\n🔀 演示多音色融合...")

    fuser = VoiceFuser(voice_manager)

    # 创建融合源
    fusion_sources = [
        FusionSource(voice_config=voices[0], weight=0.5, priority=1),  # 温柔女声
        FusionSource(voice_config=voices[1], weight=0.3, priority=2),  # 成熟男声
        FusionSource(voice_config=voices[2], weight=0.2, priority=3),  # 活泼少女
    ]

    # 使用平衡融合预设
    fusion_config = FusionPresetManager.get_balanced_preset()

    print("  📋 融合源:")
    for i, source in enumerate(fusion_sources, 1):
        print(f"    {i}. {source.voice_config.name} (权重: {source.weight:.1%})")

    # 预览融合结果
    preview = fuser.preview_fusion(fusion_sources, fusion_config)
    print(f"  🔍 预览: {len(preview['source_voices'])} 个源音色")
    print(f"  🎯 最终说话人数: {len(preview['fusion_weights']['ddsp'])}")

    # 执行融合
    result = fuser.fuse_voices(
        fusion_sources,
        "多音色融合示例",
        fusion_config
    )

    print(f"  ✅ 融合完成: {result.fused_voice_config.name}")
    print(f"  ⏱️  处理时间: {result.processing_time:.3f}s")
    print(f"  🎯 最终权重: {len(result.fusion_weights.combined_weights)} 个说话人")
    print(f"  🔧 解决冲突: {len(result.conflicts_resolved)} 个")

    # 保存融合结果
    voice_manager.save_voice(result.fused_voice_config)

    return result.fused_voice_config


def demo_inheritance_chain(voice_manager: VoiceManager, base_voice: VoiceConfig):
    """演示继承链"""
    print("\n🔗 演示继承链...")

    inheritor = VoiceInheritor(voice_manager)

    # 创建继承链配置
    voice_configs = [
        (base_voice, 0.8),  # 第一步：80%继承
        (base_voice, 0.6),  # 第二步：60%继承
        (base_voice, 0.4),  # 第三步：40%继承
    ]

    # 最终配置
    final_ddsp = DDSPSVCConfig(
        model_path="models/ddsp/chain_final.pth",
        config_path="models/ddsp/chain_final.yaml",
        speaker_id=20,
        spk_mix_dict={"chain_speaker": 1.0}
    )

    final_index = IndexTTSConfig(
        model_path="models/index/chain_final",
        config_path="models/index/chain_final.yaml",
        speaker_name="chain_final_speaker"
    )

    print(f"  📋 基础音色: {base_voice.name}")
    print(f"  🔗 继承步骤: {len(voice_configs)} 步")

    # 创建继承链
    results = inheritor.create_inheritance_chain(
        voice_configs,
        "继承链最终音色",
        final_ddsp,
        final_index
    )

    print(f"  ✅ 继承链完成: {len(results)} 步")
    for i, result in enumerate(results, 1):
        print(f"    步骤 {i}: {result.new_voice_config.name}")

    return results[-1].new_voice_config if results else None


def demo_complex_workflow(voice_manager: VoiceManager, voices: list):
    """演示复杂的继承-融合工作流"""
    print("\n🌟 演示复杂工作流 (继承 + 融合)...")

    inheritor = VoiceInheritor(voice_manager)
    fuser = VoiceFuser(voice_manager)

    # 第一步：从女性音色继承创建变体
    print("  步骤1: 创建女性音色变体...")
    female_variant_ddsp = DDSPSVCConfig(
        model_path="models/ddsp/female_variant.pth",
        config_path="models/ddsp/female_variant.yaml",
        speaker_id=30,
        spk_mix_dict={"female_variant": 1.0}
    )

    female_variant_index = IndexTTSConfig(
        model_path="models/index/female_variant",
        config_path="models/index/female_variant.yaml",
        speaker_name="female_variant_speaker"
    )

    female_variant_result = inheritor.inherit_from_voice(
        voices[0].voice_id,  # 温柔女声
        "女性音色变体",
        female_variant_ddsp,
        female_variant_index,
        InheritancePresetManager.get_innovative_preset()  # 创新继承
    )
    voice_manager.save_voice(female_variant_result.new_voice_config)
    print(f"    ✓ 创建变体: {female_variant_result.new_voice_config.name}")

    # 第二步：从男性音色继承创建变体
    print("  步骤2: 创建男性音色变体...")
    male_variant_ddsp = DDSPSVCConfig(
        model_path="models/ddsp/male_variant.pth",
        config_path="models/ddsp/male_variant.yaml",
        speaker_id=31,
        spk_mix_dict={"male_variant": 1.0}
    )

    male_variant_index = IndexTTSConfig(
        model_path="models/index/male_variant",
        config_path="models/index/male_variant.yaml",
        speaker_name="male_variant_speaker"
    )

    male_variant_result = inheritor.inherit_from_voice(
        voices[1].voice_id,  # 成熟男声
        "男性音色变体",
        male_variant_ddsp,
        male_variant_index,
        InheritancePresetManager.get_conservative_preset()  # 保守继承
    )
    voice_manager.save_voice(male_variant_result.new_voice_config)
    print(f"    ✓ 创建变体: {male_variant_result.new_voice_config.name}")

    # 第三步：融合所有变体
    print("  步骤3: 融合所有音色变体...")
    complex_fusion_sources = [
        FusionSource(voice_config=voices[0], weight=0.25),  # 原始女声
        FusionSource(voice_config=female_variant_result.new_voice_config, weight=0.25),  # 女性变体
        FusionSource(voice_config=voices[1], weight=0.25),  # 原始男声
        FusionSource(voice_config=male_variant_result.new_voice_config, weight=0.25),  # 男性变体
    ]

    complex_fusion_result = fuser.fuse_voices(
        complex_fusion_sources,
        "复杂工作流最终音色",
        FusionPresetManager.get_conservative_preset()
    )
    voice_manager.save_voice(complex_fusion_result.fused_voice_config)

    print(f"  ✅ 复杂工作流完成: {complex_fusion_result.fused_voice_config.name}")
    print(f"  📊 融合了 {len(complex_fusion_sources)} 个音色")
    print(f"  🎯 最终说话人数: {len(complex_fusion_result.fusion_weights.combined_weights)}")

    return complex_fusion_result.fused_voice_config


def demo_weight_analysis(voice_manager: VoiceManager):
    """演示权重分析功能"""
    print("\n📊 演示权重分析...")

    calculator = AdvancedWeightCalculator()

    # 获取所有音色
    voices = voice_manager.list_voices()
    print(f"  📋 分析 {len(voices)} 个音色的权重分布")

    for voice in voices[:3]:  # 只分析前3个
        ddsp_weights = voice.ddsp_config.spk_mix_dict or {}
        if ddsp_weights:
            print(f"\n  🎵 {voice.name}:")
            print(f"    说话人数量: {len(ddsp_weights)}")

            # 计算权重统计
            weights_list = list(ddsp_weights.values())
            max_weight = max(weights_list)
            min_weight = min(weights_list)
            avg_weight = sum(weights_list) / len(weights_list)

            print(f"    最大权重: {max_weight:.3f}")
            print(f"    最小权重: {min_weight:.3f}")
            print(f"    平均权重: {avg_weight:.3f}")

            # 找出主导说话人
            dominant_speaker = max(ddsp_weights.items(), key=lambda x: x[1])
            print(f"    主导说话人: {dominant_speaker[0]} ({dominant_speaker[1]:.3f})")


def main():
    """主演示函数"""
    print("🎵 音色继承和融合功能演示")
    print("=" * 50)

    # 创建临时目录用于演示
    temp_dir = tempfile.mkdtemp()
    print(f"📁 使用临时目录: {temp_dir}")

    try:
        # 初始化音色管理器
        voice_manager = VoiceManager(temp_dir)

        # 1. 创建示例音色
        sample_voices = create_sample_voices(voice_manager)

        # 2. 演示基础继承
        inherited_voice = demo_basic_inheritance(voice_manager, sample_voices[0])

        # 3. 演示多音色融合
        fused_voice = demo_multi_voice_fusion(voice_manager, sample_voices)

        # 4. 演示继承链
        chain_voice = demo_inheritance_chain(voice_manager, sample_voices[1])

        # 5. 演示复杂工作流
        complex_voice = demo_complex_workflow(voice_manager, sample_voices)

        # 6. 演示权重分析
        demo_weight_analysis(voice_manager)

        # 最终统计
        print("\n📈 最终统计:")
        all_voices = voice_manager.list_voices()
        print(f"  总音色数量: {len(all_voices)}")

        # 按标签分类
        tag_counts = {}
        for voice in all_voices:
            for tag in voice.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        print("  标签分布:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {tag}: {count} 个音色")

        print("\n✅ 演示完成！所有功能运行正常。")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 已清理临时目录: {temp_dir}")


if __name__ == "__main__":
    main()
