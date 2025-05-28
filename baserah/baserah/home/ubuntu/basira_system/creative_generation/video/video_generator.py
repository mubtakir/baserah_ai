#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة توليد الفيديو لنظام بصيرة

هذا الملف يحدد وحدة توليد الفيديو لنظام بصيرة، التي تستخدم تقنيات الذكاء الاصطناعي
لتوليد مقاطع فيديو بناءً على المدخلات النصية والدلالية من النظام الأساسي.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
from dataclasses import dataclass, field
import base64
import io
import random
import time
from enum import Enum, auto
from abc import ABC, abstractmethod
import tempfile
import subprocess
import shutil

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import SemanticVector
from knowledge_extraction_generation import KnowledgeExtractor
from creative_generation.image.image_generator import ImageGenerationConfig, ImageGenerationMode, SemanticImageGenerator

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_video_generator')


class VideoGenerationMode(Enum):
    """أنماط توليد الفيديو."""
    TEXT_TO_VIDEO = auto()  # توليد من نص
    IMAGE_TO_VIDEO = auto()  # توليد من صورة
    VIDEO_TO_VIDEO = auto()  # تعديل فيديو موجود
    CONCEPT_TO_VIDEO = auto()  # توليد من مفهوم
    SEMANTIC_TO_VIDEO = auto()  # توليد من متجه دلالي
    ANIMATION = auto()  # رسوم متحركة
    HYBRID = auto()  # توليد هجين


@dataclass
class VideoGenerationConfig:
    """تكوين توليد الفيديو."""
    mode: VideoGenerationMode = VideoGenerationMode.TEXT_TO_VIDEO
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_frames: int = 30
    fps: int = 30
    duration: float = 1.0  # بالثواني
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    input_images: Optional[List[Image.Image]] = None
    input_video_path: Optional[str] = None
    semantic_vector: Optional[SemanticVector] = None
    motion_vectors: Optional[List[Tuple[float, float]]] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerationResult:
    """نتيجة توليد الفيديو."""
    video_path: str
    prompt: str
    generation_config: VideoGenerationConfig
    generation_time: float
    preview_image: Optional[Image.Image] = None
    frames: Optional[List[Image.Image]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str) -> str:
        """
        حفظ الفيديو المولد.
        
        Args:
            path: مسار الحفظ
            
        Returns:
            مسار الملف المحفوظ
        """
        # التأكد من وجود المجلد
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # نسخ الفيديو
        shutil.copy(self.video_path, path)
        
        # حفظ البيانات الوصفية
        metadata_path = f"{os.path.splitext(path)[0]}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "prompt": self.prompt,
                "generation_config": {
                    "mode": self.generation_config.mode.name,
                    "width": self.generation_config.width,
                    "height": self.generation_config.height,
                    "num_frames": self.generation_config.num_frames,
                    "fps": self.generation_config.fps,
                    "duration": self.generation_config.duration,
                    "guidance_scale": self.generation_config.guidance_scale,
                    "seed": self.generation_config.seed
                },
                "generation_time": self.generation_time,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        # حفظ الصورة المصغرة إذا كانت متوفرة
        if self.preview_image:
            preview_path = f"{os.path.splitext(path)[0]}_preview.png"
            self.preview_image.save(preview_path)
        
        return path
    
    def to_base64(self) -> str:
        """
        تحويل الفيديو إلى تنسيق Base64.
        
        Returns:
            سلسلة Base64 للفيديو
        """
        with open(self.video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")


class VideoGeneratorBase(ABC):
    """الفئة الأساسية لمولد الفيديو."""
    
    def __init__(self):
        """تهيئة مولد الفيديو."""
        self.logger = logging.getLogger('basira_video_generator.base')
    
    @abstractmethod
    def generate(self, config: VideoGenerationConfig) -> VideoGenerationResult:
        """
        توليد فيديو.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        التحقق من توفر المولد.
        
        Returns:
            True إذا كان المولد متوفراً، وإلا False
        """
        pass


class MockVideoGenerator(VideoGeneratorBase):
    """مولد فيديو وهمي للاختبار."""
    
    def __init__(self):
        """تهيئة مولد الفيديو الوهمي."""
        super().__init__()
        self.logger = logging.getLogger('basira_video_generator.mock')
        self.image_generator = SemanticImageGenerator()
    
    def generate(self, config: VideoGenerationConfig) -> VideoGenerationResult:
        """
        توليد فيديو وهمي.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        self.logger.info(f"توليد فيديو وهمي بنمط {config.mode.name}")
        
        # قياس وقت التوليد
        start_time = time.time()
        
        # إنشاء مجلد مؤقت للإطارات
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # توليد الإطارات
        frames = []
        
        # عدد الإطارات
        num_frames = config.num_frames
        if num_frames <= 0:
            num_frames = int(config.duration * config.fps)
        
        for i in range(num_frames):
            # إنشاء تكوين توليد الصورة
            image_config = ImageGenerationConfig(
                mode=ImageGenerationMode.TEXT_TO_IMAGE,
                prompt=f"{config.prompt} - frame {i+1}",
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                seed=config.seed + i if config.seed is not None else None
            )
            
            # توليد الإطار
            result = self.image_generator.generate(image_config)
            frame = result.image
            
            # حفظ الإطار
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            frame.save(frame_path)
            
            frames.append(frame)
        
        # دمج الإطارات في فيديو
        output_path = os.path.join(temp_dir, "output.mp4")
        
        try:
            # استخدام FFmpeg لدمج الإطارات
            cmd = [
                "ffmpeg",
                "-y",  # الكتابة فوق الملف الموجود
                "-framerate", str(config.fps),
                "-i", os.path.join(frames_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # التحقق من وجود الفيديو
            if not os.path.exists(output_path):
                raise FileNotFoundError("فشل في إنشاء ملف الفيديو")
            
        except Exception as e:
            self.logger.error(f"فشل في دمج الإطارات: {e}")
            
            # إنشاء فيديو بسيط باستخدام Python
            from PIL import Image, ImageDraw, ImageFont
            import imageio
            
            # إنشاء إطارات بسيطة
            simple_frames = []
            for i in range(num_frames):
                img = Image.new('RGB', (config.width, config.height), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                draw = ImageDraw.Draw(img)
                
                # رسم بعض الأشكال العشوائية
                for _ in range(20):
                    x1 = random.randint(0, config.width)
                    y1 = random.randint(0, config.height)
                    x2 = random.randint(0, config.width)
                    y2 = random.randint(0, config.height)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=5)
                
                # إضافة نص الإرشاد
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), f"{config.prompt} - frame {i+1}", fill=(255, 255, 255), font=font)
                
                simple_frames.append(np.array(img))
            
            # حفظ الفيديو
            imageio.mimsave(output_path, simple_frames, fps=config.fps)
            
            # استخدام الإطارات البسيطة
            frames = [Image.fromarray(frame) for frame in simple_frames]
        
        # حساب وقت التوليد
        generation_time = time.time() - start_time
        
        return VideoGenerationResult(
            video_path=output_path,
            prompt=config.prompt,
            generation_config=config,
            generation_time=generation_time,
            preview_image=frames[0] if frames else None,
            frames=frames,
            metadata={
                "generator": "MockVideoGenerator",
                "note": "هذا مولد وهمي للاختبار فقط",
                "temp_dir": temp_dir  # للتنظيف لاحقاً
            }
        )
    
    def is_available(self) -> bool:
        """
        التحقق من توفر المولد.
        
        Returns:
            True دائماً لأن المولد الوهمي متوفر دائماً
        """
        return True


class StableVideoDiffusionGenerator(VideoGeneratorBase):
    """مولد فيديو باستخدام Stable Video Diffusion."""
    
    def __init__(self, model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"):
        """
        تهيئة مولد Stable Video Diffusion.
        
        Args:
            model_id: معرف النموذج
        """
        super().__init__()
        self.logger = logging.getLogger('basira_video_generator.stable_video_diffusion')
        self.model_id = model_id
        self.pipe = None
        self.is_initialized = False
    
    def _initialize(self):
        """تهيئة النموذج."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            # التحقق من توفر CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"استخدام جهاز: {device}")
            
            # تهيئة النموذج
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.pipe = self.pipe.to(device)
            
            self.is_initialized = True
            self.logger.info("تم تهيئة نموذج Stable Video Diffusion بنجاح")
        except Exception as e:
            self.logger.error(f"فشل في تهيئة نموذج Stable Video Diffusion: {e}")
            self.is_initialized = False
    
    def generate(self, config: VideoGenerationConfig) -> VideoGenerationResult:
        """
        توليد فيديو باستخدام Stable Video Diffusion.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        # التحقق من تهيئة النموذج
        if not self.is_initialized:
            self._initialize()
            if not self.is_initialized:
                self.logger.warning("استخدام المولد الوهمي كبديل")
                return MockVideoGenerator().generate(config)
        
        # التحقق من وجود صورة مدخلة
        if not config.input_images or len(config.input_images) == 0:
            self.logger.warning("Stable Video Diffusion يتطلب صورة مدخلة، استخدام المولد الوهمي كبديل")
            return MockVideoGenerator().generate(config)
        
        # قياس وقت التوليد
        start_time = time.time()
        
        # تعيين البذرة العشوائية إذا تم توفيرها
        generator = None
        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)
        
        try:
            # استخدام الصورة الأولى كمدخل
            image = config.input_images[0]
            
            # توليد الفيديو
            result = self.pipe(
                image,
                decode_chunk_size=8,
                generator=generator,
                motion_bucket_id=config.additional_params.get("motion_bucket_id", 127),
                noise_aug_strength=config.additional_params.get("noise_aug_strength", 0.1),
                num_frames=config.num_frames
            ).frames[0]
            
            # إنشاء مجلد مؤقت للإطارات
            temp_dir = tempfile.mkdtemp()
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # حفظ الإطارات
            frames = []
            for i, frame in enumerate(result):
                # تحويل التنسور إلى صورة PIL
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                    frame = (frame * 255).astype("uint8")
                    frame = Image.fromarray(frame)
                
                # حفظ الإطار
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                frame.save(frame_path)
                
                frames.append(frame)
            
            # دمج الإطارات في فيديو
            output_path = os.path.join(temp_dir, "output.mp4")
            
            # استخدام FFmpeg لدمج الإطارات
            cmd = [
                "ffmpeg",
                "-y",  # الكتابة فوق الملف الموجود
                "-framerate", str(config.fps),
                "-i", os.path.join(frames_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # التحقق من وجود الفيديو
            if not os.path.exists(output_path):
                raise FileNotFoundError("فشل في إنشاء ملف الفيديو")
            
        except Exception as e:
            self.logger.error(f"فشل في توليد الفيديو: {e}")
            return MockVideoGenerator().generate(config)
        
        # حساب وقت التوليد
        generation_time = time.time() - start_time
        
        return VideoGenerationResult(
            video_path=output_path,
            prompt=config.prompt,
            generation_config=config,
            generation_time=generation_time,
            preview_image=frames[0] if frames else None,
            frames=frames,
            metadata={
                "generator": "StableVideoDiffusionGenerator",
                "model_id": self.model_id,
                "temp_dir": temp_dir  # للتنظيف لاحقاً
            }
        )
    
    def is_available(self) -> bool:
        """
        التحقق من توفر المولد.
        
        Returns:
            True إذا كان المولد متوفراً، وإلا False
        """
        if self.is_initialized:
            return True
        
        try:
            import diffusers
            return True
        except ImportError:
            return False


class AnimationGenerator(VideoGeneratorBase):
    """مولد رسوم متحركة باستخدام تقنيات متعددة."""
    
    def __init__(self):
        """تهيئة مولد الرسوم المتحركة."""
        super().__init__()
        self.logger = logging.getLogger('basira_video_generator.animation')
        self.image_generator = SemanticImageGenerator()
    
    def generate(self, config: VideoGenerationConfig) -> VideoGenerationResult:
        """
        توليد رسوم متحركة.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        self.logger.info(f"توليد رسوم متحركة بنمط {config.mode.name}")
        
        # قياس وقت التوليد
        start_time = time.time()
        
        # إنشاء مجلد مؤقت للإطارات
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # عدد الإطارات
        num_frames = config.num_frames
        if num_frames <= 0:
            num_frames = int(config.duration * config.fps)
        
        # توليد الإطار الأول
        base_image_config = ImageGenerationConfig(
            mode=ImageGenerationMode.TEXT_TO_IMAGE,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            seed=config.seed
        )
        
        base_result = self.image_generator.generate(base_image_config)
        base_image = base_result.image
        
        # حفظ الإطار الأول
        base_image_path = os.path.join(frames_dir, "frame_0000.png")
        base_image.save(base_image_path)
        
        frames = [base_image]
        
        # توليد الإطارات التالية باستخدام image-to-image
        for i in range(1, num_frames):
            # تحديث الإرشاد للإطار الحالي
            frame_prompt = f"{config.prompt} - frame {i+1}/{num_frames}"
            
            # إنشاء تكوين توليد الصورة
            image_config = ImageGenerationConfig(
                mode=ImageGenerationMode.IMAGE_TO_IMAGE,
                prompt=frame_prompt,
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                seed=config.seed + i if config.seed is not None else None,
                input_image=frames[-1],
                input_image_strength=0.8  # قيمة أقل تعني تغيير أقل
            )
            
            # توليد الإطار
            result = self.image_generator.generate(image_config)
            frame = result.image
            
            # حفظ الإطار
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            frame.save(frame_path)
            
            frames.append(frame)
        
        # دمج الإطارات في فيديو
        output_path = os.path.join(temp_dir, "output.mp4")
        
        try:
            # استخدام FFmpeg لدمج الإطارات
            cmd = [
                "ffmpeg",
                "-y",  # الكتابة فوق الملف الموجود
                "-framerate", str(config.fps),
                "-i", os.path.join(frames_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # التحقق من وجود الفيديو
            if not os.path.exists(output_path):
                raise FileNotFoundError("فشل في إنشاء ملف الفيديو")
            
        except Exception as e:
            self.logger.error(f"فشل في دمج الإطارات: {e}")
            return MockVideoGenerator().generate(config)
        
        # حساب وقت التوليد
        generation_time = time.time() - start_time
        
        return VideoGenerationResult(
            video_path=output_path,
            prompt=config.prompt,
            generation_config=config,
            generation_time=generation_time,
            preview_image=frames[0],
            frames=frames,
            metadata={
                "generator": "AnimationGenerator",
                "temp_dir": temp_dir  # للتنظيف لاحقاً
            }
        )
    
    def is_available(self) -> bool:
        """
        التحقق من توفر المولد.
        
        Returns:
            True إذا كان المولد متوفراً، وإلا False
        """
        # التحقق من توفر FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except:
            return False


class SemanticVideoGenerator:
    """مولد فيديو دلالي يدمج المتجهات الدلالية في عملية التوليد."""
    
    def __init__(self):
        """تهيئة مولد الفيديو الدلالي."""
        self.logger = logging.getLogger('basira_video_generator.semantic')
        
        # تهيئة المولدات الأساسية
        self.generators = {
            "mock": MockVideoGenerator(),
            "stable_video_diffusion": StableVideoDiffusionGenerator(),
            "animation": AnimationGenerator()
        }
        
        # تهيئة مولد الصور
        self.image_generator = SemanticImageGenerator()
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
    
    def _select_generator(self, config: VideoGenerationConfig) -> VideoGeneratorBase:
        """
        اختيار المولد المناسب.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            المولد المناسب
        """
        # التحقق من توفر المولدات
        available_generators = {name: generator for name, generator in self.generators.items() if generator.is_available()}
        
        if not available_generators:
            self.logger.warning("لا توجد مولدات متوفرة، استخدام المولد الوهمي")
            return MockVideoGenerator()
        
        # اختيار المولد حسب النمط
        if config.mode == VideoGenerationMode.IMAGE_TO_VIDEO:
            # تفضيل Stable Video Diffusion لتحويل الصورة إلى فيديو
            if "stable_video_diffusion" in available_generators:
                return available_generators["stable_video_diffusion"]
        
        elif config.mode == VideoGenerationMode.ANIMATION:
            # تفضيل مولد الرسوم المتحركة
            if "animation" in available_generators:
                return available_generators["animation"]
        
        # استخدام أي مولد متوفر
        for name in ["stable_video_diffusion", "animation", "mock"]:
            if name in available_generators:
                return available_generators[name]
        
        # استخدام المولد الوهمي كملاذ أخير
        return MockVideoGenerator()
    
    def _enhance_prompt_with_semantics(self, prompt: str, semantic_vector: Optional[SemanticVector] = None) -> str:
        """
        تحسين الإرشاد باستخدام المعلومات الدلالية.
        
        Args:
            prompt: الإرشاد الأصلي
            semantic_vector: المتجه الدلالي
            
        Returns:
            الإرشاد المحسن
        """
        if semantic_vector is None:
            return prompt
        
        # استخراج الأبعاد الدلالية الرئيسية
        semantic_aspects = []
        for dim, value in semantic_vector.dimensions.items():
            if abs(value) > 0.5:  # اختيار الأبعاد ذات القيم العالية فقط
                aspect = dim.name.lower().replace('_', ' ')
                if value > 0:
                    semantic_aspects.append(f"with strong {aspect}")
                else:
                    semantic_aspects.append(f"with minimal {aspect}")
        
        # دمج الجوانب الدلالية مع الإرشاد
        if semantic_aspects:
            enhanced_prompt = f"{prompt}, {', '.join(semantic_aspects)}"
            self.logger.info(f"تم تحسين الإرشاد: {enhanced_prompt}")
            return enhanced_prompt
        
        return prompt
    
    def _generate_input_image_if_needed(self, config: VideoGenerationConfig) -> List[Image.Image]:
        """
        توليد صورة مدخلة إذا لزم الأمر.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            قائمة بالصور المدخلة
        """
        if config.input_images and len(config.input_images) > 0:
            return config.input_images
        
        # توليد صورة مدخلة
        image_config = ImageGenerationConfig(
            mode=ImageGenerationMode.TEXT_TO_IMAGE,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            seed=config.seed,
            semantic_vector=config.semantic_vector
        )
        
        result = self.image_generator.generate(image_config)
        return [result.image]
    
    def generate(self, config: VideoGenerationConfig) -> VideoGenerationResult:
        """
        توليد فيديو باستخدام المعلومات الدلالية.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        # تحسين الإرشاد باستخدام المعلومات الدلالية
        if config.semantic_vector is not None:
            config.prompt = self._enhance_prompt_with_semantics(config.prompt, config.semantic_vector)
        
        # توليد صورة مدخلة إذا لزم الأمر
        if config.mode in [VideoGenerationMode.IMAGE_TO_VIDEO, VideoGenerationMode.ANIMATION]:
            config.input_images = self._generate_input_image_if_needed(config)
        
        # اختيار المولد المناسب
        generator = self._select_generator(config)
        self.logger.info(f"استخدام مولد: {generator.__class__.__name__}")
        
        # توليد الفيديو
        result = generator.generate(config)
        
        return result


# تنفيذ الاختبار إذا تم تشغيل الملف مباشرة
if __name__ == "__main__":
    # إنشاء مولد الفيديو الدلالي
    generator = SemanticVideoGenerator()
    
    # إنشاء تكوين التوليد
    config = VideoGenerationConfig(
        mode=VideoGenerationMode.ANIMATION,
        prompt="منظر طبيعي لجبال مغطاة بالثلوج تنعكس على بحيرة صافية، مع غروب الشمس",
        width=512,
        height=512,
        num_frames=30,
        fps=30,
        duration=1.0,
        seed=42
    )
    
    # توليد الفيديو
    result = generator.generate(config)
    
    # عرض معلومات النتيجة
    print(f"تم توليد الفيديو في {result.generation_time:.2f} ثانية")
    print(f"الإرشاد: {result.prompt}")
    
    # حفظ الفيديو
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_video.mp4")
    result.save(output_path)
    print(f"تم حفظ الفيديو في: {output_path}")
