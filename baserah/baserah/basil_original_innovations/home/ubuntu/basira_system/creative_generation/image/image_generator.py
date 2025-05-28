#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة توليد الصور لنظام بصيرة

هذا الملف يحدد وحدة توليد الصور لنظام بصيرة، التي تستخدم تقنيات الذكاء الاصطناعي
لتوليد صور بناءً على المدخلات النصية والدلالية من النظام الأساسي.

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
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
import base64
import io
import random
import time
from enum import Enum, auto
from abc import ABC, abstractmethod

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import SemanticVector
from knowledge_extraction_generation import KnowledgeExtractor

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_image_generator')


class ImageGenerationMode(Enum):
    """أنماط توليد الصور."""
    TEXT_TO_IMAGE = auto()  # توليد من نص
    IMAGE_TO_IMAGE = auto()  # تعديل صورة موجودة
    CONCEPT_TO_IMAGE = auto()  # توليد من مفهوم
    SEMANTIC_TO_IMAGE = auto()  # توليد من متجه دلالي
    STYLE_TRANSFER = auto()  # نقل النمط
    HYBRID = auto()  # توليد هجين


@dataclass
class ImageGenerationConfig:
    """تكوين توليد الصور."""
    mode: ImageGenerationMode = ImageGenerationMode.TEXT_TO_IMAGE
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    input_image: Optional[Image.Image] = None
    input_image_strength: float = 0.8
    semantic_vector: Optional[SemanticVector] = None
    style_image: Optional[Image.Image] = None
    style_strength: float = 0.8
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageGenerationResult:
    """نتيجة توليد الصور."""
    image: Image.Image
    prompt: str
    generation_config: ImageGenerationConfig
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str) -> str:
        """
        حفظ الصورة المولدة.
        
        Args:
            path: مسار الحفظ
            
        Returns:
            مسار الملف المحفوظ
        """
        # التأكد من وجود المجلد
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # حفظ الصورة
        self.image.save(path)
        
        # حفظ البيانات الوصفية
        metadata_path = f"{os.path.splitext(path)[0]}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "prompt": self.prompt,
                "generation_config": {
                    "mode": self.generation_config.mode.name,
                    "width": self.generation_config.width,
                    "height": self.generation_config.height,
                    "num_inference_steps": self.generation_config.num_inference_steps,
                    "guidance_scale": self.generation_config.guidance_scale,
                    "seed": self.generation_config.seed
                },
                "generation_time": self.generation_time,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        return path
    
    def to_base64(self) -> str:
        """
        تحويل الصورة إلى تنسيق Base64.
        
        Returns:
            سلسلة Base64 للصورة
        """
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class ImageGeneratorBase(ABC):
    """الفئة الأساسية لمولد الصور."""
    
    def __init__(self):
        """تهيئة مولد الصور."""
        self.logger = logging.getLogger('basira_image_generator.base')
    
    @abstractmethod
    def generate(self, config: ImageGenerationConfig) -> ImageGenerationResult:
        """
        توليد صورة.
        
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


class MockImageGenerator(ImageGeneratorBase):
    """مولد صور وهمي للاختبار."""
    
    def __init__(self):
        """تهيئة مولد الصور الوهمي."""
        super().__init__()
        self.logger = logging.getLogger('basira_image_generator.mock')
    
    def generate(self, config: ImageGenerationConfig) -> ImageGenerationResult:
        """
        توليد صورة وهمية.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        self.logger.info(f"توليد صورة وهمية بنمط {config.mode.name}")
        
        # قياس وقت التوليد
        start_time = time.time()
        
        # إنشاء صورة وهمية
        if config.input_image is not None:
            # استخدام الصورة المدخلة إذا كانت متوفرة
            image = config.input_image.copy()
            # إضافة تأثير بسيط
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(0, 0), image.size], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 100))
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        else:
            # إنشاء صورة جديدة
            image = Image.new('RGB', (config.width, config.height), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            draw = ImageDraw.Draw(image)
            
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
            
            draw.text((10, 10), config.prompt[:50], fill=(255, 255, 255), font=font)
        
        # حساب وقت التوليد
        generation_time = time.time() - start_time
        
        return ImageGenerationResult(
            image=image,
            prompt=config.prompt,
            generation_config=config,
            generation_time=generation_time,
            metadata={
                "generator": "MockImageGenerator",
                "note": "هذا مولد وهمي للاختبار فقط"
            }
        )
    
    def is_available(self) -> bool:
        """
        التحقق من توفر المولد.
        
        Returns:
            True دائماً لأن المولد الوهمي متوفر دائماً
        """
        return True


class StableDiffusionGenerator(ImageGeneratorBase):
    """مولد صور باستخدام Stable Diffusion."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        تهيئة مولد Stable Diffusion.
        
        Args:
            model_id: معرف النموذج
        """
        super().__init__()
        self.logger = logging.getLogger('basira_image_generator.stable_diffusion')
        self.model_id = model_id
        self.pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.is_initialized = False
    
    def _initialize(self):
        """تهيئة النماذج."""
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
            
            # التحقق من توفر CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"استخدام جهاز: {device}")
            
            # تهيئة نموذج text-to-image
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id)
            self.pipe = self.pipe.to(device)
            
            # تهيئة نموذج image-to-image
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id)
            self.img2img_pipe = self.img2img_pipe.to(device)
            
            # تهيئة نموذج inpainting
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(self.model_id)
            self.inpaint_pipe = self.inpaint_pipe.to(device)
            
            self.is_initialized = True
            self.logger.info("تم تهيئة نماذج Stable Diffusion بنجاح")
        except Exception as e:
            self.logger.error(f"فشل في تهيئة نماذج Stable Diffusion: {e}")
            self.is_initialized = False
    
    def generate(self, config: ImageGenerationConfig) -> ImageGenerationResult:
        """
        توليد صورة باستخدام Stable Diffusion.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        # التحقق من تهيئة النماذج
        if not self.is_initialized:
            self._initialize()
            if not self.is_initialized:
                self.logger.warning("استخدام المولد الوهمي كبديل")
                return MockImageGenerator().generate(config)
        
        # قياس وقت التوليد
        start_time = time.time()
        
        # تعيين البذرة العشوائية إذا تم توفيرها
        generator = None
        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)
        
        try:
            # توليد الصورة حسب النمط
            if config.mode == ImageGenerationMode.TEXT_TO_IMAGE:
                # توليد من نص
                result = self.pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator
                )
                image = result.images[0]
            
            elif config.mode == ImageGenerationMode.IMAGE_TO_IMAGE:
                # تعديل صورة موجودة
                if config.input_image is None:
                    raise ValueError("يجب توفير صورة مدخلة لنمط IMAGE_TO_IMAGE")
                
                result = self.img2img_pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    image=config.input_image,
                    strength=config.input_image_strength,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator
                )
                image = result.images[0]
            
            elif config.mode == ImageGenerationMode.STYLE_TRANSFER:
                # نقل النمط
                if config.input_image is None or config.style_image is None:
                    raise ValueError("يجب توفير صورة مدخلة وصورة نمط لنمط STYLE_TRANSFER")
                
                # محاكاة نقل النمط باستخدام image-to-image
                # في التنفيذ الحقيقي، يمكن استخدام تقنيات أكثر تخصصاً لنقل النمط
                style_prompt = f"in the style of the reference image, {config.prompt}"
                result = self.img2img_pipe(
                    prompt=style_prompt,
                    negative_prompt=config.negative_prompt,
                    image=config.input_image,
                    strength=config.style_strength,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator
                )
                image = result.images[0]
            
            else:
                # الأنماط الأخرى (CONCEPT_TO_IMAGE, SEMANTIC_TO_IMAGE, HYBRID)
                # في التنفيذ الحقيقي، يمكن تنفيذ هذه الأنماط بشكل أكثر تخصصاً
                # هنا نستخدم TEXT_TO_IMAGE كبديل
                self.logger.warning(f"نمط {config.mode.name} غير مدعوم بالكامل، استخدام TEXT_TO_IMAGE كبديل")
                result = self.pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator
                )
                image = result.images[0]
        
        except Exception as e:
            self.logger.error(f"فشل في توليد الصورة: {e}")
            return MockImageGenerator().generate(config)
        
        # حساب وقت التوليد
        generation_time = time.time() - start_time
        
        return ImageGenerationResult(
            image=image,
            prompt=config.prompt,
            generation_config=config,
            generation_time=generation_time,
            metadata={
                "generator": "StableDiffusionGenerator",
                "model_id": self.model_id,
                "nsfw_content_detected": getattr(result, "nsfw_content_detected", False)
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


class DallEGenerator(ImageGeneratorBase):
    """مولد صور باستخدام DALL·E."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        تهيئة مولد DALL·E.
        
        Args:
            api_key: مفتاح API لـ OpenAI
        """
        super().__init__()
        self.logger = logging.getLogger('basira_image_generator.dalle')
        self.api_key = api_key
        self.client = None
        self.is_initialized = False
    
    def _initialize(self):
        """تهيئة العميل."""
        try:
            import openai
            
            # تعيين مفتاح API
            if self.api_key:
                openai.api_key = self.api_key
            
            # تهيئة العميل
            self.client = openai.OpenAI()
            
            self.is_initialized = True
            self.logger.info("تم تهيئة عميل DALL·E بنجاح")
        except Exception as e:
            self.logger.error(f"فشل في تهيئة عميل DALL·E: {e}")
            self.is_initialized = False
    
    def generate(self, config: ImageGenerationConfig) -> ImageGenerationResult:
        """
        توليد صورة باستخدام DALL·E.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        # التحقق من تهيئة العميل
        if not self.is_initialized:
            self._initialize()
            if not self.is_initialized:
                self.logger.warning("استخدام المولد الوهمي كبديل")
                return MockImageGenerator().generate(config)
        
        # قياس وقت التوليد
        start_time = time.time()
        
        try:
            # تحديد الحجم
            size = f"{config.width}x{config.height}"
            if config.width == config.height:
                if config.width == 1024:
                    size = "1024x1024"
                elif config.width == 512:
                    size = "512x512"
                else:
                    size = "256x256"
            
            # توليد الصورة
            if config.mode == ImageGenerationMode.TEXT_TO_IMAGE:
                # توليد من نص
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=config.prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
                
                # تنزيل الصورة
                image_url = response.data[0].url
                import requests
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
            
            else:
                # الأنماط الأخرى غير مدعومة حالياً في DALL·E API
                self.logger.warning(f"نمط {config.mode.name} غير مدعوم في DALL·E، استخدام TEXT_TO_IMAGE كبديل")
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=config.prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
                
                # تنزيل الصورة
                image_url = response.data[0].url
                import requests
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
        
        except Exception as e:
            self.logger.error(f"فشل في توليد الصورة: {e}")
            return MockImageGenerator().generate(config)
        
        # حساب وقت التوليد
        generation_time = time.time() - start_time
        
        return ImageGenerationResult(
            image=image,
            prompt=config.prompt,
            generation_config=config,
            generation_time=generation_time,
            metadata={
                "generator": "DallEGenerator",
                "model": "dall-e-3"
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
            import openai
            return self.api_key is not None
        except ImportError:
            return False


class SemanticImageGenerator:
    """مولد صور دلالي يدمج المتجهات الدلالية في عملية التوليد."""
    
    def __init__(self):
        """تهيئة مولد الصور الدلالي."""
        self.logger = logging.getLogger('basira_image_generator.semantic')
        
        # تهيئة المولدات الأساسية
        self.generators = {
            "mock": MockImageGenerator(),
            "stable_diffusion": StableDiffusionGenerator(),
            "dalle": DallEGenerator()
        }
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
    
    def _select_generator(self, config: ImageGenerationConfig) -> ImageGeneratorBase:
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
            return MockImageGenerator()
        
        # اختيار المولد حسب النمط
        if config.mode == ImageGenerationMode.TEXT_TO_IMAGE:
            # تفضيل DALL·E لتوليد النص إلى صورة
            if "dalle" in available_generators:
                return available_generators["dalle"]
            elif "stable_diffusion" in available_generators:
                return available_generators["stable_diffusion"]
        
        elif config.mode in [ImageGenerationMode.IMAGE_TO_IMAGE, ImageGenerationMode.STYLE_TRANSFER]:
            # تفضيل Stable Diffusion لتعديل الصور ونقل النمط
            if "stable_diffusion" in available_generators:
                return available_generators["stable_diffusion"]
        
        # استخدام أي مولد متوفر
        for name in ["stable_diffusion", "dalle", "mock"]:
            if name in available_generators:
                return available_generators[name]
        
        # استخدام المولد الوهمي كملاذ أخير
        return MockImageGenerator()
    
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
    
    def generate(self, config: ImageGenerationConfig) -> ImageGenerationResult:
        """
        توليد صورة باستخدام المعلومات الدلالية.
        
        Args:
            config: تكوين التوليد
            
        Returns:
            نتيجة التوليد
        """
        # تحسين الإرشاد باستخدام المعلومات الدلالية
        if config.semantic_vector is not None:
            config.prompt = self._enhance_prompt_with_semantics(config.prompt, config.semantic_vector)
        
        # اختيار المولد المناسب
        generator = self._select_generator(config)
        self.logger.info(f"استخدام مولد: {generator.__class__.__name__}")
        
        # توليد الصورة
        result = generator.generate(config)
        
        return result


# تنفيذ الاختبار إذا تم تشغيل الملف مباشرة
if __name__ == "__main__":
    # إنشاء مولد الصور الدلالي
    generator = SemanticImageGenerator()
    
    # إنشاء تكوين التوليد
    config = ImageGenerationConfig(
        mode=ImageGenerationMode.TEXT_TO_IMAGE,
        prompt="منظر طبيعي لجبال مغطاة بالثلوج تنعكس على بحيرة صافية",
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42
    )
    
    # توليد الصورة
    result = generator.generate(config)
    
    # عرض معلومات النتيجة
    print(f"تم توليد الصورة في {result.generation_time:.2f} ثانية")
    print(f"الإرشاد: {result.prompt}")
    
    # حفظ الصورة
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_image.png")
    result.save(output_path)
    print(f"تم حفظ الصورة في: {output_path}")
