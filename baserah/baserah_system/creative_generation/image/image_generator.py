#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generator for Basira System

This module implements the Image Generator, which generates images based on
textual descriptions, semantic concepts, or mathematical equations.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time

# Configure logging
logger = logging.getLogger('creative_generation.image.image_generator')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, some functionality will be limited")
    TORCH_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not available, some functionality will be limited")
    PIL_AVAILABLE = False


class GenerationMode(str, Enum):
    """Generation modes for the Image Generator."""
    TEXT_TO_IMAGE = "text_to_image"  # Generate image from text description
    ARABIC_TEXT_TO_IMAGE = "arabic_text_to_image"  # Generate image from Arabic text description
    CONCEPT_TO_IMAGE = "concept_to_image"  # Generate image from semantic concept
    EQUATION_TO_IMAGE = "equation_to_image"  # Generate image from mathematical equation
    STYLE_TRANSFER = "style_transfer"  # Apply style transfer to an existing image
    CALLIGRAPHY = "calligraphy"  # Generate Arabic calligraphy
    HYBRID = "hybrid"  # Combine multiple generation modes


@dataclass
class GenerationParameters:
    """Parameters for image generation."""
    mode: GenerationMode  # Generation mode
    width: int = 512  # Image width
    height: int = 512  # Image height
    seed: Optional[int] = None  # Random seed for reproducibility
    guidance_scale: float = 7.5  # Guidance scale for diffusion models
    num_inference_steps: int = 50  # Number of inference steps for diffusion models
    custom_parameters: Dict[str, Any] = field(default_factory=dict)  # Custom parameters


@dataclass
class GenerationResult:
    """Result of image generation."""
    image: Any  # Generated image (PIL Image or numpy array)
    parameters: GenerationParameters  # Parameters used for generation
    generation_time: float  # Time taken for generation (seconds)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class ImageGenerator:
    """
    Image Generator class for generating images based on textual descriptions,
    semantic concepts, or mathematical equations.

    This class implements various algorithms for image generation, including
    diffusion models, GANs, and neural style transfer.
    """

    def __init__(self,
                 models_path: Optional[str] = None,
                 config_file: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the Image Generator.

        Args:
            models_path: Path to the models directory (optional)
            config_file: Path to the configuration file (optional)
            device: Device to use for computation ("cpu" or "cuda")
        """
        self.logger = logging.getLogger('creative_generation.image.image_generator.main')

        # Set device
        self.device = device
        if TORCH_AVAILABLE:
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Load configuration
        self.config = self._load_config(config_file)

        # Set models path
        self.models_path = models_path or self.config.get("models_path", "./models")

        # Initialize models
        self.models = {}
        self._initialize_models()

        self.logger.info(f"Image Generator initialized with device: {self.device}")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use default configuration.

        Args:
            config_file: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        default_config = {
            "models_path": "./models",
            "default_mode": "text_to_image",
            "default_width": 512,
            "default_height": 512,
            "default_guidance_scale": 7.5,
            "default_num_inference_steps": 50,
            "models": {
                "text_to_image": {
                    "type": "diffusion",
                    "name": "stable-diffusion-v1-5",
                    "enabled": True
                },
                "concept_to_image": {
                    "type": "gan",
                    "name": "concept-gan",
                    "enabled": False
                },
                "equation_to_image": {
                    "type": "custom",
                    "name": "equation-renderer",
                    "enabled": True
                },
                "style_transfer": {
                    "type": "neural_style",
                    "name": "neural-style-transfer",
                    "enabled": False
                }
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # Merge user config with default config
                self._merge_configs(default_config, user_config)
                self.logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {e}")
                self.logger.info("Using default configuration")
        else:
            if config_file:
                self.logger.warning(f"Configuration file {config_file} not found, using default configuration")
            else:
                self.logger.info("No configuration file provided, using default configuration")

        return default_config

    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """
        Merge user configuration with default configuration.

        Args:
            default_config: Default configuration dictionary (modified in-place)
            user_config: User configuration dictionary
        """
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_configs(default_config[key], value)
            else:
                default_config[key] = value

    def _initialize_models(self) -> None:
        """Initialize image generation models."""
        for mode, model_config in self.config["models"].items():
            if model_config["enabled"]:
                try:
                    self._initialize_model(mode, model_config)
                except Exception as e:
                    self.logger.error(f"Error initializing model for mode {mode}: {e}")

    def _initialize_model(self, mode: str, model_config: Dict[str, Any]) -> None:
        """
        Initialize a specific image generation model.

        Args:
            mode: Generation mode
            model_config: Model configuration
        """
        model_type = model_config["type"]
        model_name = model_config["name"]

        self.logger.info(f"Initializing {model_type} model '{model_name}' for mode {mode}")

        if model_type == "diffusion" and TORCH_AVAILABLE:
            # Placeholder for diffusion model initialization
            # In a real implementation, this would load a pre-trained diffusion model
            self.models[mode] = {
                "type": model_type,
                "name": model_name,
                "model": self._create_dummy_diffusion_model()
            }
        elif model_type == "gan" and TORCH_AVAILABLE:
            # Placeholder for GAN model initialization
            # In a real implementation, this would load a pre-trained GAN model
            self.models[mode] = {
                "type": model_type,
                "name": model_name,
                "model": self._create_dummy_gan_model()
            }
        elif model_type == "neural_style" and TORCH_AVAILABLE:
            # Placeholder for neural style transfer model initialization
            # In a real implementation, this would load a pre-trained neural style transfer model
            self.models[mode] = {
                "type": model_type,
                "name": model_name,
                "model": self._create_dummy_style_transfer_model()
            }
        elif model_type == "custom":
            # Placeholder for custom model initialization
            # In a real implementation, this would load a custom model
            self.models[mode] = {
                "type": model_type,
                "name": model_name,
                "model": self._create_dummy_custom_model(mode)
            }
        else:
            self.logger.warning(f"Unsupported model type: {model_type}")

    def _create_dummy_diffusion_model(self) -> Optional[Any]:
        """
        Create a dummy diffusion model for demonstration purposes.

        Returns:
            Dummy diffusion model
        """
        if not TORCH_AVAILABLE:
            return None

        # Simple dummy model
        class DummyDiffusionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = torch.sigmoid(self.conv2(x))
                return x

            def generate(self, prompt, width, height, guidance_scale, num_inference_steps, seed=None):
                # Set seed if provided
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                # Generate random noise
                noise = torch.randn(1, 3, height, width)

                # Process through model
                result = self.forward(noise)

                # Convert to numpy array
                image = result[0].permute(1, 2, 0).detach().numpy()

                # Ensure values are in [0, 1]
                image = np.clip(image, 0, 1)

                return image

        model = DummyDiffusionModel()
        model.to(self.device)
        return model

    def _create_dummy_gan_model(self) -> Optional[Any]:
        """
        Create a dummy GAN model for demonstration purposes.

        Returns:
            Dummy GAN model
        """
        if not TORCH_AVAILABLE:
            return None

        # Simple dummy model
        class DummyGANModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(100, 256)
                self.fc2 = nn.Linear(256, 512)
                self.fc3 = nn.Linear(512, 1024)
                self.fc4 = nn.Linear(1024, 3 * 64 * 64)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = torch.sigmoid(self.fc4(x))
                return x.view(-1, 3, 64, 64)

            def generate(self, concept_vector, width, height, seed=None):
                # Set seed if provided
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                # Generate random noise
                noise = torch.randn(1, 100)

                # Process through model
                result = self.forward(noise)

                # Resize to desired dimensions
                if result.shape[2] != height or result.shape[3] != width:
                    result = F.interpolate(result, size=(height, width), mode='bilinear', align_corners=False)

                # Convert to numpy array
                image = result[0].permute(1, 2, 0).detach().numpy()

                # Ensure values are in [0, 1]
                image = np.clip(image, 0, 1)

                return image

        model = DummyGANModel()
        model.to(self.device)
        return model

    def _create_dummy_style_transfer_model(self) -> Optional[Any]:
        """
        Create a dummy style transfer model for demonstration purposes.

        Returns:
            Dummy style transfer model
        """
        if not TORCH_AVAILABLE:
            return None

        # Simple dummy model
        class DummyStyleTransferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

            def forward(self, content, style):
                x = F.relu(self.conv1(content))
                x = torch.sigmoid(self.conv2(x))
                return x

            def transfer_style(self, content_image, style_image, alpha=1.0, seed=None):
                # Set seed if provided
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                # Convert numpy arrays to tensors
                if isinstance(content_image, np.ndarray):
                    content_image = torch.from_numpy(content_image).permute(2, 0, 1).unsqueeze(0).float()

                if isinstance(style_image, np.ndarray):
                    style_image = torch.from_numpy(style_image).permute(2, 0, 1).unsqueeze(0).float()

                # Process through model
                result = self.forward(content_image, style_image)

                # Convert to numpy array
                image = result[0].permute(1, 2, 0).detach().numpy()

                # Ensure values are in [0, 1]
                image = np.clip(image, 0, 1)

                return image

        model = DummyStyleTransferModel()
        model.to(self.device)
        return model

    def _create_dummy_custom_model(self, mode: str) -> Any:
        """
        Create a dummy custom model for demonstration purposes.

        Args:
            mode: Generation mode

        Returns:
            Dummy custom model
        """
        if mode == "equation_to_image":
            # Simple equation renderer
            class EquationRenderer:
                def render(self, equation, width, height, seed=None):
                    # Set seed if provided
                    if seed is not None:
                        np.random.seed(seed)

                    # Create blank image
                    if PIL_AVAILABLE:
                        image = Image.new('RGB', (width, height), color='white')
                        draw = ImageDraw.Draw(image)

                        # Draw equation text
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except IOError:
                            font = ImageFont.load_default()

                        draw.text((width // 2 - 100, height // 2), str(equation), fill='black', font=font)

                        # Convert to numpy array
                        image_np = np.array(image) / 255.0
                        return image_np
                    else:
                        # Fallback to numpy
                        image = np.ones((height, width, 3))
                        return image

            return EquationRenderer()

        return None

    def generate_image(self,
                      input_data: Any,
                      parameters: Optional[GenerationParameters] = None) -> GenerationResult:
        """
        Generate an image based on input data and parameters.

        Args:
            input_data: Input data for generation (text, concept, equation, etc.)
            parameters: Generation parameters

        Returns:
            Generation result
        """
        # Set default parameters if not provided
        if parameters is None:
            mode = GenerationMode(self.config["default_mode"])
            parameters = GenerationParameters(
                mode=mode,
                width=self.config["default_width"],
                height=self.config["default_height"],
                guidance_scale=self.config["default_guidance_scale"],
                num_inference_steps=self.config["default_num_inference_steps"]
            )

        # Check if mode is supported
        mode_str = parameters.mode.value
        if mode_str not in self.models and mode_str != GenerationMode.CALLIGRAPHY.value and mode_str != GenerationMode.ARABIC_TEXT_TO_IMAGE.value:
            self.logger.error(f"Unsupported generation mode: {mode_str}")
            raise ValueError(f"Unsupported generation mode: {mode_str}")

        # Set seed if provided
        if parameters.seed is not None:
            np.random.seed(parameters.seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(parameters.seed)

        # Start timer
        start_time = time.time()

        # Generate image based on mode
        if parameters.mode == GenerationMode.TEXT_TO_IMAGE:
            image = self._generate_from_text(input_data, parameters)
        elif parameters.mode == GenerationMode.ARABIC_TEXT_TO_IMAGE:
            image = self._generate_from_arabic_text(input_data, parameters)
        elif parameters.mode == GenerationMode.CONCEPT_TO_IMAGE:
            image = self._generate_from_concept(input_data, parameters)
        elif parameters.mode == GenerationMode.EQUATION_TO_IMAGE:
            image = self._generate_from_equation(input_data, parameters)
        elif parameters.mode == GenerationMode.STYLE_TRANSFER:
            image = self._apply_style_transfer(input_data, parameters)
        elif parameters.mode == GenerationMode.CALLIGRAPHY:
            image = self._generate_calligraphy(input_data, parameters)
        elif parameters.mode == GenerationMode.HYBRID:
            image = self._generate_hybrid(input_data, parameters)
        else:
            self.logger.error(f"Unsupported generation mode: {parameters.mode}")
            raise ValueError(f"Unsupported generation mode: {parameters.mode}")

        # End timer
        end_time = time.time()
        generation_time = end_time - start_time

        # Create generation result
        result = GenerationResult(
            image=image,
            parameters=parameters,
            generation_time=generation_time,
            metadata={
                "mode": parameters.mode.value,
                "model": self.models.get(mode_str, {"name": "internal"})["name"]
            }
        )

        return result

    def _generate_from_text(self, text: str, parameters: GenerationParameters) -> np.ndarray:
        """
        Generate an image from text description.

        Args:
            text: Text description
            parameters: Generation parameters

        Returns:
            Generated image
        """
        self.logger.info(f"Generating image from text: {text}")

        model_info = self.models[parameters.mode.value]
        model = model_info["model"]

        if model_info["type"] == "diffusion" and TORCH_AVAILABLE:
            # Generate image using diffusion model
            image = model.generate(
                prompt=text,
                width=parameters.width,
                height=parameters.height,
                guidance_scale=parameters.guidance_scale,
                num_inference_steps=parameters.num_inference_steps,
                seed=parameters.seed
            )
        else:
            # Fallback to simple image generation
            image = np.random.rand(parameters.height, parameters.width, 3)

        return image

    def _generate_from_concept(self, concept_vector: np.ndarray, parameters: GenerationParameters) -> np.ndarray:
        """
        Generate an image from semantic concept.

        Args:
            concept_vector: Semantic concept vector
            parameters: Generation parameters

        Returns:
            Generated image
        """
        self.logger.info("Generating image from concept vector")

        model_info = self.models[parameters.mode.value]
        model = model_info["model"]

        if model_info["type"] == "gan" and TORCH_AVAILABLE:
            # Generate image using GAN model
            image = model.generate(
                concept_vector=concept_vector,
                width=parameters.width,
                height=parameters.height,
                seed=parameters.seed
            )
        else:
            # Fallback to simple image generation
            image = np.random.rand(parameters.height, parameters.width, 3)

        return image

    def _generate_from_equation(self, equation: Any, parameters: GenerationParameters) -> np.ndarray:
        """
        Generate an image from mathematical equation.

        Args:
            equation: Mathematical equation
            parameters: Generation parameters

        Returns:
            Generated image
        """
        self.logger.info(f"Generating image from equation: {equation}")

        model_info = self.models[parameters.mode.value]
        model = model_info["model"]

        if model_info["type"] == "custom":
            # Generate image using equation renderer
            image = model.render(
                equation=equation,
                width=parameters.width,
                height=parameters.height,
                seed=parameters.seed
            )
        else:
            # Fallback to simple image generation
            image = np.random.rand(parameters.height, parameters.width, 3)

        return image

    def _apply_style_transfer(self, input_data: Tuple[np.ndarray, np.ndarray], parameters: GenerationParameters) -> np.ndarray:
        """
        Apply style transfer to an image.

        Args:
            input_data: Tuple of (content_image, style_image)
            parameters: Generation parameters

        Returns:
            Stylized image
        """
        self.logger.info("Applying style transfer")

        content_image, style_image = input_data

        model_info = self.models[parameters.mode.value]
        model = model_info["model"]

        if model_info["type"] == "neural_style" and TORCH_AVAILABLE:
            # Apply style transfer
            alpha = parameters.custom_parameters.get("alpha", 1.0)
            image = model.transfer_style(
                content_image=content_image,
                style_image=style_image,
                alpha=alpha,
                seed=parameters.seed
            )
        else:
            # Fallback to simple image generation
            image = content_image.copy()

        return image

    def _generate_from_arabic_text(self, text: str, parameters: GenerationParameters) -> np.ndarray:
        """
        Generate an image from Arabic text description.

        Args:
            text: Arabic text description
            parameters: Generation parameters

        Returns:
            Generated image
        """
        self.logger.info(f"Generating image from Arabic text: {text}")

        # For now, we'll use the same approach as text_to_image
        # In a real implementation, this would use a model fine-tuned for Arabic text

        # Try to use the text_to_image model if available
        if GenerationMode.TEXT_TO_IMAGE.value in self.models:
            model_info = self.models[GenerationMode.TEXT_TO_IMAGE.value]
            model = model_info["model"]

            if model_info["type"] == "diffusion" and TORCH_AVAILABLE:
                # Generate image using diffusion model
                image = model.generate(
                    prompt=text,
                    width=parameters.width,
                    height=parameters.height,
                    guidance_scale=parameters.guidance_scale,
                    num_inference_steps=parameters.num_inference_steps,
                    seed=parameters.seed
                )
                return image

        # If no suitable model is available, create a simple image with the text
        if PIL_AVAILABLE:
            # Create a blank image
            image = Image.new('RGB', (parameters.width, parameters.height), color='white')
            draw = ImageDraw.Draw(image)

            # Try to load an Arabic font
            try:
                # Try to find a suitable font for Arabic text
                font_paths = [
                    "/usr/share/fonts/truetype/arabeyes/ae_Arab.ttf",  # Linux
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Linux
                    "/Library/Fonts/Arial Unicode.ttf",  # macOS
                    "C:\\Windows\\Fonts\\arial.ttf"  # Windows
                ]

                font = None
                for path in font_paths:
                    if os.path.exists(path):
                        font = ImageFont.truetype(path, 30)
                        break

                if font is None:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            # Draw text
            text_width = parameters.width - 40
            text_height = parameters.height - 40

            # Split text into lines
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + " " + word if current_line else word
                # Estimate width (this is not accurate for Arabic, but it's a simple approach)
                if len(test_line) * 15 < text_width:  # Rough estimate
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            # Draw each line
            y_position = 40
            for line in lines:
                draw.text((parameters.width - 20, y_position), line, fill='black', font=font, align='right')
                y_position += 40

            # Convert to numpy array
            image_np = np.array(image) / 255.0
            return image_np
        else:
            # Fallback to simple image generation
            return np.random.rand(parameters.height, parameters.width, 3)

    def _generate_calligraphy(self, text: str, parameters: GenerationParameters) -> np.ndarray:
        """
        Generate Arabic calligraphy.

        Args:
            text: Arabic text
            parameters: Generation parameters

        Returns:
            Generated calligraphy image
        """
        self.logger.info(f"Generating calligraphy for text: {text}")

        if PIL_AVAILABLE:
            # Create a blank image
            image = Image.new('RGB', (parameters.width, parameters.height), color='white')
            draw = ImageDraw.Draw(image)

            # Try to load an Arabic font
            try:
                # Try to find a suitable font for Arabic calligraphy
                font_paths = [
                    "/usr/share/fonts/truetype/arabeyes/ae_Arab.ttf",  # Linux
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Linux
                    "/Library/Fonts/Arial Unicode.ttf",  # macOS
                    "C:\\Windows\\Fonts\\arial.ttf"  # Windows
                ]

                font = None
                for path in font_paths:
                    if os.path.exists(path):
                        # Use a larger font size for calligraphy
                        font = ImageFont.truetype(path, 60)
                        break

                if font is None:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            # Calculate text position to center it
            text_width = parameters.width
            text_height = parameters.height

            # Draw text in the center
            draw.text((parameters.width // 2, parameters.height // 2), text, fill='black', font=font, anchor="mm")

            # Add decorative border
            border_width = 10
            draw.rectangle([(border_width, border_width), (parameters.width - border_width, parameters.height - border_width)],
                          outline='black', width=2)

            # Convert to numpy array
            image_np = np.array(image) / 255.0
            return image_np
        else:
            # Fallback to simple image generation
            return np.random.rand(parameters.height, parameters.width, 3)

    def _generate_hybrid(self, input_data: Dict[str, Any], parameters: GenerationParameters) -> np.ndarray:
        """
        Generate an image using hybrid approach.

        Args:
            input_data: Dictionary of input data for different modes
            parameters: Generation parameters

        Returns:
            Generated image
        """
        self.logger.info("Generating image using hybrid approach")

        # Check if input_data contains the necessary components
        if not isinstance(input_data, dict):
            self.logger.error("Input data must be a dictionary for hybrid mode")
            return np.random.rand(parameters.height, parameters.width, 3)

        # Extract components
        text = input_data.get('text')
        arabic_text = input_data.get('arabic_text')
        concept = input_data.get('concept')
        equation = input_data.get('equation')

        # Generate base image
        if text:
            # Use text as base
            base_params = copy.deepcopy(parameters)
            base_params.mode = GenerationMode.TEXT_TO_IMAGE
            base_image = self._generate_from_text(text, base_params)
        elif arabic_text:
            # Use Arabic text as base
            base_params = copy.deepcopy(parameters)
            base_params.mode = GenerationMode.ARABIC_TEXT_TO_IMAGE
            base_image = self._generate_from_arabic_text(arabic_text, base_params)
        elif concept is not None:
            # Use concept as base
            base_params = copy.deepcopy(parameters)
            base_params.mode = GenerationMode.CONCEPT_TO_IMAGE
            base_image = self._generate_from_concept(concept, base_params)
        elif equation:
            # Use equation as base
            base_params = copy.deepcopy(parameters)
            base_params.mode = GenerationMode.EQUATION_TO_IMAGE
            base_image = self._generate_from_equation(equation, base_params)
        else:
            # No valid base, use random image
            base_image = np.random.rand(parameters.height, parameters.width, 3)

        return base_image

    def save_image(self, result: GenerationResult, output_path: str) -> None:
        """
        Save generated image to file.

        Args:
            result: Generation result
            output_path: Path to save the image
        """
        if PIL_AVAILABLE:
            # Convert to PIL Image if necessary
            if isinstance(result.image, np.ndarray):
                # Ensure values are in [0, 255]
                image_array = (result.image * 255).astype(np.uint8)
                image = Image.fromarray(image_array)
            else:
                image = result.image

            # Save image
            image.save(output_path)
            self.logger.info(f"Image saved to {output_path}")
        else:
            self.logger.error("PIL not available, cannot save image")
            raise ImportError("PIL not available, cannot save image")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create Image Generator
    generator = ImageGenerator()

    # Generate image from text
    parameters = GenerationParameters(
        mode=GenerationMode.TEXT_TO_IMAGE,
        width=512,
        height=512,
        seed=42
    )
    result = generator.generate_image("A beautiful landscape with mountains and a lake", parameters)

    # Save image
    if PIL_AVAILABLE:
        generator.save_image(result, "landscape.png")

    print(f"Image generated in {result.generation_time:.2f} seconds")
