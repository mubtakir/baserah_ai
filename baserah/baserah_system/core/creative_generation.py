#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creative Generation Component for General Shape Equation

This module implements the creative generation component for the General Shape Equation,
which adds support for generating creative content like images, text, and code.

Author: Baserah System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import uuid

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import from core module
try:
    from core.general_shape_equation import (
        GeneralShapeEquation,
        EquationType,
        LearningMode,
        SymbolicExpression
    )
except ImportError:
    logging.error("Failed to import from core.general_shape_equation")
    sys.exit(1)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available, some functionality will be limited")
    TORCH_AVAILABLE = False

# Try to import PIL for image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    logging.warning("PIL not available, image generation will be limited")
    PIL_AVAILABLE = False

# Configure logging
logger = logging.getLogger('core.creative_generation')


class CreativeMode(str, Enum):
    """Creative generation modes."""
    TEXT = "text"                  # Text generation
    IMAGE = "image"                # Image generation
    CODE = "code"                  # Code generation
    MUSIC = "music"                # Music generation
    VIDEO = "video"                # Video generation
    STORY = "story"                # Story generation
    POEM = "poem"                  # Poem generation
    CALLIGRAPHY = "calligraphy"    # Arabic calligraphy generation
    HYBRID = "hybrid"              # Hybrid generation (multiple modes)


@dataclass
class GenerationParameters:
    """Parameters for creative generation."""
    mode: CreativeMode
    prompt: str
    temperature: float = 0.7
    max_length: int = 1000
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of creative generation."""
    result_id: str
    mode: CreativeMode
    content: Any
    parameters: GenerationParameters
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CreativeGenerationComponent:
    """
    Creative Generation Component for the General Shape Equation.
    
    This class extends the General Shape Equation with creative generation capabilities,
    including generating images, text, code, and other creative content.
    """
    
    def __init__(self, equation: GeneralShapeEquation):
        """
        Initialize the creative generation component.
        
        Args:
            equation: The General Shape Equation to extend
        """
        self.equation = equation
        self.generation_history = {}
        self.models = {}
        
        # Add creative generation components to the equation
        self._initialize_creative_components()
        
        # Initialize creative models if PyTorch is available
        if TORCH_AVAILABLE:
            self._initialize_creative_models()
    
    def _initialize_creative_components(self) -> None:
        """Initialize the creative generation components in the equation."""
        # Add basic creative components
        self.equation.add_component("creativity_score", "novelty * surprise * value")
        self.equation.add_component("novelty", "1.0 - similarity_to_known_examples")
        self.equation.add_component("surprise", "1.0 - predictability")
        self.equation.add_component("value", "usefulness * aesthetic_quality")
        
        # Add generation components
        self.equation.add_component("text_generation", "generate_text(prompt, parameters)")
        self.equation.add_component("image_generation", "generate_image(prompt, parameters)")
        self.equation.add_component("code_generation", "generate_code(prompt, parameters)")
        self.equation.add_component("calligraphy_generation", "generate_calligraphy(text, parameters)")
        
        # Add style components
        self.equation.add_component("style_transfer", "apply_style(content, style, parameters)")
        self.equation.add_component("style_extraction", "extract_style(content)")
    
    def _initialize_creative_models(self) -> None:
        """Initialize creative models if PyTorch is available."""
        # Text generation model (placeholder)
        class TextGenerator(nn.Module):
            def __init__(self, vocab_size=10000, embedding_dim=512, hidden_dim=1024):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2)
                self.fc = nn.Linear(hidden_dim, vocab_size)
            
            def forward(self, x):
                embedded = self.embedding(x)
                output, (hidden, cell) = self.lstm(embedded)
                return self.fc(output)
        
        # Image generation model (placeholder)
        class ImageGenerator(nn.Module):
            def __init__(self, latent_dim=100, channels=3, height=64, width=64):
                super().__init__()
                self.height = height
                self.width = width
                self.channels = channels
                
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 128 * (height // 8) * (width // 8)),
                    nn.ReLU()
                )
                
                self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()
                )
            
            def forward(self, z):
                x = self.fc(z)
                x = x.view(-1, 128, self.height // 8, self.width // 8)
                return self.deconv(x)
        
        # Code generation model (placeholder)
        class CodeGenerator(nn.Module):
            def __init__(self, vocab_size=10000, embedding_dim=512, hidden_dim=1024):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2)
                self.fc = nn.Linear(hidden_dim, vocab_size)
            
            def forward(self, x):
                embedded = self.embedding(x)
                output, (hidden, cell) = self.lstm(embedded)
                return self.fc(output)
        
        # Initialize models
        self.models["text"] = TextGenerator()
        self.models["image"] = ImageGenerator()
        self.models["code"] = CodeGenerator()
    
    def generate_text(self, parameters: GenerationParameters) -> GenerationResult:
        """
        Generate text based on parameters.
        
        Args:
            parameters: Generation parameters
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        # Ensure mode is correct
        if parameters.mode != CreativeMode.TEXT:
            parameters.mode = CreativeMode.TEXT
        
        # Generate text (placeholder implementation)
        # In a real implementation, this would use a more sophisticated model
        
        # Simple template-based generation for demonstration
        templates = [
            "{prompt} هو موضوع مهم في العصر الحديث. يمكن النظر إليه من عدة جوانب.",
            "عندما نتحدث عن {prompt}، علينا أن نفكر في تأثيره على المجتمع والفرد.",
            "{prompt} يمثل تحدياً كبيراً في عالمنا المعاصر، ويتطلب منا دراسة متأنية.",
            "من المثير للاهتمام أن نلاحظ كيف يؤثر {prompt} على حياتنا اليومية.",
            "يعتبر {prompt} من الموضوعات التي شغلت الفلاسفة والمفكرين عبر العصور."
        ]
        
        import random
        template = random.choice(templates)
        
        # Generate base text
        text = template.format(prompt=parameters.prompt)
        
        # Add some paragraphs based on temperature
        num_paragraphs = max(1, int(parameters.temperature * 5))
        
        for _ in range(num_paragraphs):
            text += "\n\n"
            text += random.choice(templates).format(prompt=parameters.prompt)
        
        # Limit to max length
        if len(text) > parameters.max_length:
            text = text[:parameters.max_length]
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Create result
        result_id = f"text_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        result = GenerationResult(
            result_id=result_id,
            mode=CreativeMode.TEXT,
            content=text,
            parameters=parameters,
            generation_time=generation_time,
            metadata={
                "length": len(text),
                "paragraphs": num_paragraphs + 1
            }
        )
        
        # Add to history
        self.generation_history[result_id] = result
        
        return result
    
    def generate_image(self, parameters: GenerationParameters) -> GenerationResult:
        """
        Generate an image based on parameters.
        
        Args:
            parameters: Generation parameters
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        # Ensure mode is correct
        if parameters.mode != CreativeMode.IMAGE:
            parameters.mode = CreativeMode.IMAGE
        
        # Set seed if provided
        if parameters.seed is not None:
            np.random.seed(parameters.seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(parameters.seed)
        
        # Generate image (placeholder implementation)
        # In a real implementation, this would use a more sophisticated model
        
        if PIL_AVAILABLE:
            # Create a simple image with the prompt text
            image = Image.new('RGB', (parameters.width, parameters.height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw prompt text
            text_width, text_height = draw.textsize(parameters.prompt, font=font)
            position = ((parameters.width - text_width) // 2, (parameters.height - text_height) // 2)
            draw.text(position, parameters.prompt, fill='black', font=font)
            
            # Draw some random shapes based on the prompt
            for _ in range(10):
                shape_type = hash(parameters.prompt + str(_)) % 3
                
                if shape_type == 0:  # Circle
                    x = np.random.randint(0, parameters.width)
                    y = np.random.randint(0, parameters.height)
                    radius = np.random.randint(10, 50)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)
                
                elif shape_type == 1:  # Rectangle
                    x1 = np.random.randint(0, parameters.width)
                    y1 = np.random.randint(0, parameters.height)
                    x2 = x1 + np.random.randint(20, 100)
                    y2 = y1 + np.random.randint(20, 100)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    draw.rectangle((x1, y1, x2, y2), outline=color)
                
                else:  # Line
                    x1 = np.random.randint(0, parameters.width)
                    y1 = np.random.randint(0, parameters.height)
                    x2 = np.random.randint(0, parameters.width)
                    y2 = np.random.randint(0, parameters.height)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    draw.line((x1, y1, x2, y2), fill=color, width=2)
            
            # Convert to numpy array
            image_array = np.array(image)
        else:
            # Generate a random image
            image_array = np.random.randint(0, 255, (parameters.height, parameters.width, 3), dtype=np.uint8)
            image = None
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Create result
        result_id = f"image_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        result = GenerationResult(
            result_id=result_id,
            mode=CreativeMode.IMAGE,
            content=image if PIL_AVAILABLE else image_array,
            parameters=parameters,
            generation_time=generation_time,
            metadata={
                "width": parameters.width,
                "height": parameters.height,
                "format": "PIL.Image" if PIL_AVAILABLE else "numpy.ndarray"
            }
        )
        
        # Add to history
        self.generation_history[result_id] = result
        
        return result
    
    def generate_code(self, parameters: GenerationParameters) -> GenerationResult:
        """
        Generate code based on parameters.
        
        Args:
            parameters: Generation parameters
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        # Ensure mode is correct
        if parameters.mode != CreativeMode.CODE:
            parameters.mode = CreativeMode.CODE
        
        # Generate code (placeholder implementation)
        # In a real implementation, this would use a more sophisticated model
        
        # Simple template-based generation for demonstration
        language = parameters.custom_params.get("language", "python")
        
        if language == "python":
            code = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{parameters.prompt}

Generated by Baserah System
\"\"\"

import os
import sys
import numpy as np
import time


def main():
    \"\"\"Main function.\"\"\"
    print("Implementing: {parameters.prompt}")
    
    # TODO: Implement {parameters.prompt}
    
    result = process_data()
    print(f"Result: {result}")
    
    return 0


def process_data():
    \"\"\"Process data for {parameters.prompt}.\"\"\"
    # Placeholder implementation
    data = np.random.rand(10)
    return np.mean(data)


if __name__ == "__main__":
    sys.exit(main())
"""
        
        elif language == "javascript":
            code = f"""/**
 * {parameters.prompt}
 * 
 * Generated by Baserah System
 */

// Main function
function main() {{
    console.log("Implementing: {parameters.prompt}");
    
    // TODO: Implement {parameters.prompt}
    
    const result = processData();
    console.log(`Result: ${{result}}`);
    
    return 0;
}}

/**
 * Process data for {parameters.prompt}
 */
function processData() {{
    // Placeholder implementation
    const data = Array.from({{length: 10}}, () => Math.random());
    return data.reduce((sum, value) => sum + value, 0) / data.length;
}}

// Run the main function
main();
"""
        
        else:
            code = f"// Code generation for {language} not implemented yet\n// Prompt: {parameters.prompt}"
        
        # Limit to max length
        if len(code) > parameters.max_length:
            code = code[:parameters.max_length]
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Create result
        result_id = f"code_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        result = GenerationResult(
            result_id=result_id,
            mode=CreativeMode.CODE,
            content=code,
            parameters=parameters,
            generation_time=generation_time,
            metadata={
                "language": language,
                "length": len(code),
                "lines": code.count("\n") + 1
            }
        )
        
        # Add to history
        self.generation_history[result_id] = result
        
        return result
    
    def generate_calligraphy(self, parameters: GenerationParameters) -> GenerationResult:
        """
        Generate Arabic calligraphy based on parameters.
        
        Args:
            parameters: Generation parameters
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        # Ensure mode is correct
        if parameters.mode != CreativeMode.CALLIGRAPHY:
            parameters.mode = CreativeMode.CALLIGRAPHY
        
        # Set seed if provided
        if parameters.seed is not None:
            np.random.seed(parameters.seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(parameters.seed)
        
        # Generate calligraphy (placeholder implementation)
        # In a real implementation, this would use a more sophisticated model
        
        if PIL_AVAILABLE:
            # Create a simple calligraphy image
            image = Image.new('RGB', (parameters.width, parameters.height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
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
                        font = ImageFont.truetype(path, 60)
                        break
                
                if font is None:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            # Draw text in the center
            text_width, text_height = draw.textsize(parameters.prompt, font=font)
            position = ((parameters.width - text_width) // 2, (parameters.height - text_height) // 2)
            draw.text(position, parameters.prompt, fill='black', font=font)
            
            # Add decorative border
            border_width = 10
            draw.rectangle([(border_width, border_width), (parameters.width - border_width, parameters.height - border_width)], 
                          outline='black', width=2)
            
            # Convert to numpy array
            image_array = np.array(image)
        else:
            # Generate a random image
            image_array = np.random.randint(0, 255, (parameters.height, parameters.width, 3), dtype=np.uint8)
            image = None
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Create result
        result_id = f"calligraphy_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        result = GenerationResult(
            result_id=result_id,
            mode=CreativeMode.CALLIGRAPHY,
            content=image if PIL_AVAILABLE else image_array,
            parameters=parameters,
            generation_time=generation_time,
            metadata={
                "text": parameters.prompt,
                "width": parameters.width,
                "height": parameters.height,
                "format": "PIL.Image" if PIL_AVAILABLE else "numpy.ndarray"
            }
        )
        
        # Add to history
        self.generation_history[result_id] = result
        
        return result
    
    def generate_video(self, parameters: GenerationParameters) -> GenerationResult:
        """
        Generate a video based on parameters.
        
        Args:
            parameters: Generation parameters
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        # Ensure mode is correct
        if parameters.mode != CreativeMode.VIDEO:
            parameters.mode = CreativeMode.VIDEO
        
        # Generate video (placeholder implementation)
        # In a real implementation, this would use a more sophisticated model
        
        # For now, just generate a sequence of frames
        frames = []
        num_frames = parameters.custom_params.get("num_frames", 30)
        
        for i in range(num_frames):
            # Create frame parameters
            frame_params = copy.deepcopy(parameters)
            frame_params.mode = CreativeMode.IMAGE
            frame_params.custom_params["frame_index"] = i
            
            # Generate frame
            frame_result = self.generate_image(frame_params)
            
            # Add frame to sequence
            if PIL_AVAILABLE:
                frames.append(frame_result.content)
            else:
                frames.append(frame_result.content)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Create result
        result_id = f"video_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        result = GenerationResult(
            result_id=result_id,
            mode=CreativeMode.VIDEO,
            content=frames,
            parameters=parameters,
            generation_time=generation_time,
            metadata={
                "num_frames": num_frames,
                "width": parameters.width,
                "height": parameters.height,
                "format": "list[PIL.Image]" if PIL_AVAILABLE else "list[numpy.ndarray]"
            }
        )
        
        # Add to history
        self.generation_history[result_id] = result
        
        return result
    
    def get_result(self, result_id: str) -> Optional[GenerationResult]:
        """
        Get a generation result by ID.
        
        Args:
            result_id: ID of the result
            
        Returns:
            Generation result or None if not found
        """
        return self.generation_history.get(result_id)
    
    def save_result(self, result_id: str, file_path: str) -> bool:
        """
        Save a generation result to a file.
        
        Args:
            result_id: ID of the result
            file_path: Path to save the result to
            
        Returns:
            True if successful, False otherwise
        """
        result = self.get_result(result_id)
        
        if not result:
            logger.error(f"Result with ID {result_id} not found")
            return False
        
        try:
            if result.mode == CreativeMode.TEXT or result.mode == CreativeMode.CODE:
                # Save text or code to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result.content)
            
            elif result.mode == CreativeMode.IMAGE or result.mode == CreativeMode.CALLIGRAPHY:
                # Save image to file
                if PIL_AVAILABLE and isinstance(result.content, Image.Image):
                    result.content.save(file_path)
                elif isinstance(result.content, np.ndarray):
                    if PIL_AVAILABLE:
                        Image.fromarray(result.content).save(file_path)
                    else:
                        np.save(file_path, result.content)
                else:
                    logger.error(f"Unsupported image format: {type(result.content)}")
                    return False
            
            elif result.mode == CreativeMode.VIDEO:
                # Save video to file (placeholder)
                # In a real implementation, this would use a video encoder
                logger.warning("Video saving not fully implemented")
                
                # Save first frame as a preview
                if PIL_AVAILABLE and isinstance(result.content[0], Image.Image):
                    result.content[0].save(file_path)
                elif isinstance(result.content[0], np.ndarray):
                    if PIL_AVAILABLE:
                        Image.fromarray(result.content[0]).save(file_path)
                    else:
                        np.save(file_path, result.content[0])
                else:
                    logger.error(f"Unsupported video frame format: {type(result.content[0])}")
                    return False
            
            else:
                logger.error(f"Unsupported mode for saving: {result.mode}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the creative generation component to a dictionary.
        
        Returns:
            Dictionary representation of the creative generation component
        """
        # Note: We don't include the actual content in the dictionary
        # to avoid large data structures
        return {
            "generation_history": {
                result_id: {
                    "result_id": result.result_id,
                    "mode": result.mode.value,
                    "parameters": {
                        "mode": result.parameters.mode.value,
                        "prompt": result.parameters.prompt,
                        "temperature": result.parameters.temperature,
                        "max_length": result.parameters.max_length,
                        "width": result.parameters.width,
                        "height": result.parameters.height,
                        "seed": result.parameters.seed,
                        "guidance_scale": result.parameters.guidance_scale,
                        "num_inference_steps": result.parameters.num_inference_steps,
                        "custom_params": result.parameters.custom_params
                    },
                    "generation_time": result.generation_time,
                    "metadata": result.metadata,
                    "content_type": str(type(result.content))
                }
                for result_id, result in self.generation_history.items()
            }
        }
    
    def to_json(self) -> str:
        """
        Convert the creative generation component to a JSON string.
        
        Returns:
            JSON string representation of the creative generation component
        """
        return json.dumps(self.to_dict(), indent=2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a General Shape Equation
    equation = GeneralShapeEquation(
        equation_type=EquationType.CREATIVE,
        learning_mode=LearningMode.HYBRID
    )
    
    # Create a creative generation component
    creative = CreativeGenerationComponent(equation)
    
    # Generate text
    text_params = GenerationParameters(
        mode=CreativeMode.TEXT,
        prompt="الذكاء الاصطناعي والإبداع",
        temperature=0.8,
        max_length=500
    )
    
    text_result = creative.generate_text(text_params)
    
    print("Text Generation Result:")
    print(f"ID: {text_result.result_id}")
    print(f"Generation Time: {text_result.generation_time:.2f} seconds")
    print(f"Content (first 100 chars): {text_result.content[:100]}...")
    
    # Generate image
    image_params = GenerationParameters(
        mode=CreativeMode.IMAGE,
        prompt="منظر طبيعي جميل",
        width=256,
        height=256,
        seed=42
    )
    
    image_result = creative.generate_image(image_params)
    
    print("\nImage Generation Result:")
    print(f"ID: {image_result.result_id}")
    print(f"Generation Time: {image_result.generation_time:.2f} seconds")
    print(f"Content Type: {type(image_result.content)}")
    
    # Generate code
    code_params = GenerationParameters(
        mode=CreativeMode.CODE,
        prompt="برنامج لحساب الأعداد الأولية",
        temperature=0.7,
        max_length=500,
        custom_params={"language": "python"}
    )
    
    code_result = creative.generate_code(code_params)
    
    print("\nCode Generation Result:")
    print(f"ID: {code_result.result_id}")
    print(f"Generation Time: {code_result.generation_time:.2f} seconds")
    print(f"Content (first 100 chars): {code_result.content[:100]}...")
