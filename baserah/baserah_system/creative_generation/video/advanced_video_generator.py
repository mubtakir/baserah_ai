#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Video Generator for Baserah System

This module implements an advanced video generator that creates videos from text
descriptions, images, or concepts using the General Shape Equation.

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import from core module
try:
    from core.general_shape_equation import (
        GeneralShapeEquation,
        EquationType,
        LearningMode,
        SymbolicExpression
    )
    from core.creative_generation import (
        CreativeGenerationComponent,
        CreativeMode,
        GenerationParameters,
        GenerationResult
    )
except ImportError:
    logging.error("Failed to import from core modules")
    sys.exit(1)

# Try to import PIL for image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    logging.warning("PIL not available, image generation will be limited")
    PIL_AVAILABLE = False

# Try to import OpenCV for video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logging.warning("OpenCV not available, video processing will be limited")
    CV2_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available, some functionality will be limited")
    TORCH_AVAILABLE = False

# Configure logging
logger = logging.getLogger('creative_generation.video.advanced_video_generator')


class VideoGenerationMode(str, Enum):
    """Video generation modes."""
    TEXT_TO_VIDEO = "text_to_video"  # Generate video from text description
    IMAGE_TO_VIDEO = "image_to_video"  # Generate video from image(s)
    CONCEPT_TO_VIDEO = "concept_to_video"  # Generate video from semantic concept
    EQUATION_TO_VIDEO = "equation_to_video"  # Generate video from mathematical equation
    ANIMATION = "animation"  # Generate animation
    CALLIGRAPHY_ANIMATION = "calligraphy_animation"  # Generate animated Arabic calligraphy
    HYBRID = "hybrid"  # Combine multiple generation modes


@dataclass
class VideoGenerationParameters:
    """Parameters for video generation."""
    mode: VideoGenerationMode
    prompt: str
    width: int = 512
    height: int = 512
    duration: float = 5.0  # Duration in seconds
    fps: int = 30  # Frames per second
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    input_images: Optional[List[Any]] = None  # List of input images (for IMAGE_TO_VIDEO mode)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerationResult:
    """Result of video generation."""
    result_id: str
    mode: VideoGenerationMode
    frames: List[Any]  # List of frames (PIL Images or numpy arrays)
    parameters: VideoGenerationParameters
    generation_time: float
    fps: int
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedVideoGenerator:
    """
    Advanced Video Generator for Baserah System.
    
    This class implements an advanced video generator that creates videos from text
    descriptions, images, or concepts using the General Shape Equation.
    """
    
    def __init__(self):
        """Initialize the advanced video generator."""
        # Initialize General Shape Equation
        self.equation = GeneralShapeEquation(
            equation_type=EquationType.CREATIVE,
            learning_mode=LearningMode.HYBRID
        )
        
        # Initialize creative generation component
        self.creative = CreativeGenerationComponent(self.equation)
        
        # Initialize equation components
        self._initialize_equation_components()
        
        # Initialize video models if PyTorch is available
        if TORCH_AVAILABLE:
            self._initialize_video_models()
        
        # Store generation history
        self.generation_history = {}
    
    def _initialize_equation_components(self) -> None:
        """Initialize the components of the General Shape Equation."""
        # Add components for video generation
        self.equation.add_component("frame_generation", "generate_frame(prompt, frame_index, total_frames)")
        self.equation.add_component("motion_prediction", "predict_motion(current_frame, next_frame)")
        self.equation.add_component("temporal_coherence", "ensure_coherence(frames)")
        self.equation.add_component("video_quality", "evaluate_quality(frames)")
        
        # Add components for animation
        self.equation.add_component("animation_curve", "interpolate(start_value, end_value, t)")
        self.equation.add_component("keyframe_generation", "generate_keyframes(prompt, num_keyframes)")
        self.equation.add_component("inbetweening", "generate_inbetweens(keyframe1, keyframe2, num_frames)")
        
        # Add components for calligraphy animation
        self.equation.add_component("calligraphy_animation", "animate_calligraphy(text, duration, style)")
        self.equation.add_component("stroke_animation", "animate_stroke(stroke, duration)")
    
    def _initialize_video_models(self) -> None:
        """Initialize video models if PyTorch is available."""
        # Text-to-video diffusion model (placeholder)
        class TextToVideoModel(nn.Module):
            def __init__(self, latent_dim=512, frame_dim=512):
                super().__init__()
                self.text_encoder = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024)
                )
                self.frame_decoder = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, frame_dim)
                )
                self.temporal_model = nn.GRU(1024, 1024, batch_first=True)
            
            def forward(self, text_embedding, num_frames):
                # Encode text
                text_features = self.text_encoder(text_embedding)
                
                # Initialize hidden state
                hidden = torch.zeros(1, 1, 1024)
                
                # Generate frames
                frames = []
                for i in range(num_frames):
                    # Update hidden state
                    _, hidden = self.temporal_model(text_features.unsqueeze(0), hidden)
                    
                    # Decode frame
                    frame = self.frame_decoder(hidden.squeeze(0))
                    frames.append(frame)
                
                return torch.stack(frames)
        
        # Motion prediction model (placeholder)
        class MotionPredictionModel(nn.Module):
            def __init__(self, frame_dim=512):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(frame_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512)
                )
                self.motion_predictor = nn.GRU(512, 512, batch_first=True)
                self.decoder = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, frame_dim)
                )
            
            def forward(self, frames, num_future_frames):
                # Encode frames
                encoded_frames = self.encoder(frames)
                
                # Initialize hidden state with last encoded frame
                hidden = encoded_frames[-1:].unsqueeze(0)
                
                # Predict future frames
                future_frames = []
                current = encoded_frames[-1:].unsqueeze(0)
                
                for i in range(num_future_frames):
                    # Update hidden state
                    output, hidden = self.motion_predictor(current, hidden)
                    
                    # Decode frame
                    future_frame = self.decoder(output.squeeze(0))
                    future_frames.append(future_frame)
                    
                    # Update current
                    current = output
                
                return torch.stack(future_frames)
        
        # Initialize models (placeholders)
        self.video_models = {
            "text_to_video": TextToVideoModel(),
            "motion_prediction": MotionPredictionModel()
        }
    
    def generate_video(self, parameters: VideoGenerationParameters) -> VideoGenerationResult:
        """
        Generate a video based on parameters.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            Video generation result
        """
        start_time = time.time()
        
        # Set seed if provided
        if parameters.seed is not None:
            np.random.seed(parameters.seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(parameters.seed)
        
        # Generate video based on mode
        if parameters.mode == VideoGenerationMode.TEXT_TO_VIDEO:
            frames = self._generate_video_from_text(parameters)
        elif parameters.mode == VideoGenerationMode.IMAGE_TO_VIDEO:
            frames = self._generate_video_from_images(parameters)
        elif parameters.mode == VideoGenerationMode.CONCEPT_TO_VIDEO:
            frames = self._generate_video_from_concept(parameters)
        elif parameters.mode == VideoGenerationMode.EQUATION_TO_VIDEO:
            frames = self._generate_video_from_equation(parameters)
        elif parameters.mode == VideoGenerationMode.ANIMATION:
            frames = self._generate_animation(parameters)
        elif parameters.mode == VideoGenerationMode.CALLIGRAPHY_ANIMATION:
            frames = self._generate_calligraphy_animation(parameters)
        elif parameters.mode == VideoGenerationMode.HYBRID:
            frames = self._generate_hybrid_video(parameters)
        else:
            raise ValueError(f"Unsupported video generation mode: {parameters.mode}")
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Create result
        result_id = f"video_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        result = VideoGenerationResult(
            result_id=result_id,
            mode=parameters.mode,
            frames=frames,
            parameters=parameters,
            generation_time=generation_time,
            fps=parameters.fps,
            duration=parameters.duration,
            metadata={
                "num_frames": len(frames),
                "width": parameters.width,
                "height": parameters.height,
                "format": "list[PIL.Image]" if PIL_AVAILABLE else "list[numpy.ndarray]"
            }
        )
        
        # Add to history
        self.generation_history[result_id] = result
        
        return result
    
    def _generate_video_from_text(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate a video from text description.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Calculate number of frames
        num_frames = int(parameters.duration * parameters.fps)
        
        # Generate frames
        frames = []
        
        for i in range(num_frames):
            # Create frame parameters for image generation
            frame_params = GenerationParameters(
                mode=CreativeMode.IMAGE,
                prompt=parameters.prompt,
                width=parameters.width,
                height=parameters.height,
                seed=parameters.seed,
                guidance_scale=parameters.guidance_scale,
                num_inference_steps=parameters.num_inference_steps,
                custom_params={
                    "frame_index": i,
                    "total_frames": num_frames,
                    **parameters.custom_params
                }
            )
            
            # Generate frame
            frame_result = self.creative.generate_image(frame_params)
            
            # Add frame to list
            frames.append(frame_result.content)
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Generated frame {i+1}/{num_frames}")
        
        return frames
    
    def _generate_video_from_images(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate a video from input images.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Check if input images are provided
        if not parameters.input_images:
            raise ValueError("Input images must be provided for IMAGE_TO_VIDEO mode")
        
        # Calculate number of frames
        num_frames = int(parameters.duration * parameters.fps)
        
        # If only one image is provided, create a simple animation
        if len(parameters.input_images) == 1:
            return self._animate_single_image(parameters.input_images[0], num_frames, parameters)
        
        # If multiple images are provided, interpolate between them
        return self._interpolate_images(parameters.input_images, num_frames, parameters)
    
    def _animate_single_image(self, image: Any, num_frames: int, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Create a simple animation from a single image.
        
        Args:
            image: Input image
            num_frames: Number of frames to generate
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Convert image to numpy array if it's a PIL Image
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Create animation (simple zoom and pan)
        frames = []
        
        for i in range(num_frames):
            # Calculate animation progress (0 to 1)
            t = i / (num_frames - 1) if num_frames > 1 else 0
            
            # Create a copy of the image
            frame = image_array.copy()
            
            # Apply zoom effect
            zoom_factor = 1.0 + 0.2 * np.sin(t * 2 * np.pi)
            
            # Apply pan effect
            pan_x = int(20 * np.sin(t * 2 * np.pi))
            pan_y = int(20 * np.cos(t * 2 * np.pi))
            
            # Apply effects (placeholder)
            # In a real implementation, this would use proper image transformation
            
            # Convert back to PIL Image if needed
            if PIL_AVAILABLE:
                frame = Image.fromarray(frame)
            
            frames.append(frame)
        
        return frames
    
    def _interpolate_images(self, images: List[Any], num_frames: int, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Interpolate between multiple images to create a video.
        
        Args:
            images: List of input images
            num_frames: Number of frames to generate
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Convert images to numpy arrays if they're PIL Images
        image_arrays = []
        for image in images:
            if PIL_AVAILABLE and isinstance(image, Image.Image):
                image_arrays.append(np.array(image))
            else:
                image_arrays.append(image)
        
        # Calculate frames per transition
        num_transitions = len(image_arrays) - 1
        frames_per_transition = num_frames // num_transitions
        
        # Generate frames
        frames = []
        
        for i in range(num_transitions):
            # Get source and target images
            source = image_arrays[i]
            target = image_arrays[i + 1]
            
            # Generate transition frames
            for j in range(frames_per_transition):
                # Calculate interpolation factor (0 to 1)
                t = j / frames_per_transition
                
                # Linear interpolation between source and target
                frame = source * (1 - t) + target * t
                
                # Convert to uint8
                frame = frame.astype(np.uint8)
                
                # Convert back to PIL Image if needed
                if PIL_AVAILABLE:
                    frame = Image.fromarray(frame)
                
                frames.append(frame)
        
        # Add the last image if needed
        if len(frames) < num_frames:
            last_image = image_arrays[-1]
            
            # Convert back to PIL Image if needed
            if PIL_AVAILABLE:
                last_image = Image.fromarray(last_image)
            
            frames.extend([last_image] * (num_frames - len(frames)))
        
        return frames
    
    def _generate_video_from_concept(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate a video from a semantic concept.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # For now, treat concept as text and generate video
        return self._generate_video_from_text(parameters)
    
    def _generate_video_from_equation(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate a video from a mathematical equation.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Calculate number of frames
        num_frames = int(parameters.duration * parameters.fps)
        
        # Generate frames
        frames = []
        
        for i in range(num_frames):
            # Calculate animation progress (0 to 1)
            t = i / (num_frames - 1) if num_frames > 1 else 0
            
            # Create frame parameters for image generation
            frame_params = GenerationParameters(
                mode=CreativeMode.IMAGE,
                prompt=parameters.prompt,
                width=parameters.width,
                height=parameters.height,
                seed=parameters.seed,
                guidance_scale=parameters.guidance_scale,
                num_inference_steps=parameters.num_inference_steps,
                custom_params={
                    "frame_index": i,
                    "total_frames": num_frames,
                    "t": t,
                    "equation": parameters.prompt,
                    **parameters.custom_params
                }
            )
            
            # Generate frame
            frame_result = self.creative.generate_image(frame_params)
            
            # Add frame to list
            frames.append(frame_result.content)
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Generated frame {i+1}/{num_frames}")
        
        return frames
    
    def _generate_animation(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate an animation.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Calculate number of frames
        num_frames = int(parameters.duration * parameters.fps)
        
        # Generate keyframes
        num_keyframes = parameters.custom_params.get("num_keyframes", 5)
        keyframes = self._generate_keyframes(parameters.prompt, num_keyframes, parameters)
        
        # Interpolate between keyframes
        frames = self._interpolate_keyframes(keyframes, num_frames, parameters)
        
        return frames
    
    def _generate_keyframes(self, prompt: str, num_keyframes: int, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate keyframes for animation.
        
        Args:
            prompt: Text prompt
            num_keyframes: Number of keyframes to generate
            parameters: Video generation parameters
            
        Returns:
            List of keyframes
        """
        keyframes = []
        
        for i in range(num_keyframes):
            # Create frame parameters for image generation
            frame_params = GenerationParameters(
                mode=CreativeMode.IMAGE,
                prompt=f"{prompt} (keyframe {i+1} of {num_keyframes})",
                width=parameters.width,
                height=parameters.height,
                seed=parameters.seed + i if parameters.seed is not None else None,
                guidance_scale=parameters.guidance_scale,
                num_inference_steps=parameters.num_inference_steps,
                custom_params={
                    "keyframe_index": i,
                    "total_keyframes": num_keyframes,
                    **parameters.custom_params
                }
            )
            
            # Generate keyframe
            keyframe_result = self.creative.generate_image(frame_params)
            
            # Add keyframe to list
            keyframes.append(keyframe_result.content)
            
            # Log progress
            logger.info(f"Generated keyframe {i+1}/{num_keyframes}")
        
        return keyframes
    
    def _interpolate_keyframes(self, keyframes: List[Any], num_frames: int, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Interpolate between keyframes to create animation frames.
        
        Args:
            keyframes: List of keyframes
            num_frames: Total number of frames to generate
            parameters: Video generation parameters
            
        Returns:
            List of animation frames
        """
        # Similar to _interpolate_images, but with more sophisticated interpolation
        return self._interpolate_images(keyframes, num_frames, parameters)
    
    def _generate_calligraphy_animation(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate animated Arabic calligraphy.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Calculate number of frames
        num_frames = int(parameters.duration * parameters.fps)
        
        # Generate frames
        frames = []
        
        for i in range(num_frames):
            # Calculate animation progress (0 to 1)
            t = i / (num_frames - 1) if num_frames > 1 else 0
            
            # Create frame parameters for calligraphy generation
            frame_params = GenerationParameters(
                mode=CreativeMode.CALLIGRAPHY,
                prompt=parameters.prompt,
                width=parameters.width,
                height=parameters.height,
                seed=parameters.seed,
                guidance_scale=parameters.guidance_scale,
                num_inference_steps=parameters.num_inference_steps,
                custom_params={
                    "frame_index": i,
                    "total_frames": num_frames,
                    "t": t,
                    **parameters.custom_params
                }
            )
            
            # Generate frame
            frame_result = self.creative.generate_calligraphy(frame_params)
            
            # Add frame to list
            frames.append(frame_result.content)
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Generated frame {i+1}/{num_frames}")
        
        return frames
    
    def _generate_hybrid_video(self, parameters: VideoGenerationParameters) -> List[Any]:
        """
        Generate a hybrid video combining multiple generation modes.
        
        Args:
            parameters: Video generation parameters
            
        Returns:
            List of video frames
        """
        # Get hybrid modes from custom parameters
        modes = parameters.custom_params.get("modes", [VideoGenerationMode.TEXT_TO_VIDEO])
        
        # Calculate number of frames
        num_frames = int(parameters.duration * parameters.fps)
        
        # Calculate frames per mode
        frames_per_mode = num_frames // len(modes)
        
        # Generate frames for each mode
        all_frames = []
        
        for i, mode in enumerate(modes):
            # Create parameters for this mode
            mode_params = copy.deepcopy(parameters)
            mode_params.mode = mode
            mode_params.duration = parameters.duration * (frames_per_mode / num_frames)
            
            # Generate frames for this mode
            if mode == VideoGenerationMode.TEXT_TO_VIDEO:
                frames = self._generate_video_from_text(mode_params)
            elif mode == VideoGenerationMode.IMAGE_TO_VIDEO:
                frames = self._generate_video_from_images(mode_params)
            elif mode == VideoGenerationMode.CONCEPT_TO_VIDEO:
                frames = self._generate_video_from_concept(mode_params)
            elif mode == VideoGenerationMode.EQUATION_TO_VIDEO:
                frames = self._generate_video_from_equation(mode_params)
            elif mode == VideoGenerationMode.ANIMATION:
                frames = self._generate_animation(mode_params)
            elif mode == VideoGenerationMode.CALLIGRAPHY_ANIMATION:
                frames = self._generate_calligraphy_animation(mode_params)
            else:
                # Default to text-to-video
                frames = self._generate_video_from_text(mode_params)
            
            # Add frames to list
            all_frames.extend(frames[:frames_per_mode])
        
        # Add remaining frames if needed
        if len(all_frames) < num_frames:
            # Duplicate the last frame
            last_frame = all_frames[-1]
            all_frames.extend([last_frame] * (num_frames - len(all_frames)))
        
        return all_frames
    
    def save_video(self, result_id: str, file_path: str) -> bool:
        """
        Save a generated video to a file.
        
        Args:
            result_id: ID of the video generation result
            file_path: Path to save the video to
            
        Returns:
            True if successful, False otherwise
        """
        # Get result
        result = self.generation_history.get(result_id)
        if not result:
            logger.error(f"Video generation result with ID {result_id} not found")
            return False
        
        # Check if OpenCV is available
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available, cannot save video")
            return False
        
        try:
            # Get video properties
            height, width = result.parameters.height, result.parameters.width
            fps = result.fps
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in result.frames:
                # Convert frame to OpenCV format
                if PIL_AVAILABLE and isinstance(frame, Image.Image):
                    # Convert PIL Image to numpy array
                    frame_array = np.array(frame)
                    # Convert RGB to BGR
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                else:
                    # Assume frame is already a numpy array
                    frame_array = frame
                    # Convert RGB to BGR if needed
                    if frame_array.shape[2] == 3:
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(frame_array)
            
            # Release video writer
            out.release()
            
            logger.info(f"Video saved to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return False
    
    def get_result(self, result_id: str) -> Optional[VideoGenerationResult]:
        """
        Get a video generation result by ID.
        
        Args:
            result_id: ID of the video generation result
            
        Returns:
            Video generation result or None if not found
        """
        return self.generation_history.get(result_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the video generator to a dictionary.
        
        Returns:
            Dictionary representation of the video generator
        """
        # Note: We don't include the actual frames in the dictionary
        # to avoid large data structures
        return {
            "generation_history": {
                result_id: {
                    "result_id": result.result_id,
                    "mode": result.mode.value,
                    "parameters": {
                        "mode": result.parameters.mode.value,
                        "prompt": result.parameters.prompt,
                        "width": result.parameters.width,
                        "height": result.parameters.height,
                        "duration": result.parameters.duration,
                        "fps": result.parameters.fps,
                        "seed": result.parameters.seed,
                        "guidance_scale": result.parameters.guidance_scale,
                        "num_inference_steps": result.parameters.num_inference_steps,
                        "custom_params": result.parameters.custom_params
                    },
                    "generation_time": result.generation_time,
                    "fps": result.fps,
                    "duration": result.duration,
                    "metadata": result.metadata,
                    "num_frames": len(result.frames)
                }
                for result_id, result in self.generation_history.items()
            }
        }
    
    def to_json(self) -> str:
        """
        Convert the video generator to a JSON string.
        
        Returns:
            JSON string representation of the video generator
        """
        return json.dumps(self.to_dict(), indent=2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create video generator
    generator = AdvancedVideoGenerator()
    
    # Generate video
    parameters = VideoGenerationParameters(
        mode=VideoGenerationMode.TEXT_TO_VIDEO,
        prompt="A beautiful sunset over the ocean",
        width=256,
        height=256,
        duration=3.0,
        fps=10,
        seed=42
    )
    
    result = generator.generate_video(parameters)
    
    print("Video Generation Result:")
    print(f"ID: {result.result_id}")
    print(f"Mode: {result.mode}")
    print(f"Number of frames: {len(result.frames)}")
    print(f"Duration: {result.duration} seconds")
    print(f"FPS: {result.fps}")
    print(f"Generation time: {result.generation_time:.2f} seconds")
    
    # Save video
    if CV2_AVAILABLE:
        generator.save_video(result.result_id, "output_video.mp4")
        print("Video saved to output_video.mp4")
