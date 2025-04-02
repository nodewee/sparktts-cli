# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import torch
import soundfile as sf
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import platform
from datetime import datetime

# Import utility modules
from .utils import DEFAULT_CONFIG, get_default_value, select_device, get_timestamp_filename, ensure_dir_exists, setup_logging
from .SparkTTS import SparkTTS
from .voice import get_voice_tokens, load_voice
from sparktts.utils.text_processing import LongTextTTS, TextSegmenter

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_CONFIG.get("model_dir", "pretrained_models/Spark-TTS-0.5B"),
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_CONFIG.get("save_dir", "output"),
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=0, 
        help="CUDA device number"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="Text for TTS generation"
    )
    
    # Optional voice specification
    parser.add_argument(
        "--voice",
        type=str,
        help="Name of a saved voice to use (optional, uses default voice if not specified)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--gender", 
        choices=["male", "female"],
        help="Gender (only used when not using a voice profile)"
    )
    parser.add_argument(
        "--pitch", 
        choices=["very_low", "low", "moderate", "high", "very_high"],
        help="Pitch level"
    )
    parser.add_argument(
        "--speed", 
        choices=["very_low", "low", "moderate", "high", "very_high"],
        help="Speed level"
    )
    parser.add_argument(
        "--emotion", 
        type=str,
        help="Emotion (e.g., NEUTRAL, HAPPY, SAD, ANGRY)"
    )
    parser.add_argument(
        "--segmentation_threshold",
        type=int,
        default=DEFAULT_CONFIG.get("segmentation_threshold", 150),
        help="Maximum number of words per segment"
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage regardless of GPU availability"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="tts_cache",
        help="Directory for caching TTS audio segments"
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regeneration of audio even if cached"
    )
    return parser.parse_args()


def select_device(device=None, force_cpu=False):
    """
    Select appropriate device based on availability and platform
    
    Args:
        device: Device identifier (e.g., "cuda:0" or 0 for CUDA device index)
        force_cpu: Force CPU usage regardless of GPU availability
        
    Returns:
        torch.device: Selected device
    """
    # Check environment variables for forced CPU usage
    env_force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    
    # Use CPU if either parameter or environment variable forces it
    if force_cpu or env_force_cpu:
        device = torch.device("cpu")
        logging.info("Forcing CPU usage for inference")
        return device
        
    # On macOS, check for MPS (Metal Performance Shaders) availability
    if platform.system() == "Darwin":
        if torch.backends.mps.is_available():
            # Check if MPS fallback is enabled (which means we should use CPU)
            mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
            if mps_fallback:
                device = torch.device("cpu")
                logging.info("MPS fallback enabled - using CPU instead of MPS")
                return device
                
            print("Using MPS device for acceleration")
            device = torch.device("mps")
            return device
        else:
            device = torch.device("cpu")
            logging.info("MPS not available on macOS - falling back to CPU")
            return device
        
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Handle different device input types
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            device_str = DEFAULT_CONFIG.get("device", "cuda:0")
            device = torch.device(device_str)
        print(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        print("GPU acceleration not available, using CPU")
        
    return device


def generate_tts_audio(
    text: str,
    model_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    voice_name: Optional[str] = None,
    gender: Optional[str] = None,
    pitch: Optional[str] = None,
    speed: Optional[str] = None,
    emotion: Optional[str] = None,
    save_dir: Optional[str] = None,
    segmentation_threshold: int = 150,
    force_cpu: bool = False,
    cache_dir: str = "tts_cache",
    force_regenerate: bool = False,
) -> str:
    """
    Generate TTS audio with intelligent text segmentation and caching
    
    Args:
        text: Text to synthesize
        model_dir: Model directory
        device: Device to use for inference
        voice_name: Name of the voice profile to use (optional, uses default voice if None)
        gender: Gender (male or female) - only used without voice profile
        pitch: Pitch level (very_low, low, moderate, high, very_high)
        speed: Speed level (very_low, low, moderate, high, very_high)
        emotion: Emotion parameter (e.g., "HAPPY", "SAD", "NEUTRAL")
        save_dir: Output directory
        segmentation_threshold: Maximum number of words per segment
        force_cpu: Whether to force CPU usage
        cache_dir: Directory for caching TTS segments
        force_regenerate: Whether to force regeneration even if cached
    
    Returns:
        Path to the generated audio file
    """
    # Use configuration values with appropriate fallbacks
    model_dir = get_default_value(model_dir, "model_dir", "pretrained_models/Spark-TTS-0.5B")
    save_dir = get_default_value(save_dir, "save_dir", "output")
    
    # Initialize parameters
    global_tokens = None
    voice_info = None
    
    # If voice_name is provided, use voice profile
    if voice_name:
        logging.info(f"Using voice profile: {voice_name}")
        voice_info = load_voice(voice_name)
        if not voice_info:
            raise ValueError(f"Voice '{voice_name}' not found")
            
        # Extract global tokens from voice profile
        global_tokens, _ = get_voice_tokens(voice_name)
        if global_tokens is None:
            raise ValueError(f"Failed to get voice tokens for '{voice_name}'")
            
        # Use gender from voice profile if not overridden
        if gender is None and "gender" in voice_info:
            gender = voice_info["gender"]
    else:
        # When no voice profile is specified, ensure we have a default gender
        if gender is None:
            gender = "female"  # Default gender
        logging.info(f"Using default voice with gender: {gender}")
            
    # Setup model and device
    if device is None:
        device = select_device(device, force_cpu)
        
    # Initialize the TTS model
    logging.info(f"Initializing TTS model from {model_dir}")
    model = SparkTTS(model_dir, device)
    
    # Ensure output directory exists
    ensure_dir_exists(save_dir)
    
    # Generate output filename with timestamp
    output_path = os.path.join(save_dir, get_timestamp_filename())
    
    # Create TTS parameters dictionary for caching
    tts_params = {
        "emotion": emotion,
        "model_dir": model_dir,
        "device": str(device),
        "pitch": pitch,
        "speed": speed,
        "gender": gender
    }
    
    # Add voice tokens if available (from voice profile)
    if global_tokens is not None:
        tts_params["global_tokens"] = global_tokens
    
    # Filter out None values to maintain consistent cache keys
    tts_params = {k: v for k, v in tts_params.items() if v is not None}
    
    # Initialize long text TTS handler
    long_text_handler = LongTextTTS(model, cache_dir=cache_dir)
    
    # Generate speech with smart segmentation and caching
    logging.info(f"Generating speech for text ({len(text)} chars)")
    final_wav, fully_cached = long_text_handler.generate(
        text=text,
        tts_params=tts_params,
        segment_size=segmentation_threshold,
        force_regenerate=force_regenerate,
    )
    
    if fully_cached and not force_regenerate:
        logging.info("Using fully cached audio (all segments were cached)")
    
    # Save generated audio
    sf.write(output_path, final_wav, samplerate=16000)
    logging.info(f"Speech saved to: {output_path}")
    
    return output_path


def main():
    """Run inference for TTS generation."""
    # Parse arguments and setup logging
    args = parse_args()
    setup_logging()
    
    # Check all required inputs
    if not args.text:
        logging.error("No text provided")
        return 1
    
    device = select_device(args.device, args.force_cpu)
    
    # Generate audio
    try:
        output_file = generate_tts_audio(
            text=args.text,
            model_dir=args.model_dir,
            device=device,
            voice_name=args.voice if hasattr(args, 'voice') else None,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
            emotion=args.emotion,
            save_dir=args.save_dir,
            segmentation_threshold=args.segmentation_threshold,
            force_cpu=args.force_cpu,
            cache_dir=args.cache_dir,
            force_regenerate=args.force_regenerate,
        )
        
        print(f"Generated audio saved to: {output_file}")
        return 0
    except Exception as e:
        logging.error(f"Error during TTS generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
