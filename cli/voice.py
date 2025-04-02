"""
Voice management module for Spark-TTS.
Handles creating, listing, and deleting voice profiles.
"""

import os
import json
import logging
import torch
import soundfile as sf
from pathlib import Path
from datetime import datetime
import traceback

from .utils import (
    DEFAULT_CONFIG,
    get_default_value,
    select_device,
    ensure_dir_exists,
    get_audio_hash,
)
from .SparkTTS import SparkTTS
from .audio import (
    extract_voice_tokens,
    validate_reference_audio
)


def get_voices_dir():
    """Get the voice directory from config, ensuring it exists"""
    voices_dir = DEFAULT_CONFIG.get("voices_dir", "voices")
    ensure_dir_exists(voices_dir)
    return voices_dir


def get_voice_path(voice_name):
    """
    Get the path to a voice configuration file.
    
    Args:
        voice_name: Name of the voice
        
    Returns:
        Path to the voice configuration file or None if not found
    """
    voices_dir = get_voices_dir()
    voice_file = os.path.join(voices_dir, f"{voice_name}.json")
    if os.path.exists(voice_file):
        return voice_file
    return None


def create_voice(
    name,
    reference_path,
    reference_text=None,
    gender=None,
    description=None,
    model_dir=None,
    force_cpu=False,
):
    """
    Create a voice profile from a reference audio file.
    
    Args:
        name: Name for the voice (used as identifier)
        reference_path: Path to the reference audio file
        reference_text: Transcript of the reference audio (optional)
        gender: Gender label for the voice (male/female)
        description: Description for this voice
        model_dir: Model directory path
        force_cpu: Whether to force CPU usage
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if voice already exists
        voice_file = get_voice_path(name)
        if voice_file:
            logging.error(f"Voice '{name}' already exists at {voice_file}")
            return False
            
        # Check if reference audio exists
        if not os.path.exists(reference_path):
            logging.error(f"Reference audio file not found: {reference_path}")
            return False
            
        # Validate reference audio format using audio.py
        validate_reference_audio(reference_path)
        
        # Initialize model with appropriate device
        device = select_device(None, force_cpu)
        model_dir = get_default_value(model_dir, "model_dir", "pretrained_models/Spark-TTS-0.5B")
        model = SparkTTS(model_dir, device)
        
        # Extract voice features using audio.py's extract_voice_tokens function
        logging.info(f"Extracting voice features from: {reference_path}")
        global_tokens, _ = extract_voice_tokens(reference_path, model)
        
        if global_tokens is None:
            logging.error("Failed to extract voice features")
            return False
            
        # Create voice configuration
        audio_info = {}
        try:
            # Get audio metadata
            audio_data, sr = sf.read(reference_path)
            audio_info = {
                "sample_rate": sr,
                "duration": len(audio_data) / sr,
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            }
        except Exception as e:
            logging.warning(f"Could not read audio metadata: {str(e)}")
            
        # Generate voice profile with privacy protection (remove private paths)
        voice_config = {
            "name": name,
            "description": description or f"Voice created from {os.path.basename(reference_path)}",
            "gender": gender,
            "created_at": datetime.now().isoformat(),
            "reference_audio": {
                "hash": get_audio_hash(reference_path),
                "text": reference_text,
                "info": audio_info
            },
            "model_info": {
                "model_name": os.path.basename(os.path.normpath(model_dir))
            },
            "global_tokens": global_tokens.tolist()
        }
        
        # Get voices directory from config
        voices_dir = get_voices_dir()
        
        # Save voice profile
        voice_file = os.path.join(voices_dir, f"{name}.json")
        with open(voice_file, 'w', encoding='utf-8') as f:
            json.dump(voice_config, f, ensure_ascii=False)
            
        logging.info(f"Voice '{name}' created and saved to {voice_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating voice: {str(e)}")
        traceback.print_exc()
        return False


def load_voice(voice_name):
    """
    Load a voice configuration.
    
    Args:
        voice_name: Name of the voice to load
        
    Returns:
        Dict containing voice configuration or None if not found
    """
    voice_file = get_voice_path(voice_name)
    if not voice_file:
        logging.error(f"Voice '{voice_name}' not found")
        return None
        
    try:
        with open(voice_file, 'r', encoding='utf-8') as f:
            voice_config = json.load(f)
        return voice_config
    except Exception as e:
        logging.error(f"Error loading voice '{voice_name}': {str(e)}")
        return None


def get_voice_tokens(voice_name):
    """
    Get the global tokens for a voice.
    
    Args:
        voice_name: Name of the voice
        
    Returns:
        tuple: (global_tokens, None) as torch.Tensor object or (None, None) if not found
    """
    voice_config = load_voice(voice_name)
    if not voice_config or "global_tokens" not in voice_config:
        return None, None
        
    global_tokens = torch.tensor(voice_config["global_tokens"])
    
    return global_tokens, None


def list_voices(json_format=False):
    """
    List all available voices.
    
    Args:
        json_format: Whether to return JSON format (True) or print to console (False)
        
    Returns:
        List of voice configurations if json_format is True, otherwise None
    """
    voices = []
    
    # Get voices directory from config
    voices_dir = get_voices_dir()
    logging.info(f"Looking for voices in: {voices_dir}")
    
    if not os.path.exists(voices_dir):
        logging.warning(f"Voices directory does not exist: {voices_dir}")
        if not json_format:
            print("No voices found - directory does not exist")
        return [] if json_format else None
        
    for file in os.listdir(voices_dir):
        if file.endswith(".json"):
            voice_file = os.path.join(voices_dir, file)
            try:
                with open(voice_file, 'r', encoding='utf-8') as f:
                    voice_config = json.load(f)
                voices.append(voice_config)
            except Exception as e:
                logging.warning(f"Could not read voice file {file}: {str(e)}")
                
    if json_format:
        return voices
        
    # Print voice information in a formatted table
    if not voices:
        print(f"No voices found in {voices_dir}")
        return None
        
    print(f"\nFound {len(voices)} voices in {voices_dir}:")
    print("-" * 80)
    print(f"{'Name':<20} {'Gender':<10} {'Created':<20} {'Description':<30}")
    print("-" * 80)
    
    for voice in voices:
        name = voice.get("name", "Unknown")
        gender = voice.get("gender", "N/A")
        created_at = voice.get("created_at", "N/A")
        description = voice.get("description", "")
        
        # Format creation date
        if created_at and created_at != "N/A":
            try:
                dt = datetime.fromisoformat(created_at)
                created_at = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
                
        # Truncate description if too long
        if len(description) > 30:
            description = description[:27] + "..."
            
        # Ensure all values are strings to avoid formatting issues with None
        gender = str(gender) if gender is not None else "N/A"
        name = str(name) if name is not None else "Unknown"
        created_at = str(created_at) if created_at is not None else "N/A"
        description = str(description) if description is not None else ""
            
        print(f"{name:<20} {gender:<10} {created_at:<20} {description:<30}")
        
    print("-" * 80)
    return voices if voices else None


def delete_voice(voice_name):
    """
    Delete a voice profile.
    
    Args:
        voice_name: Name of the voice to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    voice_file = get_voice_path(voice_name)
    if not voice_file:
        logging.error(f"Voice '{voice_name}' not found")
        return False
        
    try:
        os.remove(voice_file)
        logging.info(f"Voice '{voice_name}' deleted")
        return True
    except Exception as e:
        logging.error(f"Error deleting voice '{voice_name}': {str(e)}")
        return False 