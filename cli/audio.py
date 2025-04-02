import os
import numpy as np
import soundfile as sf
import logging
import traceback
import torch
from .utils import DEFAULT_CONFIG

def extract_voice_tokens(audio_path, model):
    """
    Extract global tokens from audio
    
    Args:
        audio_path: Audio file path
        model: SparkTTS model instance
    
    Returns:
        global_tokens: Global features
        None: For compatibility (previously semantic features)
    """
    logging.info(f"Extracting global tokens from audio: {audio_path}")
    try:
        # Call SparkTTS's audio_tokenizer for feature extraction
        global_token_ids, semantic_token_ids = model.audio_tokenizer.tokenize(audio_path)
        return global_token_ids, None
    except Exception as e:
        logging.error(f"Failed to extract global tokens: {str(e)}")
        traceback.print_exc()
        return None, None

def validate_reference_audio(audio_path):
    """
    Validate that reference audio meets requirements (16kHz mono)
    
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    try:
        audio_data, sr = sf.read(audio_path)
        # Check sample rate
        if sr != 16000:
            raise ValueError(f"Reference audio has sample rate {sr}Hz, must be 16kHz")
        # Check channel count
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            raise ValueError(f"Reference audio has {audio_data.shape[1]} channels, must be mono")
        return True
    except Exception as e:
        logging.error(f"Failed to validate reference audio: {str(e)}")
        raise 