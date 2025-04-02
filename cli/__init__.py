"""
Spark-TTS CLI tools for text-to-speech generation and voice management.
"""

from .SparkTTS import SparkTTS
from .utils import select_device, ensure_dir_exists, get_timestamp_filename
from .audio import (
    extract_voice_tokens, 
    validate_reference_audio
)
from .inference import generate_tts_audio
from .voice import create_voice, list_voices, delete_voice, get_voice_path, load_voice, get_voice_tokens

__all__ = [
    'SparkTTS',
    'select_device',
    'ensure_dir_exists',
    'get_timestamp_filename',
    'extract_voice_tokens',
    'validate_reference_audio',
    'generate_tts_audio',
    'create_voice',
    'list_voices',
    'delete_voice',
    'get_voice_path',
    'load_voice',
    'get_voice_tokens',
]
