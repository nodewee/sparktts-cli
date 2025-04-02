"""
Cache module for Spark TTS.
Provides cache functionality for storing and retrieving generated audio.
"""

from .tts_cache import Cache, TTSCache

__all__ = ["Cache", "TTSCache"]