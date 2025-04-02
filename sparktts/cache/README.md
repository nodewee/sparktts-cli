# Spark TTS Cache Module

This module provides cache functionality for Spark TTS, allowing efficient storage and retrieval of generated audio segments.

## Overview

The cache module provides a flexible caching system with the following features:

- Abstract `Cache` interface that defines standard cache operations
- Concrete implementation `TTSCache` specialized for TTS audio caching
- Support for generating cache keys based on text and TTS parameters
- Efficient storage and retrieval of cached audio segments
- Cache management functions (clearing, status checking)

## Usage

### Basic Usage

```python
from sparktts.cache import TTSCache

# Initialize cache with a directory
cache = TTSCache(cache_dir="path/to/cache")

# Generate a cache key for text with TTS parameters
cache_key = cache.generate_cache_key(text="Hello world", tts_params={
    "model_dir": "/path/to/model",
    "prompt_speech_path": "/path/to/speaker.wav",
    "gender": "female",
    "pitch": "moderate",
    "speed": "moderate",
    "emotion": "HAPPY"
})

# Retrieve cached audio
audio = cache.get_cached_data(cache_key)
if audio is not None:
    # Use cached audio
    print("Using cached audio")
else:
    # Generate new audio and cache it
    audio = generate_audio(text, params)
    cache.cache_data(cache_key, audio, text, params)
```

### Cache Status for Multiple Segments

```python
# Get cache status for a list of text segments
segments = ["Segment 1", "Segment 2", "Segment 3"]
tts_params = {...}  # TTS parameters

cache_keys, cached_keys, segments_to_generate = cache.get_cache_status(segments, tts_params)
print(f"{len(cached_keys)}/{len(segments)} segments are cached")
```

### Clearing the Cache

```python
# Clear all cached audio
success = cache.clear_cache()
if success:
    print("Cache cleared successfully")
else:
    print("Failed to clear cache")
```

## Extending with Custom Cache Implementations

To create a custom cache implementation, extend the base `Cache` class:

```python
from sparktts.cache import Cache

class MyCustomCache(Cache):
    def __init__(self, cache_dir: str):
        # Initialize your cache
        pass
        
    def generate_cache_key(self, data, params):
        # Generate a unique key for the data and parameters
        pass
        
    def get_cached_data(self, cache_key):
        # Retrieve cached data
        pass
        
    def cache_data(self, cache_key, data, original_input, params):
        # Cache the data
        pass
        
    def clear_cache(self):
        # Clear the cache
        pass
        
    def get_cache_status(self, items, params):
        # Check which items are cached
        pass
```

## Integration with TTS

The cache module is designed to work seamlessly with the TTS process:

```python
from sparktts.cache import TTSCache
from sparktts.utils.text_processing import TextSegmenter, LongTextTTS
from sparktts.models import load_tts_model

# Initialize components
model = load_tts_model(...)
cache = TTSCache(cache_dir="tts_cache")
long_text_handler = LongTextTTS(model, cache_dir="tts_cache")

# Generate audio with caching
audio, fully_cached = long_text_handler.generate(
    text="Your long text to synthesize...",
    tts_params={...},
    force_regenerate=False
)
```