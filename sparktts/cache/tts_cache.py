"""
TTS cache module for Spark TTS.
Provides abstract cache interface and concrete TTSCache implementation
for caching TTS-generated audio segments.
"""

import abc
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import torch


class Cache(Protocol):
    """
    Protocol defining the interface for cache implementations.
    Provides methods for storing, retrieving, and managing cached data.
    """
    
    def __init__(self, cache_dir: str) -> None:
        """
        Initialize the cache with a specified directory.
        
        Args:
            cache_dir: Directory to store cached data
        """
        ...
    
    def generate_cache_key(self, data: Any, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for the given data and parameters.
        
        Args:
            data: The data to generate a key for
            params: Parameters that affect the processing of the data
            
        Returns:
            A unique cache key string
        """
        ...
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached data for a given cache key.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached data if available, None otherwise
        """
        ...
    
    def cache_data(self, cache_key: str, data: Any, 
                  original_input: Any, params: Dict[str, Any]) -> bool:
        """
        Cache data for future use.
        
        Args:
            cache_key: Cache key for the data
            data: The data to cache
            original_input: The original input that generated this data
            params: Parameters used to generate the data
            
        Returns:
            True if caching was successful, False otherwise
        """
        ...
    
    def clear_cache(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            True if the cache was successfully cleared, False otherwise
        """
        ...
    
    def get_cache_status(self, items: List[Any], params: Dict[str, Any]) -> Tuple[List[str], List[str], List[Any]]:
        """
        Check which items are cached and which need processing.
        
        Args:
            items: List of items to check in cache
            params: Parameters used for processing
            
        Returns:
            Tuple of (all_cache_keys, cached_item_keys, items_to_process)
        """
        ...


class TTSCache:
    """
    Manages caching of TTS-generated audio segments for efficiency and resumability.
    Implements the Cache protocol using SQLite database for efficient storage and retrieval.
    """
    
    def __init__(self, cache_dir: str = "tts_cache") -> None:
        """
        Initialize the TTS cache manager.
        
        Args:
            cache_dir: Directory to store cached audio segments
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize SQLite database and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Create cache entries table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    params TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                ''')
                # Create index on created_at for efficient LRU operations
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)')
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Failed to initialize cache database: {e}")
    
    @staticmethod
    def _extract_basename(file_path: str) -> str:
        """
        Extract base filename from a path, without extension.
        
        Args:
            file_path: Full path to a file
            
        Returns:
            Base filename without path and extension
        """
        if not file_path:
            return ""
            
        # Get just the last part of the path
        basename = os.path.basename(file_path)
        
        # Remove file extension if present
        name = os.path.splitext(basename)[0]
        return name if name else basename
    
    def _extract_model_params(self, tts_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract essential model parameters for caching.
        
        Args:
            tts_params: Full TTS parameters dictionary
            
        Returns:
            Dictionary with essential parameters for cache indexing
        """
        cache_params = {}
        
        # Extract model name
        if 'model_dir' in tts_params and tts_params['model_dir']:
            cache_params['model_name'] = self._extract_basename(tts_params['model_dir'])
        
        # Extract speaker information
        if 'prompt_speech_path' in tts_params and tts_params['prompt_speech_path']:
            cache_params['speaker'] = self._extract_basename(tts_params['prompt_speech_path'])
        
        # For voice cloning with pre-extracted global tokens
        if 'global_tokens' in tts_params and tts_params['global_tokens'] is not None:
            cache_params['speaker_token_hash'] = self._hash_tokens(tts_params['global_tokens'])
        
        # Include other critical parameters
        for param_name in ['gender', 'pitch', 'speed', 'emotion', 'segmentation_threshold']:
            if param_name in tts_params and tts_params[param_name] is not None:
                cache_params[param_name] = tts_params[param_name]
                
        return cache_params
    
    @staticmethod
    def _hash_tokens(tokens: Union[torch.Tensor, np.ndarray, List[Any]]) -> str:
        """
        Generate hash from token data.
        
        Args:
            tokens: Token data in various formats
            
        Returns:
            MD5 hash of the token data
        """
        if isinstance(tokens, torch.Tensor):
            tokens_array = tokens.detach().cpu().numpy()
        elif isinstance(tokens, np.ndarray):
            tokens_array = tokens
        else:
            tokens_array = np.array(tokens)
            
        return hashlib.md5(tokens_array.tobytes()).hexdigest()
    
    def generate_cache_key(self, text: str, tts_params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a text segment and TTS parameters.
        
        Args:
            text: Text segment to generate audio for
            tts_params: Parameters used for TTS generation
            
        Returns:
            Unique cache key
        """
        # Create a dict with text and essential parameters
        cache_dict = {"text": text.strip()}
        cache_dict.update(self._extract_model_params(tts_params))
        
        # Convert to JSON string and hash
        json_str = json.dumps(cache_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()
    
    def get_cached_data(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Retrieve cached audio for a given cache key.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Numpy array of audio data if cached, None otherwise
        """
        try:
            # Check if the entry exists in the database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM cache_entries WHERE cache_key = ?", (cache_key,))
                if cursor.fetchone() is None:
                    return None
            
            # Check if the corresponding file exists
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if not cache_file.exists():
                # Clean up database if file is missing
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                return None
            
            # Update access time for LRU tracking
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE cache_entries SET created_at = ? WHERE cache_key = ?", 
                              (time.time(), cache_key))
                conn.commit()
                
            return np.load(cache_file)
        except Exception as e:
            logging.error(f"Failed to load cached audio {cache_key}: {e}")
            return None
    
    def cache_data(self, cache_key: str, audio: np.ndarray, text: str, tts_params: Dict[str, Any]) -> bool:
        """
        Cache audio data for future use.
        
        Args:
            cache_key: Cache key for the audio
            audio: Audio data as numpy array
            text: Original text that generated this audio
            tts_params: TTS parameters used
            
        Returns:
            True if caching was successful, False otherwise
        """
        try:
            # Save audio data
            cache_file = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_file, audio)
            
            # Extract essential parameters for the index
            json_params = self._extract_model_params(tts_params)
            params_json = json.dumps(json_params, ensure_ascii=False)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache_entries (cache_key, text, params, created_at) VALUES (?, ?, ?, ?)",
                    (cache_key, text, params_json, time.time())
                )
                conn.commit()
            
            return True
        except Exception as e:
            logging.error(f"Failed to cache audio {cache_key}: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear all cached audio files and database entries.
        
        Returns:
            True if cache was successfully cleared, False otherwise
        """
        success = True
        # Delete all cached audio files
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except IOError as e:
                logging.error(f"Failed to delete cache file {cache_file}: {e}")
                success = False
        
        # Clear database entries
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries")
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Failed to clear database: {e}")
            success = False
            
        return success
        
    def get_cache_status(self, text_segments: List[str], 
                        tts_params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """
        Check which segments are cached and which need generation.
        
        Args:
            text_segments: List of text segments
            tts_params: TTS parameters
            
        Returns:
            Tuple of (cache_keys, cached_segments, segments_to_generate)
        """
        cache_keys = []
        cached_segment_keys = []
        segments_to_generate = []
        
        # Generate cache keys for all segments
        for segment in text_segments:
            if not segment.strip():
                continue
                
            cache_key = self.generate_cache_key(segment, tts_params)
            cache_keys.append(cache_key)
        
        # Check which keys exist in the database
        if cache_keys:
            placeholders = ','.join(['?'] * len(cache_keys))
            query = f"SELECT cache_key FROM cache_entries WHERE cache_key IN ({placeholders})"
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, cache_keys)
                    found_keys = {row[0] for row in cursor.fetchall()}
                    
                    # Separate into cached and to-generate
                    for i, key in enumerate(cache_keys):
                        if key in found_keys:
                            cached_segment_keys.append(key)
                        else:
                            segments_to_generate.append(text_segments[i])
            except sqlite3.Error as e:
                logging.error(f"Database error while checking cache status: {e}")
                # Fallback to regenerating all segments
                segments_to_generate = [s for s in text_segments if s.strip()]
                cached_segment_keys = []
        
        return cache_keys, cached_segment_keys, segments_to_generate
        
    def prune_cache(self, max_age_days: Optional[float] = None, max_entries: Optional[int] = None) -> int:
        """
        Prune cache entries based on age or count.
        
        Args:
            max_age_days: Maximum age in days for cache entries
            max_entries: Maximum number of entries to keep (removes oldest first)
            
        Returns:
            Number of entries removed
        """
        removed_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove by age if specified
                if max_age_days is not None:
                    max_age_seconds = max_age_days * 86400
                    cutoff_time = time.time() - max_age_seconds
                    
                    # Get keys to delete
                    cursor.execute("SELECT cache_key FROM cache_entries WHERE created_at < ?", (cutoff_time,))
                    keys_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    # Delete from database
                    cursor.execute("DELETE FROM cache_entries WHERE created_at < ?", (cutoff_time,))
                    removed_count += cursor.rowcount
                    
                    # Delete corresponding files
                    for key in keys_to_delete:
                        cache_file = self.cache_dir / f"{key}.npy"
                        if cache_file.exists():
                            cache_file.unlink()
                
                # Limit total entries if specified
                if max_entries is not None:
                    # Get count of entries
                    cursor.execute("SELECT COUNT(*) FROM cache_entries")
                    total_entries = cursor.fetchone()[0]
                    
                    if total_entries > max_entries:
                        # Calculate how many to remove
                        to_remove = total_entries - max_entries
                        
                        # Get oldest entries to remove
                        cursor.execute(
                            "SELECT cache_key FROM cache_entries ORDER BY created_at ASC LIMIT ?", 
                            (to_remove,)
                        )
                        keys_to_delete = [row[0] for row in cursor.fetchall()]
                        
                        # Remove from database
                        placeholders = ','.join(['?'] * len(keys_to_delete))
                        cursor.execute(f"DELETE FROM cache_entries WHERE cache_key IN ({placeholders})", 
                                      keys_to_delete)
                        removed_count += cursor.rowcount
                        
                        # Delete corresponding files
                        for key in keys_to_delete:
                            cache_file = self.cache_dir / f"{key}.npy"
                            if cache_file.exists():
                                cache_file.unlink()
                
                conn.commit()
                
            return removed_count
        except sqlite3.Error as e:
            logging.error(f"Error pruning cache: {e}")
            return 0