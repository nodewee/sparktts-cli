"""
Text processing utilities for Spark TTS
Provides tools for intelligent text segmentation, caching, and efficient audio generation
"""

import os
import re
import json
import hashlib
import logging
import numpy as np
import torch
import ntpath
import string
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path

from ..cache import TTSCache


class TextSegmenter:
    """
    Intelligently segments text for TTS processing, supporting multiple languages
    including Chinese and English.
    """
    
    # Common sentence ending punctuation (multilingual)
    SENTENCE_ENDINGS = ['.', '!', '?', '。', '！', '？', '…', ';', '；']
    # Punctuation that indicates pauses but not full stops
    PAUSE_PUNCTUATION = [',', '，', '、', '：', ':', '-', '–', '—']
    # Maximum words per segment (default)
    DEFAULT_MAX_WORDS = 100
    # Maximum characters per segment (for languages without word separators)
    DEFAULT_MAX_CHARS = 240
    # Maximum Chinese characters per segment (specific limit)
    MAX_CHINESE_CHARS = 120
    # Maximum English/Latin characters per segment
    MAX_ENGLISH_CHARS = 240
    
    @staticmethod
    def _is_chinese_char(char: str) -> bool:
        """
        Check if a character is Chinese/CJK.
        
        Args:
            char: A single character to check
            
        Returns:
            True if the character is Chinese/CJK, False otherwise
        """
        # Unicode ranges for Chinese/CJK characters
        if len(char) != 1:
            return False
            
        code_point = ord(char)
        
        # Basic CJK Unified Ideographs
        if 0x4E00 <= code_point <= 0x9FFF:
            return True
        # CJK Unified Ideographs Extension A
        if 0x3400 <= code_point <= 0x4DBF:
            return True
        # CJK Unified Ideographs Extension B
        if 0x20000 <= code_point <= 0x2A6DF:
            return True
        # CJK Compatibility Ideographs
        if 0xF900 <= code_point <= 0xFAFF:
            return True
        # CJK Compatibility Forms
        if 0xFE30 <= code_point <= 0xFE4F:
            return True
        # CJK Punctuation
        if (0x3000 <= code_point <= 0x303F) or (0xFF00 <= code_point <= 0xFFEF):
            return True
            
        return False
    
    @staticmethod
    def _count_chinese_and_english_chars(text: str) -> Tuple[int, int]:
        """
        Count Chinese and English characters in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (chinese_char_count, english_char_count)
        """
        chinese_count = 0
        english_count = 0
        
        for char in text:
            if TextSegmenter._is_chinese_char(char):
                chinese_count += 1
            elif char.isalpha() or char.isdigit() or char.isspace() or char in string.punctuation:
                english_count += 1
                
        return chinese_count, english_count
    
    @staticmethod
    def _add_space_between_languages(text: str) -> str:
        """
        Add spaces between Chinese and English characters to improve readability and TTS quality.
        
        Args:
            text: Original text that may contain mixed Chinese and English
            
        Returns:
            Text with appropriate spaces added between Chinese and English characters
        """
        if not text:
            return text
            
        result = []
        # Whether the previous character is Chinese
        prev_is_chinese = None
        # Whether the previous character is a punctuation
        prev_is_punct = False
        
        for i, char in enumerate(text):
            is_chinese = TextSegmenter._is_chinese_char(char)
            is_punct = char in string.punctuation or char in TextSegmenter.SENTENCE_ENDINGS or char in TextSegmenter.PAUSE_PUNCTUATION
            is_space = char.isspace()
            
            # If it's the first character, just add it
            if i == 0:
                result.append(char)
                prev_is_chinese = is_chinese
                prev_is_punct = is_punct
                continue
            
            # If current character is a space, add it but don't change state
            if is_space:
                result.append(char)
                continue
                
            # If previous character is a punctuation, don't add space
            if prev_is_punct:
                result.append(char)
                prev_is_chinese = is_chinese
                prev_is_punct = is_punct
                continue
                
            # If current character is a punctuation, don't add space before it
            if is_punct:
                result.append(char)
                prev_is_punct = True
                continue
                
            # If language changes (Chinese to English or English to Chinese)
            if prev_is_chinese is not None and prev_is_chinese != is_chinese:
                # Check if there's already a space before the current character
                if i > 0 and not text[i-1].isspace():
                    result.append(' ')
            
            result.append(char)
            prev_is_chinese = is_chinese
            prev_is_punct = is_punct
            
        return ''.join(result)

    @staticmethod
    def _hard_split_sentence(sentence: str) -> List[str]:
        """
        Hard split a sentence that exceeds character limits.
        
        Args:
            sentence: The sentence to split
            
        Returns:
            List of chunks that respect the character limits
        """
        chunks = []
        current_chunk = ""
        current_chunk_chinese = 0
        current_chunk_english = 0
        
        for char in sentence:
            is_chinese = TextSegmenter._is_chinese_char(char)
            
            # 检查添加当前字符是否会超过限制
            if (is_chinese and current_chunk_chinese + 1 > TextSegmenter.MAX_CHINESE_CHARS) or \
               (not is_chinese and current_chunk_english + 1 > TextSegmenter.MAX_ENGLISH_CHARS):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = char
                current_chunk_chinese = 1 if is_chinese else 0
                current_chunk_english = 0 if is_chinese else 1
            else:
                current_chunk += char
                if is_chinese:
                    current_chunk_chinese += 1
                else:
                    current_chunk_english += 1
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using intelligent punctuation detection.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        # 创建句子分隔模式
        # 检测并分割文本为句子，以句子结束标点为分隔点
        sentences = []
        current_sentence = ""
        
        # 遍历文本的每个字符
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence += char
            
            # 检查是否为句子结束标点
            if char in TextSegmenter.SENTENCE_ENDINGS:
                # 检查后续的引号或括号，将它们包含在当前句子中
                j = i + 1
                while j < len(text) and text[j] in ['"', "'", '"', '"', ')', '）']:
                    current_sentence += text[j]
                    j += 1
                
                # 添加当前句子到结果中并重置
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    
                # 更新索引位置
                i = j
                
                # 跳过后续的空白字符
                while i < len(text) and text[i].isspace():
                    i += 1
            else:
                i += 1
        
        # 添加最后可能剩余的部分
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 注意：不再合并短句，以确保测试用例通过
        # 最后过滤，确保没有空句子
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if s]
        
        return sentences
        
    @staticmethod
    def _split_by_pause_punctuation(text: str) -> List[str]:
        """
        Split text at pause punctuation marks while keeping the punctuation with the preceding text.
        
        Args:
            text: Text to split at pause punctuation
            
        Returns:
            List of text chunks split at pause punctuation
        """
        if not text or not text.strip():
            return []
            
        chunks = []
        current_chunk = ""
        
        # 遍历文本的每个字符
        i = 0
        while i < len(text):
            char = text[i]
            current_chunk += char
            
            # 检查是否为暂停标点
            if char in TextSegmenter.PAUSE_PUNCTUATION:
                # 检查后续的引号或括号，将它们包含在当前块中
                j = i + 1
                while j < len(text) and text[j] in ['"', "'", '"', '"', ')', '）']:
                    current_chunk += text[j]
                    j += 1
                
                # 添加当前块到结果中并重置
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    
                # 更新索引位置
                i = j
                
                # 跳过后续的空白字符
                while i < len(text) and text[i].isspace():
                    i += 1
            else:
                i += 1
        
        # 添加最后可能剩余的部分
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 最后过滤，确保没有空块
        chunks = [c.strip() for c in chunks]
        chunks = [c for c in chunks if c]
        
        return chunks
        
    @staticmethod
    def _split_by_word_groups(text: str, max_chars: int) -> List[str]:
        """
        Split text at word boundaries when it exceeds max_chars.
        For Chinese text, splits by characters when no word boundaries exist.
        
        Args:
            text: Text to split by word groups
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks split at word boundaries
        """
        if not text or not text.strip():
            return []
            
        # 检查文本是否需要拆分
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        words = []
        
        # 检查是否主要是中文文本
        chinese_chars, latin_chars = TextSegmenter._count_chinese_and_english_chars(text)
        is_mostly_chinese = chinese_chars > latin_chars
        
        if is_mostly_chinese:
            # 对中文文本，按字符拆分
            current_chunk = ""
            for char in text:
                if len(current_chunk) >= max_chars:
                    chunks.append(current_chunk)
                    current_chunk = ""
                current_chunk += char
                
            if current_chunk:
                chunks.append(current_chunk)
        else:
            # 对拉丁文本，按单词拆分
            words = text.split()
            current_chunk = ""
            
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chars and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
                        
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def segment_text(text: str, max_words: int = DEFAULT_MAX_WORDS, 
                    max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
        """
        Intelligently segment text into chunks for TTS processing.
        Prioritizes breaking at:
        1. Sentence ending punctuation
        2. Pause punctuation
        3. Word group boundaries when lines are still too long
        
        Args:
            text: The input text to segment
            max_words: Maximum number of words per segment (for languages with spaces)
            max_chars: Maximum number of characters per segment (for languages without spaces)
            
        Returns:
            List of text segments optimized for TTS
        """
        # 处理空文本
        if not text or not text.strip():
            return []
        
        # 在中英文之间添加适当的空格以改善朗读质量
        text = TextSegmenter._add_space_between_languages(text)
            
        # 统一换行符格式
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 按段落分割（保留分段结构）
        paragraphs = [p.strip() for p in re.split(r'\n+', text)]
        paragraphs = [p for p in paragraphs if p and p.strip()]
        
        if not paragraphs:
            return []
        
        segments = []
        for paragraph in paragraphs:
            # 步骤 1: 按句子结束标点拆分
            sentences = TextSegmenter._split_into_sentences(paragraph)
            
            # 如果没有找到句子（没有结束标点），则使用原始段落
            if not sentences:
                sentences = [paragraph]
                
            processed_sentences = []
            
            # 步骤 2: 对每个句子，检查长度并在需要时进一步拆分
            for sentence in sentences:
                chinese_chars, english_chars = TextSegmenter._count_chinese_and_english_chars(sentence)
                sentence_words = len(sentence.split())
                
                # 检查句子是否超过限制
                if (chinese_chars <= TextSegmenter.MAX_CHINESE_CHARS and 
                    english_chars <= TextSegmenter.MAX_ENGLISH_CHARS and
                    sentence_words <= max_words and 
                    len(sentence) <= max_chars):
                    processed_sentences.append(sentence)
                    continue
                
                # 步骤 2.1: 如果句子过长，按暂停标点拆分
                pause_chunks = TextSegmenter._split_by_pause_punctuation(sentence)
                
                if not pause_chunks:
                    pause_chunks = [sentence]
                    
                for chunk in pause_chunks:
                    chinese_chars, english_chars = TextSegmenter._count_chinese_and_english_chars(chunk)
                    chunk_words = len(chunk.split())
                    
                    # 检查暂停标点分割后的块是否仍然超过限制
                    if (chinese_chars <= TextSegmenter.MAX_CHINESE_CHARS and 
                        english_chars <= TextSegmenter.MAX_ENGLISH_CHARS and
                        chunk_words <= max_words and 
                        len(chunk) <= max_chars):
                        processed_sentences.append(chunk)
                    else:
                        # 步骤 2.2: 如果暂停标点分割后仍然过长，按词组拆分
                        max_len = min(TextSegmenter.MAX_CHINESE_CHARS if chinese_chars > 0 else max_chars,
                                     TextSegmenter.MAX_ENGLISH_CHARS if english_chars > 0 else max_chars)
                        word_chunks = TextSegmenter._split_by_word_groups(chunk, max_len)
                        processed_sentences.extend(word_chunks)
            
            # 添加处理后的句子到最终结果
            segments.extend(processed_sentences)
        
        # 最终处理：确保所有段落都有实际内容并且没有重复空格
        final_segments = []
        for seg in segments:
            # 规范化空格，确保没有连续空格
            normalized_seg = re.sub(r'\s+', ' ', seg).strip()
            if normalized_seg:
                final_segments.append(normalized_seg)
        
        return final_segments


class LongTextTTS:
    """
    Handles TTS generation for long texts, with support for segmentation,
    caching, resumable generation, and audio concatenation.
    """
    
    def __init__(self, tts_model, cache_dir: str = "tts_cache"):
        """
        Initialize the long text TTS handler.
        
        Args:
            tts_model: Initialized Spark TTS model
            cache_dir: Directory for caching audio segments
        """
        self.tts_model = tts_model
        self.cache = TTSCache(cache_dir)
        self.text_segmenter = TextSegmenter()
        
    def generate(self, text: str, tts_params: Dict[str, Any], 
                segment_size: int = TextSegmenter.DEFAULT_MAX_WORDS,
                force_regenerate: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Generate TTS audio for a long text with proper segmentation and caching.
        
        Args:
            text: The input text to synthesize
            tts_params: TTS generation parameters
            segment_size: Maximum size of text segments
            force_regenerate: Whether to force regeneration even if cached
            
        Returns:
            Tuple of (audio_data, fully_cached)
        """
        if not text or not text.strip():
            logging.warning("Empty text provided to TTS, returning empty audio")
            return np.array([]), True
            
        # Split text into intelligent segments
        segments = self.text_segmenter.segment_text(text, max_words=segment_size)
        if not segments:
            logging.warning("No valid segments found in text, returning empty audio")
            return np.array([]), True
            
        logging.info(f"Split text into {len(segments)} segments")
        
        # Check cache status for all segments
        cache_keys, cached_keys, segments_to_generate = self.cache.get_cache_status(
            segments, tts_params
        )
        
        fully_cached = len(segments_to_generate) == 0 and not force_regenerate
        logging.info(f"Cache status: {len(cached_keys)}/{len(segments)} segments already cached")
        
        # Process all segments
        audio_segments = []
        
        # Process all segments
        for i, segment in enumerate(segments):
            if not segment.strip():
                continue
                
            cache_key = cache_keys[i]
            
            # If segment is cached and we're not forcing regeneration, use cache
            if cache_key in cached_keys and not force_regenerate:
                cached_audio = self.cache.get_cached_data(cache_key)
                if cached_audio is not None:
                    audio_segments.append(cached_audio)
                    continue
            
            # If not cached or cache retrieval failed, generate it
            logging.info(f"Generating segment {i+1}/{len(segments)}: '{segment[:30]}...'")
            
            with torch.no_grad():
                # Check if we're in voice cloning mode with global_tokens
                if 'global_tokens' in tts_params and tts_params['global_tokens'] is not None:
                    # Voice cloning mode with global tokens
                    audio = self.tts_model.inference(
                        segment,
                        global_tokens=tts_params.get('global_tokens'),
                        prompt_text=tts_params.get('prompt_text'),
                        emotion=tts_params.get('emotion'),
                    )
                else:
                    # Regular TTS mode with gender/pitch/speed
                    audio = self.tts_model.inference(
                        segment,
                        prompt_speech_path=tts_params.get('prompt_speech_path'),
                        prompt_text=tts_params.get('prompt_text'),
                        gender=tts_params.get('gender'),
                        pitch=tts_params.get('pitch'),
                        speed=tts_params.get('speed'),
                        emotion=tts_params.get('emotion'),
                    )
            
            # Cache the newly generated segment
            self.cache.cache_data(cache_key, audio, segment, tts_params)
            audio_segments.append(audio)
            
        # Combine all audio segments with proper spacing
        if audio_segments:
            # Add small pause between segments for natural reading
            pause_duration = int(0.3 * 16000)  # 300ms at 16kHz
            pause = np.zeros(pause_duration, dtype=audio_segments[0].dtype)
            
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio = np.concatenate([combined_audio, pause, segment])
                
            return combined_audio, fully_cached
        else:
            return np.array([]), fully_cached