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

import re
import torch
import os
import json
from typing import Tuple, Union, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Use the common utilities
from .utils import DEFAULT_CONFIG, get_default_value, ensure_dir_exists

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP, EMO_MAP, TokenParser
from .audio import extract_voice_tokens

# Constants for voice control
GENDER_MAP = {
    "male": 0,
    "female": 1,
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

TASK_TOKEN_MAP = {
    "tts": "<|TTS|>",
    "controllable_tts": "<|ControlTTS|>",
}

# Emotion map for controllable TTS
EMO_MAP = {
    "NEUTRAL": 0,
    "ANGRY": 1,
    "HAPPY": 2,
    "SAD": 3,
    "SURPRISED": 4,
}

class TokenParser:
    """解析文本中的各种token标记"""
    
    @staticmethod
    def emotion(emotion: str) -> str:
        """
        解析情感标签
        
        Args:
            emotion: 情感标签，例如HAPPY, SAD等
            
        Returns:
            情感token字符串
        """
        emotion = emotion.strip().upper()
        if emotion in EMO_MAP:
            return f"<|emo_{EMO_MAP[emotion]}|>"
        return ""

class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Union[str, Path] = None, device: torch.device = None):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        # Use values from config or defaults
        self.model_dir = get_default_value(model_dir, "model_dir", "pretrained_models/Spark-TTS-0.5B")
        
        # Set default device to CPU if none provided
        if device is None:
            device_str = DEFAULT_CONFIG.get("device", "cpu")
            device = torch.device(device_str)
            
        self.device = device
        self.configs = load_config(f"{self.model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Union[str, Path],
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process the input text and prompt audio to create a proper prompt for the model.
        
        Args:
            text: The text to be converted to speech
            prompt_speech_path: Path to the audio file used as reference
            prompt_text: Optional transcript of prompt audio
            
        Returns:
            tuple: (prompt_string, global_token_ids)
        """
        if prompt_speech_path is None:
            # If no prompt speech is provided, use default gender/control parameters
            return self.process_prompt_control("female", text=text)
            
        # 使用特征提取器从参考语音中提取说话人特征
        # 使用audio.py中的extract_voice_tokens函数替代直接调用tokenizer
        global_token_ids, _ = extract_voice_tokens(prompt_speech_path, self)
        
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        text: str = "",
        emotion: Optional[str] = None,
    ) -> Tuple[str, None]:
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.
            emotion (str, optional): Emotion parameter (e.g., "HAPPY", "SAD", "ANGRY").

        Return:
            Tuple[str, None]: (Input prompt, None)
        """
        # 确保gender不为None且值有效
        if gender is None:
            gender = "female"
            logging.info("No gender specified, using default: female")
            
        assert gender in GENDER_MAP.keys(), f"Gender '{gender}' not in {list(GENDER_MAP.keys())}"
        
        # 使用默认值"moderate"替代None值
        pitch = pitch or "moderate"
        speed = speed or "moderate"
        
        # 检查pitch和speed值的有效性
        assert pitch in LEVELS_MAP.keys(), f"Pitch '{pitch}' not in {list(LEVELS_MAP.keys())}"
        assert speed in LEVELS_MAP.keys(), f"Speed '{speed}' not in {list(LEVELS_MAP.keys())}"

        # 获取对应的数值ID
        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        # 构建token字符串
        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        # 构建属性tokens列表
        attribute_tokens_list = [gender_tokens, pitch_label_tokens, speed_label_tokens]
        
        # 如果提供了情绪参数，添加到属性tokens
        if emotion is not None and emotion.strip():
            emotion_upper = emotion.upper().strip()
            if emotion_upper in EMO_MAP:
                emotion_token = TokenParser.emotion(emotion_upper)
                attribute_tokens_list.append(emotion_token)
                logging.info(f"Using emotion: {emotion_upper}")
            else:
                available_emotions = list(EMO_MAP.keys())
                logging.warning(f"Emotion '{emotion}' not supported. Available options: {available_emotions}")

        # 合并所有属性tokens
        attribute_tokens = "".join(attribute_tokens_list)

        # 构建控制TTS输入
        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribute_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs), None

    def _prepare_prompt(
        self,
        text: str,
        prompt_speech_path: Optional[Union[str, Path]] = None,
        prompt_text: Optional[str] = None, 
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        emotion: Optional[str] = None,
        global_tokens: Optional[torch.Tensor] = None,
        semantic_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Prepare inference prompt based on provided parameters
        
        Args:
            text: Text to synthesize
            prompt_speech_path: Path to reference audio file
            prompt_text: Transcript of the reference audio
            gender: Gender (male/female) for controllable TTS
            pitch: Pitch level for controllable TTS
            speed: Speed level for controllable TTS
            emotion: Emotion for controllable TTS
            global_tokens: Pre-computed global tokens
            semantic_tokens: Kept for backward compatibility but not used
            
        Returns:
            tuple: (prompt, global_token_ids)
        """
        # 使用默认配置
        emotion = get_default_value(emotion, "emotion", None)
        
        # 处理不同的情况: 全局语音特征优先级
        # 1. 如果提供了global_tokens，优先使用
        # 2. 如果没有tokens但有prompt_speech_path，从音频提取特征
        # 3. 如果以上都没有，使用gender+控制参数生成语音
        
        if global_tokens is not None:
            # Case 1: 使用预先计算的全局语音特征
            logging.info("Using provided global tokens for voice cloning")
            global_token_ids = global_tokens
            if len(global_token_ids.shape) == 1:
                global_token_ids = global_token_ids.unsqueeze(0)
            
            global_tokens_str = "".join(
                [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
            )
            
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens_str,
                "<|end_global_token|>",
            ]
            prompt = "".join(inputs)
            
        elif prompt_speech_path is not None:
            # Case 2: 从参考音频提取特征
            logging.info(f"Extracting voice features from reference audio: {prompt_speech_path}")
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
            
        else:
            # Case 3: 使用控制参数生成语音 (没有克隆的语音)
            logging.info(f"Using controllable TTS with gender={gender}, pitch={pitch}, speed={speed}")
            
            # 确保有默认性别
            if gender is None:
                gender = "female"
                logging.info("No gender specified, using default: female")
                
            prompt, global_token_ids = self.process_prompt_control(
                gender, pitch, speed, text, emotion
            )
        
        return prompt, global_token_ids

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Optional[Union[str, Path]] = None,
        prompt_text: Optional[str] = None,
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        emotion: Optional[str] = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        global_tokens: Optional[torch.Tensor] = None,
        semantic_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            emotion (str, optional): Emotion for the speech (e.g., "HAPPY", "SAD", "NEUTRAL").
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.
            global_tokens (torch.Tensor, optional): Pre-computed global features (if provided, 
                                                  don't extract from prompt_speech_path)
            semantic_tokens (torch.Tensor, optional): Kept for backward compatibility but not used

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        # Regular TTS generation path
        # Prepare inference prompt
        prompt, global_token_ids = self._prepare_prompt(
            text, prompt_speech_path, prompt_text, gender, pitch, speed, emotion, global_tokens, None
        )
                
        # Tokenize the prompt
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens into text
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        # For voice creation, extract global tokens from output
        if gender is not None and global_token_ids is None:
            # Try to extract global tokens from the output
            global_token_matches = re.findall(r"bicodec_global_(\d+)", predicts)
            if global_token_matches:
                global_token_ids = (
                    torch.tensor([int(token) for token in global_token_matches])
                    .long()
                    .unsqueeze(0)
                )
                logging.info(f"Extracted {len(global_token_matches)} global tokens from output")
            else:
                logging.info("No global tokens found in output, using configured defaults")
        
        # Convert semantic tokens back to waveform
        if global_token_ids is None:
            # When global_token_ids is still None after all processing
            logging.warning("No global tokens available, creating empty token tensor")
            
            # Create a tensor with the correct shape for the BiCodec model
            # The shape should be [batch_size, sequence_length]
            # For speaker_encoder the expected shape for a single entry is [1, num_tokens]
            # where num_tokens is specific to the model architecture (often 8)
            n_global_tokens = self.configs.get("n_global_tokens", 8)
            global_token_ids = torch.zeros(1, n_global_tokens, dtype=torch.long, device=self.device)
        elif len(global_token_ids.shape) == 3:
            # If shape is [1, 1, n], reshape to [1, n]
            global_token_ids = global_token_ids.squeeze(1)
            
        # Log tensor shape for debugging
        logging.debug(f"Global token shape: {global_token_ids.shape}, Semantic token shape: {pred_semantic_ids.shape}")
        
        try:
            wav = self.audio_tokenizer.detokenize(
                global_token_ids.to(self.device),
                pred_semantic_ids.to(self.device),
            )
            return wav
        except RuntimeError as e:
            logging.error(f"Detokenize error: {str(e)}")
            logging.error(f"Global token shape: {global_token_ids.shape}, Semantic token shape: {pred_semantic_ids.shape}")
            raise