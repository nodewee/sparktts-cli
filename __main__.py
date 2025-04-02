#!/usr/bin/env python3
"""
Unified CLI entry point for Spark-TTS with clear separation of voice management and TTS generation
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# First, import only configuration utilities
from cli.utils import DEFAULT_CONFIG, setup_logging

# Set up logging immediately
setup_logging(level=logging.DEBUG)

# Set up environment variables based on configuration BEFORE importing TTS modules
if DEFAULT_CONFIG.get("force_cpu", False):
    # Set environment variables to force CPU usage
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback to CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
    os.environ["FORCE_CPU"] = "1"  # Custom environment variable
    
    # For PyTorch specifically
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS memory allocation
    
    # Log the configuration
    logging.info("Environment configured to force CPU usage")

# Now import the TTS-related modules after environment setup
from cli.inference import generate_tts_audio, select_device
from cli.voice import create_voice, list_voices, delete_voice, get_voice_path
from sparktts.utils.text_processing import TextSegmenter

def read_text_from_file(file_path):
    """Read text content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to read text file: {str(e)}")
        raise ValueError(f"Could not read text from file: {file_path}")

def segment_text_file(file_path, output_suffix="_segmented", max_chinese_chars=120, max_english_chars=240):
    """
    Segment text from a file and save the result to a new file.
    
    Args:
        file_path: Path to the input text file
        output_suffix: Suffix to add to the output filename
        max_chinese_chars: Maximum Chinese characters per segment
        max_english_chars: Maximum English characters per segment
        
    Returns:
        Path to the output file
    """
    try:
        # Read input text
        logging.info(f"Reading text from {file_path}")
        text = read_text_from_file(file_path)
        
        # Set custom limits for segmentation
        original_max_chinese = TextSegmenter.MAX_CHINESE_CHARS
        original_max_english = TextSegmenter.MAX_ENGLISH_CHARS
        
        TextSegmenter.MAX_CHINESE_CHARS = max_chinese_chars
        TextSegmenter.MAX_ENGLISH_CHARS = max_english_chars
        
        # Segment text
        logging.info("Segmenting text...")
        segments = TextSegmenter.segment_text(text)
        
        # Restore original limits
        TextSegmenter.MAX_CHINESE_CHARS = original_max_chinese
        TextSegmenter.MAX_ENGLISH_CHARS = original_max_english
        
        if not segments:
            logging.warning("No segments produced from input text")
            return None
            
        # Create output file path
        input_path = Path(file_path)
        output_path = input_path.parent / f"{input_path.stem}{output_suffix}{input_path.suffix}"
        
        # Write segmented text to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n---\n'.join(segments))
            
        logging.info(f"Segmented text written to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logging.error(f"Error segmenting text file: {str(e)}")
        traceback.print_exc()
        return None

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Spark-TTS - Text to Speech with Voice Management")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create voice management command
    voice_parser = subparsers.add_parser("voice", help="Voice management (create, list, delete)")
    voice_subparsers = voice_parser.add_subparsers(dest="voice_action", help="Voice operation")
    
    # Create voice subcommand
    create_parser = voice_subparsers.add_parser("create", help="Create a new voice from reference audio")
    create_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the voice (used as identifier)"
    )
    create_parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference audio file path (must be 16kHz mono WAV)"
    )
    create_parser.add_argument(
        "--text",
        type=str,
        help="Transcript of reference audio (optional, but recommended)"
    )
    create_parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female"],
        help="Gender label for the voice"
    )
    create_parser.add_argument(
        "--description",
        type=str,
        help="Description for this voice"
    )
    create_parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_CONFIG.get("model_dir", "pretrained_models/Spark-TTS-0.5B"),
        help="Model directory path"
    )
    create_parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU for voice creation"
    )
    
    # List voices subcommand
    list_parser = voice_subparsers.add_parser("list", help="List all available voices")
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    # Delete voice subcommand
    delete_parser = voice_subparsers.add_parser("delete", help="Delete a voice")
    delete_parser.add_argument(
        "name",
        type=str,
        help="Name of the voice to delete"
    )
    
    # Create TTS command
    tts_parser = subparsers.add_parser("tts", help="Generate speech from text using a voice")
    text_group = tts_parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", 
        type=str,
        help="Text to synthesize"
    )
    text_group.add_argument(
        "--text-file", 
        type=str,
        help="Path to a text file containing content to synthesize"
    )
    
    # Voice source for TTS (optional - use default voice if not specified)
    tts_parser.add_argument(
        "--voice",
        type=str,
        help="Name of a voice to use (created with 'voice create'). If not specified, the default voice will be used."
    )
    
    # Optional TTS parameters
    tts_parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_CONFIG.get("model_dir", "pretrained_models/Spark-TTS-0.5B"),
        help="Model directory path"
    )
    tts_parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG.get("device", "cuda:0"),
        help="Device identifier (e.g., cuda:0 or cpu)"
    )
    tts_parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female"],
        help="Gender (male or female)"
    )
    tts_parser.add_argument(
        "--pitch",
        type=str,
        choices=["very_low", "low", "moderate", "high", "very_high"],
        help="Pitch (very_low, low, moderate, high, very_high)"
    )
    tts_parser.add_argument(
        "--speed",
        type=str,
        choices=["very_low", "low", "moderate", "high", "very_high"],
        help="Speed (very_low, low, moderate, high, very_high)"
    )
    tts_parser.add_argument(
        "--emotion",
        type=str,
        help="Emotion (e.g., NEUTRAL, HAPPY, SAD, ANGRY)"
    )
    tts_parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_CONFIG.get("save_dir", "output"),
        help="Output audio directory"
    )
    tts_parser.add_argument(
        "--segmentation_threshold",
        type=int,
        default=DEFAULT_CONFIG.get("segmentation_threshold", 150),
        help="Maximum words per segment"
    )
    tts_parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU inference"
    )
    tts_parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CONFIG.get("cache_dir", "tts_cache"),
        help="Directory for caching TTS audio segments"
    )
    tts_parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regeneration of audio even if cached"
    )
    
    # Create cache command for management
    cache_parser = subparsers.add_parser("cache", help="Manage TTS segment cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache operation")
    
    # TTS segment cache
    tts_cache_parser = cache_subparsers.add_parser("segments", help="Manage TTS segment audio cache")
    tts_cache_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear TTS segment cache"
    )
    tts_cache_parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CONFIG.get("cache_dir", "tts_cache"),
        help="Directory for TTS segment cache"
    )
    
    # Create text segmentation command
    segment_parser = subparsers.add_parser("segment", help="Segment long text for check text segmentation")
    file_group = segment_parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument(
        "input_file",
        type=str,
        nargs="?",
        help="Path to the text file to segment (positional argument)"
    )
    file_group.add_argument(
        "--file",
        type=str,
        help="Path to the text file to segment"
    )
    segment_parser.add_argument(
        "--output-suffix",
        type=str,
        default="_segmented",
        help="Suffix to add to the output filename"
    )
    segment_parser.add_argument(
        "--max-chinese-chars",
        type=int,
        default=DEFAULT_CONFIG.get("max_chinese_chars", 150),
        help="Maximum Chinese characters per segment"
    )
    segment_parser.add_argument(
        "--max-english-chars",
        type=int,
        default=DEFAULT_CONFIG.get("max_english_chars", 300),
        help="Maximum English characters per segment"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Display configuration information
    if DEFAULT_CONFIG:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default-config.toml")
        if os.path.exists(config_path):
            logging.info(f"Using configuration from: {config_path}")
            logging.info(f"Key config values: model_dir={DEFAULT_CONFIG.get('model_dir')}, "
                        f"device={DEFAULT_CONFIG.get('device')}, "
                        f"save_dir={DEFAULT_CONFIG.get('save_dir')}")
        else:
            logging.warning("Configuration file exists but not at expected path")
    else:
        logging.warning("No configuration loaded, using built-in defaults")
    
    # Parse command-line arguments
    args = parse_args()
    
    # If no command provided, show help
    if not args.command:
        print("Error: Please specify a command (voice, tts, cache, or segment)")
        print("Run 'python . --help' for usage information")
        sys.exit(1)
    
    try:
        # Execute requested command
        if args.command == "voice":
            if not hasattr(args, 'voice_action') or not args.voice_action:
                print("Error: Please specify a voice action (create, list, or delete)")
                print("Run 'python . voice --help' for usage information")
                sys.exit(1)
                
            if args.voice_action == "create":
                # Create a new voice profile
                success = create_voice(
                    name=args.name,
                    reference_path=args.reference,
                    reference_text=args.text,
                    gender=args.gender, 
                    description=args.description,
                    model_dir=args.model_dir,
                    force_cpu=args.force_cpu,
                )
                
                if success:
                    print(f"✅ Voice '{args.name}' created successfully")
                else:
                    print(f"❌ Failed to create voice '{args.name}'")
                    sys.exit(1)
                    
            elif args.voice_action == "list":
                # List all available voices
                voices = list_voices(json_format=args.json)
                if not voices:
                    print("No voices found")
                
            elif args.voice_action == "delete":
                # Delete a voice
                if delete_voice(args.name):
                    print(f"✅ Voice '{args.name}' deleted successfully")
                else:
                    print(f"❌ Failed to delete voice '{args.name}' or voice not found")
                    sys.exit(1)
        
        elif args.command == "tts":
            # Process text from file if specified
            if args.text_file:
                if not os.path.exists(args.text_file):
                    raise FileNotFoundError(f"Text file not found: {args.text_file}")
                text = read_text_from_file(args.text_file)
                logging.info(f"Read text content from file: {args.text_file}")
            else:
                text = args.text
            
            # Determine voice source (either named voice or reference audio)
            voice_tokens = None
            if args.voice:
                # Get voice data from saved voice profile
                voice_path = get_voice_path(args.voice)
                if not voice_path:
                    print(f"❌ Voice '{args.voice}' not found")
                    sys.exit(1)
            
            # Generate TTS audio with specified parameters
            device = select_device(args.device, args.force_cpu)
            output_file = generate_tts_audio(
                text=text,
                model_dir=args.model_dir,
                device=device,
                voice_name=args.voice if args.voice else None,
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
            
            print("✅ Generated audio saved to:", output_file)
            
        elif args.command == "cache":
            # Manage cache
            if not hasattr(args, 'cache_command') or not args.cache_command:
                print("Please specify a cache operation (segments)")
                print("Run 'python . cache --help' for usage information")
                sys.exit(1)
                
            if args.cache_command == "segments":
                if args.clear:
                    # Import here to avoid circular imports
                    from sparktts.cache import TTSCache
                    cache = TTSCache(args.cache_dir)
                    if cache.clear_cache():
                        print(f"✅ TTS segments cache cleared successfully from {args.cache_dir}")
                    else:
                        print("❌ Failed to clear TTS segments cache")
                else:
                    print("No cache action specified. Use --clear to clear the cache.")
                    
        elif args.command == "segment":
            # Determine file path
            file_path = args.input_file if args.input_file else args.file
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Text file not found: {file_path}")
                
            # Execute text segmentation
            output_file = segment_text_file(
                file_path,
                output_suffix=args.output_suffix,
                max_chinese_chars=args.max_chinese_chars,
                max_english_chars=args.max_english_chars
            )
            
            if output_file:
                print(f"✅ Text segmented successfully. Output saved to: {output_file}")
                print(f"   Total segments: {len(open(output_file, 'r', encoding='utf-8').read().split('---'))}")
            else:
                print("❌ Failed to segment text")
                
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 