import os
import json
import tomli
import torch
import platform
import logging
import hashlib
from datetime import datetime
from pathlib import Path
import sys


# Look for default config in multiple locations
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "default-config.toml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Load configuration file, return empty dict if file doesn't exist
    """
    try:
        # Convert to absolute path and check file existence 
        abs_path = Path(config_path).resolve()
        if not abs_path.exists():
            logging.warning(f"Config file not found at {abs_path}")
            
            # Try alternate path directly in project root
            alt_path = Path(os.path.dirname(os.path.dirname(__file__))) / "default-config.toml"
            if alt_path.exists():
                abs_path = alt_path
                logging.info(f"Found config file at alternate location: {abs_path}")
            else:
                logging.warning(f"Config file not found at alternate location: {alt_path}")
                return {}
        
        logging.info(f"Loading config from {abs_path}")
        with open(abs_path, 'rb') as f:
            config = tomli.load(f)
            logging.info(f"Config loaded successfully with {len(config)} keys")
            return config
    except tomli.TOMLDecodeError as e:
        logging.error(f"TOML parsing error in config file {config_path}: {str(e)}")
        return {}
    except Exception as e:
        logging.warning(f"Could not load config file {config_path}: {str(e)}")
        return {}

# Load default configuration
DEFAULT_CONFIG = load_config()
logging.info(f"Loaded default configuration with {len(DEFAULT_CONFIG)} entries")

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
                
            device = torch.device("mps")
            logging.info("Running on macOS - using MPS for GPU acceleration")
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
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")
        
    return device

def get_timestamp_filename(extension="wav"):
    """Generate unique filename using timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}.{extension}"

def ensure_dir_exists(directory):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_audio_hash(file_path):
    """Generate a unique hash for an audio file"""
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        logging.error(f"Error calculating audio hash: {str(e)}")
        return None

def get_default_value(user_value, config_key, default_value):
    """Get a value prioritizing user input, then config, then default"""
    if user_value is not None:
        return user_value
    return DEFAULT_CONFIG.get(config_key, default_value)

def setup_logging(level=logging.INFO, force=True):
    """
    配置统一的日志系统
    
    Args:
        level: 日志级别，默认为INFO
        force: 是否强制重新配置日志处理器
        
    Returns:
        根日志记录器
    """
    # 基本配置
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=force
    )
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 移除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加标准错误流处理器
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logging.debug("Logging system initialized")
    return root_logger 