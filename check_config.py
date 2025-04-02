#!/usr/bin/env python3
"""
Configuration validation utility for Spark-TTS
"""

import os
import sys
import logging
from pathlib import Path
import tomli
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def check_config_file():
    """Check if the configuration file exists and can be loaded"""
    # Get file locations
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    
    # Check standard locations
    config_locations = [
        project_root / "default-config.toml",  # Root config
        project_root / "cli" / "default-config.toml",  # CLI module config
    ]
    
    found_configs = []
    for loc in config_locations:
        if loc.exists():
            found_configs.append(loc)
            
    if not found_configs:
        logging.error("Configuration file not found")
        return False
    
    # Try to load each found config
    for config_path in found_configs:
        try:
            with open(config_path, 'rb') as f:
                config = tomli.load(f)
                
            # Check minimal required keys
            required_keys = ["model_dir", "save_dir"]
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                logging.error(f"Config at {config_path} is missing required keys: {missing_keys}")
                continue
                
            logging.info(f"Found valid config at: {config_path}")
            print(f"Configuration values:")
            for key in sorted(config.keys()):
                if isinstance(config[key], str) and len(config[key]) > 0:
                    print(f"  - {key}: {config[key]}")
            
            # Check if model_dir exists
            model_dir = Path(config["model_dir"])
            if not model_dir.exists():
                logging.warning(f"Model directory does not exist: {model_dir}")
                print(f"Warning: Model directory not found at {model_dir}")
            
            return True
        except tomli.TOMLDecodeError as e:
            logging.error(f"Error parsing {config_path}: {str(e)}")
        except Exception as e:
            logging.error(f"Error loading {config_path}: {str(e)}")
            
    return False

def main():
    """Main validation function"""
    print("Spark-TTS Configuration Check")
    print("=============================")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU devices: {torch.cuda.device_count()}")
        
    # Check config
    print("\nChecking configuration...")
    config_ok = check_config_file()
    
    if not config_ok:
        print("\nConfiguration issues detected:")
        print("- Make sure 'default-config.toml' is in the project root directory.")
        print("- The configuration must be valid TOML syntax")
        print("- Required keys: model_dir, save_dir")
        print("\nTroubleshooting:")
        print("1. Make sure default-config.toml is in the correct location")
        print("2. Verify TOML syntax with a validator")
        print("3. Try adding a symlink from cli/default-config.toml to the root config file:")
        print("   ln -s ../default-config.toml cli/default-config.toml")
    else:
        print("\nConfiguration check completed successfully.")
        
if __name__ == "__main__":
    main() 