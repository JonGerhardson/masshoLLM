import os
import yaml
from llm_provider import BriefingConfig
from provider_factory import get_provider

def test_gemini_provider():
    """Test function to verify Gemini provider works with a simple API call"""
    import sys
    from pathlib import Path
    
    print("Testing Gemini provider...")
    
    # Load config from project root
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    
    # Load config
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    config = BriefingConfig(full_config)
    
    # Only test if provider is gemini
    if config.provider_name != 'gemini':
        print("Configured provider is not Gemini, skipping test.")
        return
        
    # Create a simple prompt for testing
    prompts = {"test": "Say 'Hello, World!' in a single sentence."}
    
    try:
        provider = get_provider(config, prompts)
        
        # Make a simple test call
        response = provider._call_llm_with_backoff("Say 'Hello, World!' in a single sentence.", config.flash_model_name)
        if response:
            print(f"Gemini test successful: {response}")
        else:
            print("Gemini test failed: No response received")
    except Exception as e:
        print(f"Gemini test failed with error: {e}")

def test_openrouter_provider():
    """Test function to verify OpenRouter provider works with a simple API call"""
    import sys
    from pathlib import Path
    
    print("Testing OpenRouter provider...")
    
    # Load config from project root
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    
    # Load config
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    config = BriefingConfig(full_config)
    
    # Only test if provider is openrouter
    if config.provider_name != 'openrouter':
        print("Configured provider is not OpenRouter, skipping test.")
        return
        
    # Create a simple prompt for testing
    prompts = {"test": "Say 'Hello, World!' in a single sentence."}
    
    try:
        provider = get_provider(config, prompts)
        
        # Make a simple test call
        response = provider._call_llm_with_backoff("Say 'Hello, World!' in a single sentence.", config.flash_model_name)
        if response:
            print(f"OpenRouter test successful: {response}")
        else:
            print("OpenRouter test failed: No response received")
    except Exception as e:
        print(f"OpenRouter test failed with error: {e}")

def test_lmstudio_provider():
    """Test function to verify LMStudio provider works with a simple API call"""
    import sys
    from pathlib import Path
    
    print("Testing LMStudio provider...")
    
    # Load config from project root
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    
    # Load config
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    config = BriefingConfig(full_config)
    
    # Only test if provider is lmstudio
    if config.provider_name != 'lmstudio':
        print("Configured provider is not LMStudio, skipping test.")
        return
        
    # Create a simple prompt for testing
    prompts = {"test": "Say 'Hello, World!' in a single sentence."}
    
    try:
        provider = get_provider(config, prompts)
        
        # Make a simple test call
        response = provider._call_llm_with_backoff("Say 'Hello, World!' in a single sentence.", config.flash_model_name)
        if response:
            print(f"LMStudio test successful: {response}")
        else:
            print("LMStudio test failed: No response received")
    except Exception as e:
        print(f"LMStudio test failed with error: {e}")

def run_provider_tests():
    """Run tests for the configured provider"""
    import sys
    from pathlib import Path
    
    print("Running provider tests...")
    
    # Load config from project root
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    
    # Load config to determine which provider is configured
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    config = BriefingConfig(full_config)
    
    provider_name = config.provider_name
    print(f"Current provider configured: {provider_name}")
    
    if provider_name == 'gemini':
        test_gemini_provider()
    elif provider_name == 'openrouter':
        test_openrouter_provider()
    elif provider_name == 'lmstudio':
        test_lmstudio_provider()
    else:
        print(f"Unknown provider: {provider_name}")

if __name__ == "__main__":
    run_provider_tests()