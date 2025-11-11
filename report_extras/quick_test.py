import yaml
from llm_provider import BriefingConfig
from provider_factory import get_provider

def quick_test():
    # Load config
    import os
    config_path = os.path.join("..", "config.yaml")
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    config = BriefingConfig(full_config)
    print(f"Configured provider: {config.provider_name}")
    
    # Test that the provider is loaded correctly based on config
    prompts = {"test": "test prompt"}
    provider = get_provider(config, prompts)
    
    print(f"Loaded provider type: {type(provider).__name__}")
    print(f"Flash model: {config.flash_model_name}")
    print(f"Pro model: {config.pro_model_name}")
    
    # Test to see if it's the right provider
    if config.provider_name == "gemini":
        assert hasattr(provider, "pro_model"), "Should be a Gemini provider"
        print("✓ Gemini provider loaded correctly")
    elif config.provider_name == "openrouter":
        assert hasattr(provider, "base_url"), "Should be an OpenRouter provider"
        print("✓ OpenRouter provider loaded correctly")
    elif config.provider_name == "lmstudio":
        assert hasattr(provider, "base_url"), "Should be an LMStudio provider"
        print("✓ LMStudio provider loaded correctly")

if __name__ == "__main__":
    quick_test()