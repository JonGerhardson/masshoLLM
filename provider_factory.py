from typing import Dict, Any
from base_provider import BaseLLMProvider

def get_provider(config, prompts: Dict[str, Any]) -> BaseLLMProvider:
    """
    Factory function to instantiate the appropriate LLM provider based on configuration.
    
    Args:
        config: BriefingConfig object containing provider settings
        prompts: Dictionary of prompt templates
        
    Returns:
        An instance of a class that inherits from BaseLLMProvider
    """
    provider_name = config.provider_name.lower()
    
    if provider_name == 'gemini':
        from llm_provider import GeminiProvider
        return GeminiProvider(config, prompts)
    elif provider_name == 'openrouter':
        from openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(config, prompts)
    elif provider_name == 'lmstudio':
        from lmstudio_provider import LMStudioProvider
        return LMStudioProvider(config, prompts)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}. Supported: gemini, openrouter, lmstudio")