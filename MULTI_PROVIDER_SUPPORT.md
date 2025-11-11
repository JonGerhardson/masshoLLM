# Multi-Provider Support for report_extras

This document describes the multi-provider support feature added to the `report_extras` module, which allows the system to work with different LLM providers while maintaining backward compatibility with the existing Gemini setup.

## Overview

The `report_extras` module has been enhanced to support multiple LLM providers:
- **Gemini** (default, existing provider)
- **OpenRouter** 
- **LMStudio**

The implementation follows a factory pattern that allows easy switching between providers via configuration.

## Configuration

To change the LLM provider, update the `config.yaml` file:

```yaml
llm_settings:
  # Choose your provider: "gemini", "openrouter", or "lmstudio"
  provider: "gemini"  # Default provider

  # Model names (provider-specific)
  flash_model: 'gemini-2.0-flash'  # For initial processing
  pro_model: 'gemini-2.5-pro'      # For final processing

  # Rate limits (requests per minute)
  flash_rpm: 15    # flash model RPM
  pro_rpm: 2       # pro model RPM

  # API Keys
  api_keys:
    gemini: "your-gemini-api-key"
    openrouter: "your-openrouter-api-key"
  
  # Endpoints (for OpenRouter/LMStudio)
  endpoints:
    openrouter: "https://openrouter.ai/api/v1/chat/completions"
    lmstudio: "http://localhost:1234/v1/chat/completions"
    
  # Model selections per provider
  models:
    gemini: 'gemini-2.0-flash'
    openrouter: 'openrouter/andromeda-alpha'
    lmstudio: 'liquid/lfm2-1.2b'
```

## Provider-Specific Setup

### Gemini (Default)
- Requires `GEMINI_API_KEY` environment variable or config setting
- Uses Google's Generative AI SDK
- Default models: `gemini-2.0-flash` (fast) and `gemini-2.5-pro` (accurate)

### OpenRouter
- Requires `OPENROUTER_API_KEY` environment variable or config setting
- Compatible with OpenRouter's API
- Uses OpenAI-compatible API format

### LMStudio
- Connects to local LMStudio instance at `http://localhost:1234`
- No API key typically required for local instance
- Default endpoint: `http://localhost:1234/v1/chat/completions`

## Architecture

The multi-provider system consists of:

1. **BaseLLMProvider** - Abstract base class defining the interface
2. **Provider Implementations**:
   - `GeminiProvider` - For Google Gemini models
   - `OpenRouterProvider` - For OpenRouter API
   - `LMStudioProvider` - For local LMStudio API
3. **Provider Factory** - Centralized factory function to instantiate providers
4. **BriefingConfig** - Updated to handle provider-specific settings

### Key Files

- `base_provider.py` - Abstract base class definition
- `llm_provider.py` - Gemini provider implementation
- `openrouter_provider.py` - OpenRouter provider implementation  
- `lmstudio_provider.py` - LMStudio provider implementation
- `provider_factory.py` - Factory function
- `report_extras.py` - Main entry point using the factory

## Usage

The `report_extras` command works the same as before:

```bash
python report_extras/report_extras.py 2025-10-28
```

The selected provider will be used automatically based on the configuration.

## Testing

A basic test utility is provided in `test_providers.py`:

```bash
python report_extras/test_providers.py
```

This runs a simple API call with the configured provider to verify it works.

## Backward Compatibility

- Default provider is still Gemini with the same models
- All existing configuration remains valid
- Changing providers is optional - no action required for existing users
- Original functionality is preserved

## Adding New Providers

To add a new provider, implement the `BaseLLMProvider` interface:

1. Create a new provider class inheriting from `BaseLLMProvider`
2. Implement all abstract methods
3. Add the provider to the factory in `provider_factory.py`
4. Update the config structure if needed

## Troubleshooting

### Common Issues:

1. **"Provider not found"** - Check the `provider` value in config.yaml
2. **API Key errors** - Verify the correct API key is set in environment or config
3. **Import errors** - Ensure all provider files are in the report_extras directory

### Verification:

Run the quick test to verify provider selection:
```bash
cd report_extras && python quick_test.py
```