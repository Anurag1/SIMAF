# Unified LLM Provider for Fractal Agent Ecosystem

## Overview

The `UnifiedLM` class provides a unified interface for all LLM calls in the Fractal Agent Ecosystem with automatic failover between providers.

**Architecture:**

- Primary Provider: **Anthropic Claude** (via Claude Code MAX subscription)
- Fallback Provider: **Google Gemini** (via API key)
- Extensible: Easy to add new providers

## Key Features

✓ **Subscription-Based Authentication**: Uses Claude Code MAX subscription (NO API KEY required)
✓ **Automatic Failover**: If primary provider fails, automatically tries next provider in chain
✓ **Retry Logic**: Built-in exponential backoff for transient errors
✓ **Provider Abstraction**: Unified interface regardless of underlying provider
✓ **Usage Metrics**: Track calls, tokens, cache hits across all providers
✓ **Extensible**: Add new providers by implementing `LLMProvider` base class

## Authentication Setup

### Anthropic Claude (Primary)

The Anthropic provider uses the `claude-agent-sdk` which leverages your Claude Code CLI subscription.

**NO API KEY NEEDED** - Authentication is automatic via Claude Code CLI.

```bash
# DO NOT set ANTHROPIC_API_KEY in .env
# The SDK uses your Claude Code subscription automatically
```

### Google Gemini (Fallback)

Set your Google API key in `.env`:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Basic Usage

```python
from fractal_agent.utils.llm_provider import UnifiedLM

# Create LLM instance (uses default provider chain)
lm = UnifiedLM()

# Simple query
response = lm(
    prompt="What is the Viable System Model?",
    max_tokens=200
)

print(response['text'])        # The response text
print(response['provider'])    # Which provider was used
print(response['model'])       # Which model was used
print(response['tokens_used']) # Token usage
```

### Advanced Usage with Messages

```python
# Multi-turn conversation with system message
response = lm(
    messages=[
        {"role": "system", "content": "You are an expert in cybernetics."},
        {"role": "user", "content": "Explain VSM in one sentence."},
        {"role": "assistant", "content": "The Viable System Model..."},
        {"role": "user", "content": "What are the 5 systems?"}
    ],
    max_tokens=500,
    temperature=0.7
)
```

### Custom Provider Chain

```python
# Anthropic only (no fallback)
lm = UnifiedLM(
    providers=[
        ("anthropic", "claude-sonnet-4.5")
    ]
)

# Gemini only
lm = UnifiedLM(
    providers=[
        ("gemini", "gemini-2.0-flash-exp")
    ]
)

# Custom chain with multiple fallbacks
lm = UnifiedLM(
    providers=[
        ("anthropic", "claude-sonnet-4.5"),
        ("gemini", "gemini-2.0-flash-exp"),
        # Future: add more providers
    ]
)
```

### Usage Metrics

```python
# Get comprehensive metrics
metrics = lm.get_metrics()

print(f"Total calls: {metrics['total_calls']}")
print(f"Total failures: {metrics['total_failures']}")
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")

# Per-provider metrics
for provider in metrics['per_provider']:
    print(f"{provider['provider']}: {provider['calls']} calls, {provider['tokens_used']} tokens")
```

## Provider Implementation Details

### AnthropicProvider

- **Authentication**: Claude Code subscription via `claude-agent-sdk`
- **Models**: claude-sonnet-4.5, claude-opus-4.1, claude-haiku-4.5, etc.
- **Features**: Prompt caching, thinking blocks, function calling
- **Async**: Uses async implementation internally with sync wrapper

**Key Implementation:**

```python
from claude_agent_sdk import query as claude_query
from claude_agent_sdk.types import ClaudeAgentOptions

# Automatically uses Claude Code CLI authentication
async for message in claude_query(prompt=prompt, options=options):
    # Process messages...
```

### GeminiProvider

- **Authentication**: API key from environment variable
- **Models**: gemini-2.0-flash-exp, gemini-1.5-pro, etc.
- **Parameter Mapping**: Automatically converts Anthropic-style params to Gemini format

**Parameter Conversions:**

- `max_tokens` → `generation_config.max_output_tokens`
- `temperature` → `generation_config.temperature`
- `top_p` → `generation_config.top_p`

## Testing

### Run Built-in Tests

```bash
# Test basic functionality
python -m fractal_agent.utils.llm_provider

# Test both providers independently
python test_llm_simple.py

# Test failover mechanism
python test_failover.py
```

### Test Results

```
✓ Test 1: Anthropic Primary Provider
  - Model: claude-sonnet-4.5
  - Response: ✓ Success
  - Tokens: ~1400 tokens

✓ Test 2: Gemini Fallback Provider
  - Model: gemini-2.0-flash-exp
  - Response: ✓ Success
  - Tokens: ~100 tokens (estimated)

✓ Failover Architecture: VERIFIED
  - Provider chain works correctly
  - Retry logic with exponential backoff
  - Automatic parameter conversion
```

## Adding New Providers

To add a new LLM provider:

### 1. Implement the Provider Class

```python
class NewProvider(LLMProvider):
    def __init__(self, model: str, **config):
        super().__init__(model, **config)
        # Initialize client

    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        # Implement provider-specific logic

        return {
            "text": response_text,
            "tokens_used": token_count,
            "cache_hit": False,
            "provider": "new_provider",
            "model": self.model
        }
```

### 2. Register in Provider Registry

```python
PROVIDER_REGISTRY = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "new_provider": NewProvider,  # Add here
}
```

### 3. Use in Provider Chain

```python
lm = UnifiedLM(
    providers=[
        ("anthropic", "claude-sonnet-4.5"),
        ("new_provider", "model-name"),
        ("gemini", "gemini-2.0-flash-exp")
    ]
)
```

## Configuration

### Environment Variables

```bash
# Anthropic - DO NOT SET (uses subscription)
# ANTHROPIC_API_KEY=  # Leave empty

# Google Gemini - Required for fallback
GOOGLE_API_KEY=your_api_key_here
```

### Provider-Specific Config

```python
lm = UnifiedLM(
    providers=[("anthropic", "claude-sonnet-4.5")],
    enable_caching=True,  # Enable prompt caching (Anthropic)
    anthropic={
        "cwd": "/path/to/workspace"  # Working directory for SDK
    },
    gemini={
        # Future: add Gemini-specific config
    }
)
```

## Error Handling

The system includes robust error handling with automatic failover:

```python
try:
    response = lm(prompt="Your question")
except Exception as e:
    # Only raised if ALL providers in chain fail
    print(f"All providers failed: {e}")
```

Errors are logged with detailed information:

```
INFO: Trying provider 1/2: claude-sonnet-4.5 (AnthropicProvider)
WARNING: Provider failed: claude-sonnet-4.5: NetworkError
INFO: Trying next provider in chain...
INFO: Trying provider 2/2: gemini-2.0-flash-exp (GeminiProvider)
INFO: Provider succeeded: gemini-2.0-flash-exp (cache_hit=False)
```

## Best Practices

1. **Use Default Provider Chain**: The default Anthropic → Gemini chain is optimized for reliability
2. **Monitor Metrics**: Regularly check metrics to understand usage patterns
3. **Set Appropriate max_tokens**: Prevents runaway costs
4. **Use System Messages**: For better prompt caching (Anthropic)
5. **Handle Exceptions**: Always wrap LLM calls in try/except

## Future Enhancements

- [ ] Add OpenAI provider
- [ ] Add Together.ai provider
- [ ] Implement streaming responses
- [ ] Add cost tracking and budgets
- [ ] Implement prompt caching metrics for Anthropic
- [ ] Add provider health monitoring
- [ ] Implement circuit breaker pattern

## Troubleshooting

### Anthropic Authentication Issues

```
Error: Could not resolve authentication method
```

**Solution**: Ensure you're NOT setting `ANTHROPIC_API_KEY` in .env. The provider should use Claude Code subscription automatically.

### Gemini API Errors

```
Error: DefaultCredentialsError
```

**Solution**: Verify `GOOGLE_API_KEY` is set in `.env` file and `load_dotenv()` is called.

### Parameter Errors

```
TypeError: got an unexpected keyword argument 'max_tokens'
```

**Solution**: The GeminiProvider automatically converts parameters. If you see this error, it means the conversion logic needs updating for new parameters.

## License

Part of the BMAD Fractal Agent Ecosystem
Author: BMad
Date: 2025-10-18
