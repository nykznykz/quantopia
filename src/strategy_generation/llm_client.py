"""LLM client for strategy generation and code generation."""

import os
from typing import Dict, List, Optional, Any
import json
import logging

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError(
        "OpenAI library not found. Install with: pip install openai"
    )

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with OpenAI GPT-4 API."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """Initialize LLM client.

        Args:
            provider: LLM provider ('openai', 'azure', 'deepseek', 'anthropic')
            api_key: API key (if None, reads from environment)
            model: Model name (default: 'gpt-4')
                   Recommended for code generation:
                   - 'gpt-4' (OpenAI)
                   - 'claude-sonnet-4' (Anthropic) - best for code
                   - 'deepseek-coder' (DeepSeek)
            temperature: Sampling temperature (default: 0.7)
                        Use 0.2-0.3 for code generation (more deterministic)
            max_tokens: Maximum tokens in response (default: 2000)
                       Use 4000+ for code generation
            azure_endpoint: Azure OpenAI endpoint (for Azure provider)
            api_version: Azure API version (for Azure provider)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize client based on provider
        if provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            self.client = OpenAI(api_key=self.api_key)

        elif provider == "azure":
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
            self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

            if not self.api_key or not self.azure_endpoint:
                raise ValueError(
                    "Azure OpenAI credentials not found. Set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY) and "
                    "AZURE_OPENAI_ENDPOINT environment variables."
                )

            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )

        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "Anthropic library not found. Install with: pip install anthropic"
                )
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
                )
            self.client = anthropic.Anthropic(api_key=self.api_key)

        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable."
                )
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                "Supported providers: 'openai', 'azure', 'anthropic', 'deepseek'"
            )

        logger.info(
            f"Initialized LLMClient: provider={provider}, model={model}, "
            f"temperature={temperature}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None
    ) -> str:
        """Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override default temperature (optional)
            max_tokens: Override default max_tokens (optional)
            response_format: Response format ('json' or None)

        Returns:
            Generated text
        """
        try:
            # Handle Anthropic API separately
            if self.provider == "anthropic":
                request_params = {
                    "model": self.model,
                    "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }

                if system_prompt:
                    request_params["system"] = system_prompt

                response = self.client.messages.create(**request_params)
                generated_text = response.content[0].text

            # Handle OpenAI-compatible APIs (OpenAI, Azure, DeepSeek)
            else:
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                }

                # Add response format if specified (for structured output)
                if response_format == "json":
                    request_params["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**request_params)
                generated_text = response.choices[0].message.content

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate JSON response.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override default temperature (optional)
            max_tokens: Override default max_tokens (optional)

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If response is not valid JSON
        """
        # Request JSON format
        response_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json"
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """Generate multiple completions (sequential).

        Args:
            prompts: List of user prompts
            system_prompt: System prompt (same for all)
            temperature: Override default temperature (optional)
            max_tokens: Override default max_tokens (optional)

        Returns:
            List of generated texts
        """
        responses = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Generating batch {i+1}/{len(prompts)}")

            try:
                response = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in batch {i+1}: {e}")
                responses.append(None)

        return responses


def get_recommended_code_models() -> Dict[str, Dict[str, str]]:
    """Get recommended models for code generation by provider.
    
    Returns:
        Dict mapping provider to model configuration
    """
    return {
        "openai": {
            "model": "gpt-4",
            "temperature": "0.2",
            "max_tokens": "4000",
            "description": "Good for code, widely available"
        },
        "anthropic": {
            "model": "claude-sonnet-4",
            "temperature": "0.2",
            "max_tokens": "4096",
            "description": "Excellent for code generation, strong reasoning"
        },
        "deepseek": {
            "model": "deepseek-coder",
            "temperature": "0.2",
            "max_tokens": "4000",
            "description": "Code-specialized model, cost-effective"
        }
    }


def create_code_generation_client(provider: str = "openai") -> LLMClient:
    """Create LLM client optimized for code generation.
    
    Args:
        provider: LLM provider name
        
    Returns:
        Configured LLMClient instance
    """
    recommended = get_recommended_code_models()
    
    if provider not in recommended:
        logger.warning(f"No recommendation for provider '{provider}', using defaults")
        return LLMClient(provider=provider, temperature=0.2, max_tokens=4000)
    
    config = recommended[provider]
    logger.info(f"Creating code generation client: {config['description']}")
    
    return LLMClient(
        provider=provider,
        model=config["model"],
        temperature=float(config["temperature"]),
        max_tokens=int(config["max_tokens"])
    )
