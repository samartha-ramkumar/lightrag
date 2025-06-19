import numpy as np
from typing import Any, Union, Optional
from collections.abc import AsyncIterator

from openai import (
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
    get_env_value
)



class AzureOpenAIProvider:
    """Azure OpenAI provider class for handling Azure OpenAI API interactions."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, api_version: str | None = None):
        self.api_key = get_env_value("AZURE_OPENAI_API_KEY", api_key)
        self.base_url = get_env_value("AZURE_OPENAI_ENDPOINT", base_url)
        self.api_version = get_env_value("AZURE_OPENAI_API_VERSION", api_version)
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key must be provided or set in AZURE_OPENAI_API_KEY environment variable.")
        if not self.base_url:
            raise ValueError("Azure OpenAI endpoint must be provided or set in AZURE_OPENAI_ENDPOINT environment variable.")

    def create_azure_openai_async_client(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
    ) -> AsyncAzureOpenAI:
        """Create an AsyncAzureOpenAI client with the given configuration."""
        api_key = api_key or self.api_key
        base_url = base_url or self.base_url
        api_version = api_version or self.api_version

        return AsyncAzureOpenAI(
            azure_endpoint=base_url,
            azure_deployment=model,
            api_key=api_key,
            api_version=api_version,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APIConnectionError)
        ),
    )
    async def azure_openai_complete_if_cache(
        self,
        model,
        prompt,
        system_prompt=None,
        history_messages=[],
        base_url=None,
        api_key=None,
        api_version=None,
        **kwargs,
    ):

        openai_async_client = self.create_azure_openai_async_client(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
        )
        
        kwargs.pop("hashing_kv", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

        if hasattr(response, "__aiter__"):

            async def inner():
                async for chunk in response:
                    if len(chunk.choices) == 0:
                        continue
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))
                    yield content

            return inner()
        else:
            content = response.choices[0].message.content
            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))
            return content

    async def generate_text(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text using Azure OpenAI API."""
        if history_messages is None:
            history_messages = []
        
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = "json"
            
        # Get model from hashing_kv or use default
        model_name = get_env_value("LLM_MODEL", "gpt-4o-mini")
        if "hashing_kv" in kwargs and hasattr(kwargs["hashing_kv"], "global_config"):
            model_name = kwargs["hashing_kv"].global_config.get("llm_model_name", model_name)
        
        result = await self.azure_openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        
        if keyword_extraction:  # TODO: use JSON API
            return locate_json_string_body_from_string(result)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        api_version: str = None,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings using Azure OpenAI API."""
        if model is None:
            model = get_env_value("EMBEDDING_MODEL", "text-embedding-3-small")

        openai_async_client = self.create_azure_openai_async_client(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
        )

        response = await openai_async_client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

    async def analyze_image(
        self, 
        image_data: str, 
        prompt: str, 
        model: str = "gpt-4o",
    ) -> str:
        """Analyze image with Azure OpenAI's vision model"""
        openai_async_client = self.create_azure_openai_async_client(model=model)
        
        try:
            response = await openai_async_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Azure OpenAI vision API error: {str(e)}")
            return f"Error analyzing image: {str(e)}"

    async def cleanup(self) -> None:
        """Clean up resources"""
        pass
