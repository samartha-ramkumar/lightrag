import sys

from collections.abc import AsyncIterator
from lightrag.utils import get_env_value
import ollama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from lightrag.llm.base import BaseLLMProvider
import numpy as np
from typing import Union, Any, Optional, List
from lightrag.utils import logger


class OllamaProvider(BaseLLMProvider):
    """Ollama provider class for handling Ollama API interactions."""

    def __init__(self, host: str = None, api_key: str = None):
        self.host = get_env_value("OLLAMA_HOST", host or "http://localhost:11434")
        self.api_key = get_env_value("OLLAMA_API_KEY", api_key)
        self.timeout = 300  # Default timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def _ollama_model_if_cache(
        self,
        model,
        prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        stream = True if kwargs.get("stream") else False

        kwargs.pop("max_tokens", None)
        host = kwargs.pop("host", None) or self.host
        timeout = kwargs.pop("timeout", None) or self.timeout
        kwargs.pop("hashing_kv", None)
        api_key = kwargs.pop("api_key", None) or self.api_key
        
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            response = await ollama_client.chat(model=model, messages=messages, **kwargs)
            if stream:
                """cannot cache stream response and process reasoning"""

                async def inner():
                    try:
                        async for chunk in response:
                            yield chunk["message"]["content"]
                    except Exception as e:
                        logger.error(f"Error in stream response: {str(e)}")
                        raise
                    finally:
                        try:
                            await ollama_client._client.aclose()
                            logger.debug("Successfully closed Ollama client for streaming")
                        except Exception as close_error:
                            logger.warning(f"Failed to close Ollama client: {close_error}")

                return inner()
            else:
                model_response = response["message"]["content"]
                return model_response
        except Exception as e:
            try:
                await ollama_client._client.aclose()
                logger.debug("Successfully closed Ollama client after exception")
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client after exception: {close_error}"
                )
            raise e
        finally:
            if not stream:
                try:
                    await ollama_client._client.aclose()
                    logger.debug(
                        "Successfully closed Ollama client for non-streaming response"
                    )
                except Exception as close_error:
                    logger.warning(
                        f"Failed to close Ollama client in finally block: {close_error}"
                    )

    async def generate_text(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text using Ollama API."""
        if history_messages is None:
            history_messages = []
            
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["format"] = "json"
            
        # Get model from hashing_kv or use default
        model_name = get_env_value("LLM_MODEL", "llama2")
        if "hashing_kv" in kwargs and hasattr(kwargs["hashing_kv"], "global_config"):
            model_name = kwargs["hashing_kv"].global_config.get("llm_model_name", model_name)
            
        return await self._ollama_model_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "nomic-embed-text",
        base_url: str = None,
        api_key: str = None,
        client_configs: dict[str, Any] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings using Ollama API."""
        api_key = api_key or self.api_key
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        host = kwargs.pop("host", None) or self.host
        timeout = kwargs.pop("timeout", None) or 90  # Default timeout 90s

        ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

        try:
            data = await ollama_client.embed(model=model, input=texts)
            return np.array(data["embeddings"])
        except Exception as e:
            logger.error(f"Error in ollama_embed: {str(e)}")
            try:
                await ollama_client._client.aclose()
                logger.debug("Successfully closed Ollama client after exception in embed")
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client after exception in embed: {close_error}"
                )
            raise e
        finally:
            try:
                await ollama_client._client.aclose()
                logger.debug("Successfully closed Ollama client after embed")
            except Exception as close_error:
                logger.warning(f"Failed to close Ollama client after embed: {close_error}")

    async def analyze_image(
        self, 
        image_data: str, 
        prompt: str, 
        model: str = "llava",
    ) -> str:
        """Analyze image with Ollama's vision model"""
        try:
            # Create message with text and image content
            message = {
                "role": "user", 
                "content": prompt,
                "images": [image_data]  # Ollama expects base64 image data in images array
            }
            
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            ollama_client = ollama.AsyncClient(host=self.host, timeout=self.timeout, headers=headers)
            
            try:
                response = await ollama_client.chat(
                    model=model,
                    messages=[message]
                )
                return response["message"]["content"]
            finally:
                try:
                    await ollama_client._client.aclose()
                    logger.debug("Successfully closed Ollama client after image analysis")
                except Exception as close_error:
                    logger.warning(f"Failed to close Ollama client after image analysis: {close_error}")
                    
        except Exception as e:
            logger.error(f"Ollama vision API error: {str(e)}")
            return f"Error analyzing image: {str(e)}"

    async def cleanup(self) -> None:
        """Clean up resources"""
        pass
