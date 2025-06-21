from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

from openai import (
    AsyncOpenAI,
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
)
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.llm.base import BaseLLMProvider
import numpy as np
from typing import Any, Union, Optional

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider class for handling OpenAI API interactions."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment variables.")
        
    def create_openai_async_client(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client_configs: dict[str, Any] = None,
    ) -> AsyncOpenAI:
        """Create an AsyncOpenAI client with the given configuration.

        Args:
            api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
            base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
            client_configs: Additional configuration options for the AsyncOpenAI client.
                These will override any default configurations but will be overridden by
                explicit parameters (api_key, base_url).

        Returns:
            An AsyncOpenAI client instance.
        """
        if not api_key:
            api_key = self.api_key

        default_headers = {
            "Content-Type": "application/json",
        }

        if client_configs is None:
            client_configs = {}

        # Create a merged config dict with precedence: explicit params > client_configs > defaults
        merged_configs = {
            **client_configs,
            "default_headers": default_headers,
            "api_key": api_key,
        }

        if base_url is not None:
            merged_configs["base_url"] = base_url
        else:
            merged_configs["base_url"] = self.base_url

        return AsyncOpenAI(**merged_configs)


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(RateLimitError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(APITimeoutError)
            | retry_if_exception_type(InvalidResponseError)
        ),
    )
    async def openai_complete_if_cache(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        token_tracker: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """Complete a prompt using OpenAI's API with caching support.

        Args:
            model: The OpenAI model to use.
            prompt: The prompt to complete.
            system_prompt: Optional system prompt to include.
            history_messages: Optional list of previous messages in the conversation.
            base_url: Optional base URL for the OpenAI API.
            api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.
                Special kwargs:
                - openai_client_configs: Dict of configuration options for the AsyncOpenAI client.
                    These will be passed to the client constructor but will be overridden by
                    explicit parameters (api_key, base_url).
                - hashing_kv: Will be removed from kwargs before passing to OpenAI.
                - keyword_extraction: Will be removed from kwargs before passing to OpenAI.

        Returns:
            The completed text or an async iterator of text chunks if streaming.

        Raises:
            InvalidResponseError: If the response from OpenAI is invalid or empty.
            APIConnectionError: If there is a connection error with the OpenAI API.
            RateLimitError: If the OpenAI API rate limit is exceeded.
            APITimeoutError: If the OpenAI API request times out.
        """
        if history_messages is None:
            history_messages = []

        # Set openai logger level to INFO when VERBOSE_DEBUG is off
        if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
            logging.getLogger("openai").setLevel(logging.INFO)

        # Extract client configuration options
        client_configs = kwargs.pop("openai_client_configs", {})

        # Create the OpenAI client
        openai_async_client = self.create_openai_async_client(
            api_key=api_key, base_url=base_url, client_configs=client_configs
        )

        # Remove special kwargs that shouldn't be passed to OpenAI
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)

        # Prepare messages
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        logger.debug("===== Entering func of LLM =====")
        logger.debug(f"Model: {model}   Base URL: {base_url}")
        logger.debug(f"Additional kwargs: {kwargs}")
        logger.debug(f"Num of history messages: {len(history_messages)}")
        verbose_debug(f"System prompt: {system_prompt}")
        verbose_debug(f"Query: {prompt}")
        logger.debug("===== Sending Query to LLM =====")

        try:
            # Don't use async with context manager, use client directly
            if "response_format" in kwargs:
                response = await openai_async_client.beta.chat.completions.parse(
                    model=model, messages=messages, **kwargs
                )
            else:
                response = await openai_async_client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
        except APIConnectionError as e:
            logger.error(f"OpenAI API Connection Error: {e}")
            await openai_async_client.close()  # Ensure client is closed
            raise
        except RateLimitError as e:
            logger.error(f"OpenAI API Rate Limit Error: {e}")
            await openai_async_client.close()  # Ensure client is closed
            raise
        except APITimeoutError as e:
            logger.error(f"OpenAI API Timeout Error: {e}")
            await openai_async_client.close()  # Ensure client is closed
            raise
        except Exception as e:
            logger.error(
                f"OpenAI API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
            )
            await openai_async_client.close()  # Ensure client is closed
            raise

        if hasattr(response, "__aiter__"):

            async def inner():
                # Track if we've started iterating
                iteration_started = False
                try:
                    iteration_started = True
                    async for chunk in response:
                        # Check if choices exists and is not empty
                        if not hasattr(chunk, "choices") or not chunk.choices:
                            logger.warning(f"Received chunk without choices: {chunk}")
                            continue

                        # Check if delta exists and has content
                        if not hasattr(chunk.choices[0], "delta") or not hasattr(
                            chunk.choices[0].delta, "content"
                        ):
                            logger.warning(
                                f"Received chunk without delta content: {chunk.choices[0]}"
                            )
                            continue
                        content = chunk.choices[0].delta.content
                        if content is None:
                            continue
                        if r"\u" in content:
                            content = safe_unicode_decode(content.encode("utf-8"))
                        yield content
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    # Try to clean up resources if possible
                    if (
                        iteration_started
                        and hasattr(response, "aclose")
                        and callable(getattr(response, "aclose", None))
                    ):
                        try:
                            await response.aclose()
                            logger.debug("Successfully closed stream response after error")
                        except Exception as close_error:
                            logger.warning(
                                f"Failed to close stream response: {close_error}"
                            )
                    # Ensure client is closed in case of exception
                    await openai_async_client.close()
                    raise
                finally:
                    # Ensure resources are released even if no exception occurs
                    if (
                        iteration_started
                        and hasattr(response, "aclose")
                        and callable(getattr(response, "aclose", None))
                    ):
                        try:
                            await response.aclose()
                            logger.debug("Successfully closed stream response")
                        except Exception as close_error:
                            logger.warning(
                                f"Failed to close stream response in finally block: {close_error}"
                            )

                    # This prevents resource leaks since the caller doesn't handle closing
                    try:
                        await openai_async_client.close()
                        logger.debug(
                            "Successfully closed OpenAI client for streaming response"
                        )
                    except Exception as client_close_error:
                        logger.warning(
                            f"Failed to close OpenAI client in streaming finally block: {client_close_error}"
                        )

            return inner()

        else:
            try:
                if (
                    not response
                    or not response.choices
                    or not hasattr(response.choices[0], "message")
                    or not hasattr(response.choices[0].message, "content")
                ):
                    logger.error("Invalid response from OpenAI API")
                    await openai_async_client.close()  # Ensure client is closed
                    raise InvalidResponseError("Invalid response from OpenAI API")

                content = response.choices[0].message.content

                if not content or content.strip() == "":
                    logger.error("Received empty content from OpenAI API")
                    await openai_async_client.close()  # Ensure client is closed
                    raise InvalidResponseError("Received empty content from OpenAI API")

                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))

                if token_tracker and hasattr(response, "usage"):
                    token_counts = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)

                logger.debug(f"Response content len: {len(content)}")
                verbose_debug(f"Response: {response}")

                return content
            finally:
                # Ensure client is closed in all cases for non-streaming responses
                await openai_async_client.close()


    async def generate_text(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        if history_messages is None:
            history_messages = []
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = "json"
        model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
        return await self.openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )


    # @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192) # Temporarily commented out, suspected issue with 'self' handling
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=(
            retry_if_exception_type(RateLimitError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(APITimeoutError)
        ),
    )
    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
        base_url: str = None,
        api_key: str = None,
        client_configs: dict[str, Any] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings for a list of texts using OpenAI's API.

        Args:
            texts: List of texts to embed.
            model: The OpenAI embedding model to use.
            base_url: Optional base URL for the OpenAI API.
            api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
            client_configs: Additional configuration options for the AsyncOpenAI client.
                These will override any default configurations but will be overridden by
                explicit parameters (api_key, base_url).

        Returns:
            A numpy array of embeddings, one per input text.

        Raises:
            APIConnectionError: If there is a connection error with the OpenAI API.
            RateLimitError: If the OpenAI API rate limit is exceeded.
            APITimeoutError: If the OpenAI API request times out.
        """
        # Create the OpenAI client
        self.async_client = self.create_openai_async_client(
            api_key=api_key, base_url=base_url, client_configs=client_configs
        )

        async with self.async_client:
            response = await self.async_client.embeddings.create(
                model=model, input=texts, encoding_format="float"
            )
            return np.array([dp.embedding for dp in response.data])

    async def analyze_image(
        self, 
        image_data: str, 
        prompt: str, 
        model: str = "gpt-4o",
    ) -> str:
        """Analyze image with OpenAI's vision model"""
        if not self.async_client:
            self.async_client = self.create_openai_async_client(
            api_key=self.api_key, base_url=self.base_url,
        )
        
        try:
            response = await self.async_client.chat.completions.create(
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
            logger.error(f"OpenAI vision API error: {str(e)}")
            return f"Error analyzing image: {str(e)}"
        
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass
        


if __name__ == "__main__":
    # Example usage
    provider = OpenAIProvider()
    import types  # Import the types module

    async def test_openai():
        try:
            # Create an object for hashing_kv
            hashing_kv_obj = types.SimpleNamespace(
                global_config={"llm_model_name": "gpt-4o-mini"}
            )
            # Test chat completion
            response = await provider.generate_text(
                "Hello, how are you?",
                system_prompt="You are a helpful assistant.",
                history_messages=[],
                hashing_kv=hashing_kv_obj,  # Pass the object directly
            )
            print(f"Chat Completion Response: {response}")

            # Test embedding
            embedding = await provider.generate_embeddings(texts=["This is a test string."])
            print(f"Embedding Shape: {embedding.shape}")
        except Exception as e:
            print(f"Error: {e}")
    import asyncio
    asyncio.run(test_openai())