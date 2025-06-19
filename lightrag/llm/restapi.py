import os
import requests
import atexit
import aiohttp
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union, List
from functools import wraps
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
)
from lightrag.utils import logger
from lightrag.utils import get_env_value
from collections.abc import AsyncIterator

def llm_error_handler(func):
    """Decorator for consistent error handling in LLM operations"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Get the class name of the first argument (self)
            class_name = args[0].__class__.__name__ if args else "Unknown"
            
            # Get the function name
            func_name = func.__name__
            
            # Get provider info if available in self
            provider_info = ""
            if args and hasattr(args[0], "provider_name"):
                provider_info = f" (Provider: {args[0].provider_name})"
            
            # Log the error with context
            logger.error(f"{class_name}.{func_name}{provider_info} error: {str(e)}", exc_info=True)
            
            # Return error message
            return f"Error in {func_name}: {str(e)}"
    return wrapper


class RestApiProvider:
    """OpenAI provider class for handling Rest API interactions."""

    def __init__(self):
        """Initialize the REST API provider"""
        self.base_url = get_env_value("api_base_url" "http://host.docker.internal:2100")
        self.app_id = get_env_value("app_id", "Research-12345")
        self.session = None
        
        # Map of task types to model categories
        self.category_mapping = {
            "chat": "text",
            "vision": "multimodal",
            "embedding": "embedding"
        }
        
        # Register cleanup on exit
        atexit.register(self._sync_cleanup)
        
        logger.info(f"Initialized REST API provider with base URL: {self.base_url}")
    
    def _sync_cleanup(self):
        """Synchronous cleanup to be called on exit"""
        if self.session and not self.session.closed:
            logger.info("Cleaning up unclosed session on exit")
            asyncio.run(self.cleanup())
    
    @asynccontextmanager
    async def _session_context(self):
        """Context manager for session handling"""
        session = await self._get_session()
        try:
            yield session
        except Exception as e:
            logger.error(f"Error during session operation: {str(e)}")
            raise
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False, force_close=True),
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self.session

        
    async def generate_text(
            self,
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            model: Optional[str] = None,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text by making a request to the backend chat completion endpoint"""
        category = self.category_mapping.get("chat", "text")
        endpoint = f"{self.base_url}/model/process/{category}/chat/completion"
        
        # Build messages in OpenAI format
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history messages if provided
        if history_messages:
            messages.extend(history_messages)
        
        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "temperature": 0.01,
            "max_tokens": 4000
        }
        
        if model:
            payload["model"] = model
        
        headers = {"app-id": self.app_id, "Content-Type": "application/json"}
        
        async with self._session_context() as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {response.status} - {error_text}")
                    return f"Error: API returned status {response.status}: {error_text}"
                
                result = await response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
    async def analyze_image(
        self, 
        image_data: str, 
        prompt: str,
        model: Optional[str] = None
    ) -> str:
        """Analyze image by making a request to the backend vision API"""
        category = self.category_mapping.get("vision", "multimodal")
        endpoint = f"{self.base_url}/model/process/{category}/chat/completion"
        
        # Create message with text and image content
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
        
        payload = {
            "messages": [message],
            "temperature": 0.01,
            "max_tokens": 4000
        }

        if model:
            payload["model"] = model

        headers = {"app-id": self.app_id, "Content-Type": "application/json"}
        
        logger.debug(f"Making image analysis request to {endpoint}")
        
        async with self._session_context() as session:
            try:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API vision error: {response.status} - {error_text}")
                        return f"Error analyzing image: API returned status {response.status}: {error_text}"
                    
                    result = await response.json()
                    if "error" in result:
                        logger.error(f"API vision error in response: {result['error']}")
                        return f"Error analyzing image: {result['error']}"
                        
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0].get("message", {}).get("content", "")
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return "Error: Unexpected response format from vision API"
            except Exception as e:
                logger.error(f"Exception during image analysis: {str(e)}", exc_info=True)
                return f"Error analyzing image: {str(e)}"

    @llm_error_handler
    async def generate_embeddings(
        self, 
        texts: str | list[str] | list[dict[str, str]],
        model: str = "text-embedding-3-small",
        base_url: str = None,
        api_key: str = None,
        client_configs: dict[str, Any] = None,
        **kwargs) -> List[List[float]]:
        """Generate embeddings by making a request to the embedding endpoint"""
        category = "embedding"  # This is fixed as per model.mjs
        
        endpoint = f"{self.base_url}/model/process/{category}/embedding"
        
        
        # If text is a list of dictionaries, convert to list of strings
        if isinstance(texts, List) and all(isinstance(item, dict) for item in texts):
            payload = {
                "input": [item.get("content", "") for item in texts]
            }
        # If input is str or List[str], directly use it
        elif isinstance(texts, str) or (isinstance(texts, List) and all(isinstance(item, str) for item in text)):
            payload = {
                "input": texts
            }
        
        if model:
            payload["model"] = model
        
        headers = {"app-id": self.app_id, "Content-Type": "application/json"}
        async with self._session_context() as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API embedding error: {response.status} - {error_text}")
                    return []
                
                result = await response.json()
                embeddings = [item.get("embedding", []) for item in result.get("data", [])]
                return embeddings
            
    async def cleanup(self) -> None:
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("REST API provider session closed")