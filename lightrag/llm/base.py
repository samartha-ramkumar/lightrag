from typing import Optional, Union, AsyncIterator, List, Any
from abc import ABC, abstractmethod




class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text based on prompt"""
        pass
    
    # @abstractmethod
    # async def generate_streaming_response(
    #     self,
    #     system_prompt: str,
    #     user_prompt: str,
    #     model: Optional[str] = None
    # ) -> AsyncIterator[str]:
    #     """Generate streaming response"""
    #     pass
    
    @abstractmethod
    async def analyze_image(
        self, 
        image_data: str, 
        prompt: str, 
        model: Optional[str] = None
    ) -> str:
        """Analyze image based on prompt"""
        pass
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass