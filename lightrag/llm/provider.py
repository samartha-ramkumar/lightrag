import asyncio
import re
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from lightrag.llm.openai import OpenAIProvider
from lightrag.llm.azure_openai import AzureOpenAIProvider
# from lightrag.llm.ollama import OllamaProvider
from lightrag.llm.restapi import RestApiProvider
from lightrag.utils import logger
from lightrag.utils import get_env_value

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


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text based on prompt"""
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate streaming response"""
        pass
    
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
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass


class LLMServiceFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def get_provider(provider_name: str) -> BaseLLMProvider:
        """Get appropriate provider based on settings"""
            
        if not provider_name:
            provider_name = get_env_value('LLM_PROVIDER', 'openai').lower()
                
        
        if provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "azure_openai" or provider_name == "azure-openai":
            return AzureOpenAIProvider()
        # elif provider_name == "ollama":
        #     return OllamaProvider()
        elif provider_name == "restapi":
            return RestApiProvider()

        else:
            # Log warning and default to OpenAI if provider is unrecognized
            logger.warning(f"Unrecognized provider '{provider_name}' in settings, defaulting to OpenAI")
            return OpenAIProvider()
        

class LLMService:
    """Service for interacting with language models with improved concurrency control"""
    
    def __init__(self, provider_name: Optional[str] = None):
        """Initialize the appropriate LLM provider based on settings"""
        # Initialize provider based on settings

        # Get provider from factory based on settings
        self.provider = LLMServiceFactory.get_provider(provider_name=provider_name)
        self.provider_name = provider_name or self.provider.__class__.__name__.replace('Provider', '').lower()
    
    async def generate_text(
            self,
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate text from a prompt (with caching)
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            conversation_history: Optional conversation history for context
            model: Optional model to use for generation
        """
        if self.provider:
            response = await self.provider.generate_text(prompt, system_prompt=system_prompt, history_messages=history_messages, keyword_extraction=keyword_extraction, **kwargs)
        else:
            logger.error("No LLM provider initialized. Cannot generate text.")
            return "Error: No LLM provider initialized."
        
        return response
    

    async def analyze_image(self, image_data: str, prompt_type: str = "general", model: Optional[str] = None) -> str:
        """
        Analyze image with vision API
        
        Args:
            image_data: Base64 encoded image data
            prompt_type: Type of analysis to perform ('general', 'chart', 'document', 'id_document')
            model: Optional specific model to use
        """
            
        prompt = self._get_vision_prompt(prompt_type)
        
        # Add explicit instruction to avoid meta-language
        direct_instruction = "IMPORTANT: Provide ONLY factual information visible in the image. DO NOT include statements about your capabilities or limitations."
        
        enhanced_prompt = f"{direct_instruction}\n\n{prompt}"

        raw_response = await self.provider.analyze_image(image_data, enhanced_prompt, model)
        
        # Post-process to remove common meta-language patterns
        processed_response = self._clean_vision_response(raw_response)
        
        return processed_response

    def _clean_vision_response(self, response: str) -> str:
        """Remove meta-language patterns from vision responses"""
        # Remove common disclaimers and meta-language
        patterns_to_remove = [
            r"I('m| am) (unable|not able) to analyze images directly",
            r"As an AI( model| assistant)?",
            r"I don't have the ability to",
            r"I cannot see",
            r"I do not have access to",
            r"I'm just an AI",
            r"I cannot directly view"
        ]
        
        cleaned = response
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Remove lines that are just disclaimers
        lines = cleaned.split('\n')
        filtered_lines = [line for line in lines if not any(disclaimer in line.lower() for disclaimer in 
                         ["disclaimer", "i cannot", "i don't", "i'm not", "as an ai"])]
        
        return '\n'.join(filtered_lines).strip()
    
    def _get_vision_prompt(self, prompt_type: str) -> str:
        """Get appropriate prompt for vision analysis"""
        if prompt_type == "chart":
            return """
            DATA EXTRACTION TASK: Extract precise data from this chart/graph.
            
            INSTRUCTIONS:
            1. First identify the chart type (bar chart, line graph, pie chart, scatter plot, etc.)
            2. Extract ALL numerical data points visible in the chart
            3. Include ALL labels, values, units, and relationships between data
            
            OUTPUT FORMAT:
            ```
            Type: [precise chart type]
            Title: [exact title - write "None" if not present]
            X-axis: [label and units]
            Y-axis: [label and units]
            
            Data series: [name of data series if multiple series exist]
            Data points:
            - [Category/Label]: [Exact numeric value] [unit]
            - [Category/Label]: [Exact numeric value] [unit]
            
            Key insights:
            - [brief factual observation about highest/lowest values]
            - [brief factual observation about trends or patterns]
            ```
            
            IMPORTANT:
            - Use EXACT numbers from the chart - do not round or approximate unless necessary
            - Include ALL data points visible in the image
            - Maintain original formatting and precision
            - For pie charts, include percentages if shown
            - For multi-series charts, clearly separate each series
            - If any information is unclear or not present, indicate with [not visible]
            - DO NOT include statements about your capabilities or limitations
            """
        elif prompt_type == "document":
            return """
            DATA EXTRACTION TASK: Extract ALL text from this document.

            Format: 
            ```
            [Transcribed text exactly as appears, maintaining original structure]
            ```
            
            RULES:
            - Transcribe ALL visible text in reading order
            - Preserve paragraph breaks, bullet points, and formatting
            - Include tables as structured text using alignment
            - Include headers, footers, and page numbers
            - Include form field labels AND values
            - For tables, preserve column/row structure using spaces or markdown tables
            - EXCLUDE any commentary or statements about your capabilities
            - DO NOT include phrases like "I see" or "the image shows"
            - DO NOT include any disclaimers
            - FOCUS ONLY on the document content
            """
        elif prompt_type == "general":
            return """
            DATA EXTRACTION TASK: Describe this image factually and completely.
            
            INSTRUCTIONS:
            1. Start with the main subject/content of the image
            2. Describe key visual elements in order of importance
            3. Include ALL relevant details: people, objects, text, activities, settings
            4. If text is visible in the image, transcribe it exactly
            5. For data visualizations, extract key numerical information
            
            FORMAT:
            ```
            Content: [Brief overview of what the image shows]
            
            Details:
            - [Main subject/focus with specific details]
            - [Secondary elements with specific details]
            - [Background/setting with specific details]
            - [Any visible text, transcribed exactly]
            
            Data points: [If applicable, list key numerical data]
            ```
            
            IMPORTANT:
            - Be SPECIFIC and PRECISE - mention exact numbers, colors, positions
            - Include ALL relevant information visible in the image
            - Avoid ambiguous language like "appears to be" unless truly uncertain
            - DO NOT include statements about your capabilities or analysis process
            - DO NOT include ANY content not directly visible in the image
            """
        else:
            return """
            Describe everything you see in this image in detail, including:
            
            1. The main subject or focus
            2. Any text visible in the image (transcribe exactly)
            3. Key visual elements and their arrangement
            4. If it's a chart or graph, extract the specific data points and values
            5. If it's a document, transcribe the important text content
            
            Be factual, specific and comprehensive. Include all relevant information visible in the image.
            """
    
    async def generate_embeddings(
        self, 
        texts: list[str],
        model: str = "text-embedding-3-small",
        base_url: str = None,
        api_key: str = None,
        client_configs: dict[str, Any] = None,
        **kwargs) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to generate embeddings for
            model: Optional model name to use for embeddings
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []

        if self.provider_name == "restapi":
            return await self.provider.generate_embeddings(
                    texts=texts,
                    **kwargs
                )
        else:
            # Use environment variable or default model based on provider
            embedding_model = model
            if not embedding_model:
                if self.provider_name == "ollama":
                    embedding_model = get_env_value('EMBEDDING_MODEL', "nomic-embed-text")
                elif self.provider_name == "openai":
                    embedding_model = get_env_value('EMBEDDING_MODEL', "text-embedding-3-small")                

            try:
                
                logger.info(f"Generating embeddings with {self.provider_name} using model: {embedding_model}")
                embeddings = await self.provider.generate_embeddings(
                    texts=texts,
                    model=embedding_model,
                    base_url=base_url,
                    api_key=api_key,
                    client_configs=client_configs,
                    **kwargs
                )

                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings with {self.provider_name}: {str(e)}")
        

    async def batch_process(self, prompts: List[str], model: Optional[str] = None, 
                            max_concurrency: int = 5) -> List[str]:
        """
        Process multiple prompts efficiently in parallel with rate limiting
        
        Args:
            prompts: List of prompts to process
            model: Optional model to use
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of responses in same order as prompts
        """
        if not prompts:
            return []
        
        # Create a semaphore to limit concurrency within this batch
        batch_semaphore = asyncio.Semaphore(min(max_concurrency, len(prompts)))
        
        async def process_single(prompt):
            async with batch_semaphore:
                return await self.provider.generate_text(prompt=prompt, model=model)
        
        # Create tasks for all prompts
        tasks = [process_single(prompt) for prompt in prompts]
        
        # Execute all tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, converting exceptions to error messages
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing prompt {i}: {str(result)}")
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results