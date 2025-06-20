"""
This module contains all query-related routes for the LightRAG API.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from lightrag.base import QueryParam
from pydantic import BaseModel, Field, field_validator
from lightrag import LightRAG
from ascii_colors import trace_exception

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="The query text",
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )

    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for continuing an existing chat. If not provided, a new conversation will be created.",
    )

    only_need_context: Optional[bool] = Field(
        default=False,
        description="If True, only returns the retrieved context without generating a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        default="Multiple Paragraphs",
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        default=60,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    # conversation_history: Optional[List[Dict[str, Any]]] = Field(
    #     default=None,
    #     description="Stores past conversation history to maintain context. Format: [{'role': 'user/assistant', 'content': 'message'}].",
    # )

    history_turns: Optional[int] = Field(
        ge=0,
        default=3,
        description="Number of complete conversation turns (user-assistant pairs) to consider in the response context.",
    )

    ids: list[str] | None = Field(
        default=None, description="List of ids to filter the results."
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        return query.strip()

    # @field_validator("conversation_history", mode="after")
    # @classmethod
    # def conversation_history_role_check(
    #     cls, conversation_history: List[Dict[str, Any]] | None
    # ) -> List[Dict[str, Any]] | None:
    #     if conversation_history is None:
    #         return None
    #     for msg in conversation_history:
    #         if "role" not in msg or msg["role"] not in {"user", "assistant"}:
    #             raise ValueError(
    #                 "Each message must have a 'role' key with value 'user' or 'assistant'."
    #             )
    #     return conversation_history

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        request_data = self.model_dump(exclude_none=True, exclude={"query", })

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        print(f"QueryParam: {param}")
        return param


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    conversation_id: str = Field(
        description="The conversation ID for this chat session",
    )


def create_query_routes(rag: LightRAG):

    @router.post(
        "/query", response_model=QueryResponse
    )
    async def query_text(request: QueryRequest):
        """
        Handle a POST request at the /query endpoint to process user queries using RAG chat capabilities.

        Parameters:
            request (QueryRequest): The request object containing the query parameters.
        Returns:
            QueryResponse: A Pydantic model containing the response and conversation ID.

        Raises:
            HTTPException: Raised when an error occurs during the request handling process,
                       with status code 500 and detail containing the exception message.
        """
        try:
            param = request.to_query_params(False)
            
            # Use achat for conversation management
            if request.conversation_id:
                # Continue existing conversation
                response, conversation_id = await rag.achat(
                    request.query, 
                    conversation_id=request.conversation_id,
                    param=param
                )
            else:
                # Start new conversation
                response, conversation_id = await rag.achat(
                    request.query,
                    param=param
                )

            return QueryResponse(response=response, conversation_id=conversation_id)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/query/stream")
    async def query_text_stream(request: QueryRequest):
        """
        This endpoint performs a retrieval-augmented generation (RAG) query and streams the response.

        Args:
            request (QueryRequest): The request object containing the query parameters.
            optional_api_key (Optional[str], optional): An optional API key for authentication. Defaults to None.

        Returns:
            StreamingResponse: A streaming response containing the RAG query results.
        """
        try:
            param = request.to_query_params(True)
            response = await rag.aquery(request.query, param=param)

            from fastapi.responses import StreamingResponse

            async def stream_generator():
                if isinstance(response, str):
                    # If it's a string, send it all at once
                    yield f"{json.dumps({'response': response})}\n"
                else:
                    # If it's an async generator, send chunks one by one
                    try:
                        async for chunk in response:
                            if chunk:  # Only send non-empty content
                                yield f"{json.dumps({'response': chunk})}\n"
                    except Exception as e:
                        logging.error(f"Streaming error: {str(e)}")
                        yield f"{json.dumps({'error': str(e)})}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Accel-Buffering": "no",  # Ensure proper handling of streaming response when proxied by Nginx
                },
            )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    return router