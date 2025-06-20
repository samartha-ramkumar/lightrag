"""
Configs for the LightRAG API.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from lightrag.utils import get_env_value

from lightrag.constants import (
    DEFAULT_WOKERS,
    DEFAULT_TIMEOUT,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)




class DefaultRAGStorageConfig:
    KV_STORAGE = "PGKVStorage"
    VECTOR_STORAGE = "PGVectorStorage"
    GRAPH_STORAGE = "FalkorDBStorage"
    DOC_STATUS_STORAGE = "PGDocStatusStorage"


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    args = argparse.Namespace()

    # Server configuration
    args.host = get_env_value("HOST", "0.0.0.0")
    args.port = get_env_value("PORT", 8080, int)

    # Directory configuration
    args.working_dir = get_env_value("WORKING_DIR", "./rag_storage")
    args.input_dir = get_env_value("INPUT_DIR", "./rag_storage/input")

    args.timeout = get_env_value("TIMEOUT", DEFAULT_TIMEOUT, int, special_none=True)

    # RAG configuration
    args.max_async = get_env_value("MAX_ASYNC", 4, int)
    args.max_tokens = get_env_value("MAX_TOKENS", 32768, int)

    # Logging configuration
    args.log_level = get_env_value("LOG_LEVEL", "INFO")
    # Ensure choices are handled if necessary, though get_env_value doesn't enforce choices.
    # For simplicity, we assume valid values are provided via environment.
    args.verbose = get_env_value("VERBOSE", False, bool)

    args.history_turns = get_env_value("HISTORY_TURNS", 3, int)
    
    # Search parameters
    args.top_k = get_env_value("TOP_K", 60, int)
    args.cosine_threshold = get_env_value("COSINE_THRESHOLD", 0.2, float)

    # Namespace
    args.namespace_prefix = get_env_value("NAMESPACE_PREFIX", "")
    args.workspace = get_env_value("WORKSPACE", "project_2")

    # Server workers configuration
    args.workers = get_env_value("WORKERS", DEFAULT_WOKERS, int)


    # convert relative path to absolute path
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)

    # Inject storage configuration from environment variables
    args.kv_storage = get_env_value(
        "LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE
    )
    args.doc_status_storage = get_env_value(
        "LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE
    )
    args.graph_storage = get_env_value(
        "LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE
    )
    args.vector_storage = get_env_value(
        "LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE
    )
    args.conversation_storage = get_env_value(
        "LIGHTRAG_CONVERSATION_STORAGE", "PGConversationStorage"
    )

    # Get MAX_PARALLEL_INSERT from environment
    args.max_parallel_insert = get_env_value("MAX_PARALLEL_INSERT", 2, int)

    # Inject model configuration
    args.llm_provider = get_env_value("LLM_PROVIDER", "openai")
    args.llm_model = get_env_value("LLM_MODEL", "gpt-4o-mini")
    args.embedding_model = get_env_value("EMBEDDING_MODEL", "text-embedding-3-small")
    args.embedding_dim = get_env_value("EMBEDDING_DIM", 1536, int)
    args.max_embed_tokens = get_env_value("MAX_EMBED_TOKENS", 8192, int)

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1536, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 128, int)

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = get_env_value(
        "ENABLE_LLM_CACHE_FOR_EXTRACT", True, bool
    )
    args.enable_llm_cache = get_env_value("ENABLE_LLM_CACHE", True, bool)

    # Inject LLM temperature configuration
    args.temperature = get_env_value("TEMPERATURE", 0.01, float)

    # Select Document loading tool (DOCLING, DEFAULT)
    args.document_loading_engine = get_env_value("DOCUMENT_LOADING_ENGINE", "DEFAULT")

    # Add environment variables that were previously read directly
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    args.summary_language = get_env_value("SUMMARY_LANGUAGE", "English")
    args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/api/*")



    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if global_args.workers > 1:
        original_workers = global_args.workers
        global_args.workers = 1
        # Log warning directly here
        logging.warning(
            f"In uvicorn mode, workers parameter was set to {original_workers}. Forcing workers=1"
        )


global_args = parse_args()