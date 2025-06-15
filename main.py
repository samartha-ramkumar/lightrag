"""
Coreflow RAG FastAPI Server
"""

from fastapi import FastAPI, Depends, HTTPException, status
import asyncio
import os
import logging
import logging.config
import uvicorn
from fastapi.responses import RedirectResponse
from pathlib import Path
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.utils import get_env_value
import sys
from lightrag import LightRAG, __version__ 
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
)
from api.documents.processor import (
    DocumentManager,
    create_document_routes
)
from api.queries.router import create_query_routes

from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    initialize_pipeline_status,
)

from lightrag.llm.provider import LLMService
from settings import global_args, update_uvicorn_mode_config

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Add this new global variable to hold the app instance
app = None

def create_app(args):
    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    llm_provider = LLMService(provider_name=args.llm_provider)

    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=args.max_embed_tokens,
        func=lambda texts: llm_provider.generate_embeddings(
            texts,
            model=args.embedding_model
        ),
    )

    rag = LightRAG(
        working_dir=args.working_dir,
        workspace=args.workspace,
        llm_model_func= llm_provider.generate_text,
        llm_model_name=args.llm_model,
        llm_model_max_async=args.max_async,
        llm_model_max_token_size=args.max_tokens,
        chunk_token_size=int(args.chunk_size),
        chunk_overlap_token_size=int(args.chunk_overlap_size),
        # llm_model_kwargs={
        #     "timeout": args.timeout,
        #     "options": {"num_ctx": args.max_tokens},
        # },
        embedding_func=embedding_func,
        kv_storage=args.kv_storage,
        graph_storage=args.graph_storage,
        vector_storage=args.vector_storage,
        doc_status_storage=args.doc_status_storage,
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": args.cosine_threshold
        },
        enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
        enable_llm_cache=args.enable_llm_cache,
        auto_manage_storages_states=False,
        max_parallel_insert=args.max_parallel_insert,
    )

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            await rag.initialize_storages()

            await initialize_pipeline_status()
            pipeline_status = await get_namespace_data("pipeline_status")
            logger.info(f"Pipeline status initialized: {pipeline_status}")

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            await rag.finalize_storages()

    # Initialize FastAPI
    app_kwargs = {
        "title": "Coreflow RAG API",
        "description": "Providing API for LightRAG core, Web UI and Ollama Model Emulation",
        "version": __version__,
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": "/docs",  # Explicitly set docs URL
        "redoc_url": "/redoc",  # Explicitly set redoc URL
        "lifespan": lifespan,
    }

    app = FastAPI(**app_kwargs)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes
    app.include_router(create_document_routes(rag, doc_manager))
    app.include_router(create_query_routes(rag, args.top_k))

    @app.get("/")
    async def root():
        """Redirects to the API documentation."""
        return RedirectResponse(url="/docs")


    @app.get("/health")
    async def get_status():
        """Get current system status"""
        try:
            pipeline_status = await get_namespace_data("pipeline_status")

            return {
                "status": "healthy",
                "pipeline_busy": pipeline_status.get("busy", False),

            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app



def configure_logging():
    """Configure logging for uvicorn startup"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(name)s - %(levelname)s - [%(filename)s:%(lineno)d]-  %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Configure all uvicorn related loggers
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.utils.LightragPathFilter",
                },
            },
        }
    )



def main():
    from multiprocessing import freeze_support
    
    freeze_support()

    # Configure logging before parsing args
    configure_logging()
    update_uvicorn_mode_config()

    # When running directly, create the app and run using import string
    # Create application instance directly instead of using factory function
    global app
    app = create_app(global_args)

    uvicorn.run(app=app, 
                host=global_args.host,
                port=global_args.port, 
                workers=global_args.workers, 
                timeout_keep_alive=global_args.timeout)


if __name__ == "__main__":
    main()