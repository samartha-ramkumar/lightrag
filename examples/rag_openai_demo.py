import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.provider import LLMService
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
from settings import global_args as args


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "coreflow_1.log"))

    # print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 5242880))  # Default 5MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(filename)s:%(lineno)d: - %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s",
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
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


WORKING_DIR = "./testing"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag(workspace_name: str | None = None):
    
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
        workspace=workspace_name,
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
        log_level="INFO"
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    import time
    start = time.time()
    workspace_name = "project_1"
    try:

        # Initialize RAG instance
        rag = await initialize_rag(workspace_name=workspace_name)
        end = time.time()
        print(f"\nTotal time initialization: {end - start:.2f} seconds")

        with open("./test_samples/sample.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(input=f.read(), ids=["doc_1"])

        print("=====================")
        
        # Example 1: Using chat method with automatic conversation management
        response, conversation_id = await rag.achat(
            "What is this document about?",
            param=QueryParam(mode="mix")
        )
        print(f"Conversation ID: {conversation_id}")
        print(f"Response: {response}")
        
        # Continue the conversation
        # conversation_id = "c7bb9a0f-72c1-41e7-8dc7-9db495865692"
        start = time.time()
        response2, _ = await rag.achat(
            # "I understand ..but what you gave is too much...Can you answer my question in less than 50-60 words?",
            # "My apologies, that became a little too less now. Can you give a bit more than that but still keep it concise?",
            # "What is the unique aspect of this document?",
            # "Is there nothing special about this document?",
            "Tell me something boring about this document.",
            conversation_id=conversation_id,
            param=QueryParam(mode="mix")
        )
        print(f"Follow-up response: {response2}")
        print(f"Conversation ID: {_}")
        end = time.time()
        print(f"\nTotal time for follow-up query: {end - start:.2f} seconds")
        # Example 2: Manual conversation history (original approach still works)
        # print("\n=== Manual Conversation History ===")
        # print(
        #     await rag.aquery(
        #         "Something unique about this document?",
        #         param=QueryParam(mode="mix",
        #                          conversation_history=[
        #                              {"role": "user", "content": "What is this document about?"},
        #                              {"role": "assistant", "content": "This document is about interviews..."}
        #                          ])
        #     )
        # )
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rag' in locals():
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
