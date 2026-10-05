from mem0 import Memory


OLLAMA_URL = "http://localhost:11434"

memory_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "aml_analyst_memories",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen3",
            "temperature": 0,
            "ollama_base_url": OLLAMA_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": OLLAMA_URL,
        },
    },
}


memory = Memory.from_config(memory_config)


def add_memory(
    user_id: str,
    messages: list[dict],
):
    """Store useful information about an AML analyst."""
    return memory.add(
        messages,
        user_id=user_id,
    )


def get_memories(
    user_id: str,
    query: str,
    limit: int = 5,
):
    """Retrieve memories relevant to the current question."""
    result = memory.search(
        query=query,
        user_id=user_id,
        limit=limit,
    )

    return result.get("results", [])