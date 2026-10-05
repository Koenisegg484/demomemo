from src.memory import add_memory, get_memories


user_id = "analyst_a"

add_memory(
    user_id,
    [
        {
            "role": "user",
            "content": "I prefer concise AML reports and I always check transaction velocity first."
        }
    ],
)

memories = get_memories(
    user_id,
    "What are my AML investigation preferences?"
)

for memory in memories:
    print(memory)