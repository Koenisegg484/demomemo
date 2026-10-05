from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from src.tools import TOOLS


MODEL_NAME = "qwen2.5:7b-instruct"


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


SYSTEM_PROMPT = """
You are an AML investigation assistant.

You analyze financial transaction datasets using the available tools.

Rules:
- Use tools whenever the user's question requires data analysis.
- Never invent transaction numbers or statistics.
- Clearly distinguish between calculated facts and potential AML indicators.
- If the user asks for an AML investigation, use relevant tools before answering.
- Keep answers concise but provide enough reasoning to explain the result.
"""


agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)


def ask_agent(question: str) -> str:
    """Send a question to the AML agent."""

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    return result["messages"][-1].content