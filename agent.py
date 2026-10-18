from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from tools.rag_search import build_search_documents_tool
from tools.usda_api import calculate_average_nutrition, lookup_multiple_foods

load_dotenv()

class GuardDecision(BaseModel):
    verdict: Literal["SAFE", "UNSAFE"]
    reason: str

guard_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
).with_structured_output(GuardDecision)

guard_prompt = """
Classify the text as UNSAFE only if it provides personalized medical diagnosis,
recommends specific medication or dosage, or gives individualized treatment instructions.

General health information, nutrition guidance, food-allergen information, and advice to
seek professional or emergency care are SAFE.

Return only SAFE or UNSAFE.

Text: {agent_response}
"""


system_prompt = """
You are an expert food nutrition assistant.

Rules:
- Use only information returned by the available tools.
- Never invent numbers, facts, calculations, sources, or recommendations.
- Use calculate_average_nutrition for quantitative nutrition information.
- Use search_documents for dietary guidance, vegan and vegetarian diets,
  food allergies, food labels, and general nutrition knowledge.
- Use both tools when a question requires both nutrition data and general
  nutrition knowledge.
- Nutrition values returned by the nutrition tool must be used exactly as
  provided.
- Do not treat missing values as zero.
- Do not estimate missing nutritional values.
- When calculating quantities for a meal, calculate them from the returned
  nutrition values.
- Nutrition values are averages per 100 g unless another amount is explicitly
  requested.
- Always state how many USDA products were used when reporting average values.
- When answering from retrieved documents, cite the relevant information
  using clickable Markdown links, for example: [NHS - The vegan diet](https://www.nhs.uk/...)
- Use the exact URL provided in the retrieved document context.
- Never invent, modify, or guess source URLs.
- Never cite a source that was not returned by the search_documents tool.
- If the documents do not contain the requested information, explicitly say
  that the information could not be found in the provided sources.
- Do not use external knowledge to fill gaps in retrieved documents.
- Do not assume that a specific commercial food is vegan or allergen-free
  unless the available information supports that conclusion.
- For allergies, intolerances, medical conditions, treatment, or medication,
  do not provide personalized medical advice.
- For potentially serious allergic reactions, advise seeking emergency
  medical care.
- Answer directly and avoid unnecessary disclaimers.
- Use tables for nutritional comparisons.
"""

def get_agent():
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    tools = [calculate_average_nutrition, build_search_documents_tool(), lookup_multiple_foods]
    agent = create_agent(
        model,
        tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite"),
                trigger=("tokens", 2500),
                keep=("messages", 6),
                trim_tokens_to_summarize=1200,
            )
        ],
    )
    return agent


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])

        return "\n".join(parts).strip()

    return ""

@traceable
def main():
    agent = get_agent()
    thread_config = {"configurable": {"thread_id": "food-agent-session"}}

    while True:
        print("-------------------------------")
        try:
            question = input("Ask your question (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        print("\n")
        if not question:
            continue
        if question.lower() == "q":
            break

        try:
            result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=thread_config)
            agent_response_text = content_to_text(result["messages"][-1].content)

            guard_decision = guard_model.invoke([ HumanMessage(content=guard_prompt.format(agent_response=agent_response_text))])

            if guard_decision.verdict != "SAFE":
                agent_response_text = ("I cannot provide personalized medical advice. Please consult a qualified healthcare professional.")

        except Exception as exc:
            print(f"An error occurred: {exc}")
            continue

        print(agent_response_text)


if __name__ == "__main__":
    main()