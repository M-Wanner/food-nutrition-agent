import os

import numpy as np
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from rag_search import build_search_documents_tool
from usda_api import calculate_average_nutrition

load_dotenv()

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
  using [Source N].
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


def main():
	model = ChatOllama(model="qwen3.5:4b", temperature=0)
	tools = [calculate_average_nutrition, build_search_documents_tool()]
	agent = create_agent(
	    model,
	    tools,
	    system_prompt=system_prompt,
	    checkpointer=InMemorySaver(),
	    middleware=[
	        SummarizationMiddleware(
	            model=ChatOllama( model="qwen3.5:4b", temperature=0),
	            trigger=("tokens", 2500),
	            keep=("messages", 6),
	            trim_tokens_to_summarize=1200,
	        )
	    ],
	)

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
        except Exception as exc:
            print(f"An error occurred: {exc}")
        continue

	    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()