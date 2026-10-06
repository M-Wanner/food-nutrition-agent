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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

nutrients = {
    1008: "energy_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbohydrates_g",
    2000: "sugar_g",
    1079: "fiber_g",
    1093: "sodium_mg",
}

filtered_terms = ["seasoning", "flavored", "potato", "chips", "scoop"]


def search_product(product: str, page_size: int = 200) -> dict:
    """Search USDA FoodData Central for foods matching a product name."""

    api_key = os.getenv("USDA_API_KEY")

    if not api_key:
        raise RuntimeError("USDA_API_KEY is not configured.")

    response = requests.get(
        "https://api.nal.usda.gov/fdc/v1/foods/search",
        params={
            "api_key": api_key,
            "query": product,
            "pageSize": page_size,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def filter_relevant_foods(foods: list, product: str) -> list:
    """Keep only search results that plausibly match the requested product."""
    product = product.lower().strip()
    relevant_foods = []

    for food in foods:
        description = food.get("description", "").lower()

        if product not in description:
            continue
        if any(term in description for term in filtered_terms):
            continue

        relevant_foods.append(food)

    return relevant_foods


def extract_nutrient_values(foods: list) -> dict:
    """Extract nutrient values from a list of USDA food results."""
    values = {name: [] for name in nutrients.values()}

    for food in foods:
        for nutrient in food.get("foodNutrients", []):
            nutrient_id = nutrient.get("nutrientId")
            value = nutrient.get("value")
            if nutrient_id in nutrients and value is not None:
                values[nutrients[nutrient_id]].append(value)

    return values


def calculate_averages(values: dict) -> dict:
    """Calculate the average value for each nutrient across matched foods."""
    averages = {}
    for name, nutrient_values in values.items():
        averages[name] = round(float(np.mean(nutrient_values)), 2) if nutrient_values else None
    return averages


@tool
def calculate_average_nutrition(product: str) -> dict:
    """Calculate average nutrition values per 100 g for a food."""
    try:
        data = search_product(product)
    except requests.RequestException as exc:
        return {
            "food": product,
            "products_found": 0,
            "nutrition_per_100g": None,
            "message": f"USDA API request failed: {exc}",
        }

    foods = filter_relevant_foods(data.get("foods", []), product)

    if not foods:
        return {
            "food": product,
            "products_found": 0,
            "nutrition_per_100g": None,
            "message": f"No nutrition data found for '{product}'.",
        }

    averages = calculate_averages(extract_nutrient_values(foods))

    return {
        "food": product,
        "products_found": len(foods),
        "nutrition_per_100g": {
            "energy": f"{averages['energy_kcal']} kcal",
            "protein": f"{averages['protein_g']} g",
            "fat": f"{averages['fat_g']} g",
            "carbohydrates": f"{averages['carbohydrates_g']} g",
            "sugar": f"{averages['sugar_g']} g",
            "fiber": f"{averages['fiber_g']} g",
            "sodium": f"{averages['sodium_mg']} mg",
        },
    }


sources = [
    "https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/food-allergies",  # food allergies
    "https://cdn.realfood.gov/DGA.pdf",  # general dietary advice
    "https://assets.publishing.service.gov.uk/media/5a749fece5274a44083b82d8/government_dietary_recommendations.pdf",  # general advice
    "https://www.nhs.uk/live-well/eat-well/food-guidelines-and-food-labels/how-to-read-food-labels/",  # food labels
    "https://www.nhs.uk/live-well/eat-well/how-to-eat-a-balanced-diet/the-vegan-diet/",  # vegan diet
    "https://www.nhs.uk/live-well/eat-well/how-to-eat-a-balanced-diet/the-vegetarian-diet/",  # vegetarian diet
    "https://www.nhs.uk/live-well/eat-well/how-to-eat-a-balanced-diet/eight-tips-for-healthy-eating/",  # general tips
]


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents as readable text with a source label."""
    return "\n\n".join(
        f"[Source {index}]\n"
        f"{doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content[:1500]}"
        for index, doc in enumerate(docs, start=1)
    )


def build_search_documents_tool():
    """Load and index all sources, then return the bound `search_documents` tool."""
    documents = []
    for url in sources:
        try:
            loader = PyPDFLoader(url) if url.endswith(".pdf") else WebBaseLoader(url)
            documents.extend(loader.load())
        except Exception as exc:
            print(f"Warning: could not load {url}: {exc}")

    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)

    vector_store = Chroma(
        collection_name="Documents",
        persist_directory="./chroma_db",
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
    )

    existing_ids = vector_store.get().get("ids", [])
    if existing_ids:
        vector_store.delete(ids=existing_ids)
    vector_store.add_documents(documents=splits)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    @tool
    def search_documents(question: str) -> str:
        """Search the nutrition knowledge base for relevant information."""
        try:
            documents = retriever.invoke(question)

            if not documents:
                return "No relevant information was found in the provided documents."

            return format_docs(documents)

        except Exception as exc:
            return f"Error while searching documents: {exc}"

    return search_documents


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