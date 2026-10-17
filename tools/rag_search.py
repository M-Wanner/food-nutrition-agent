import os
import numpy as np
import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from dotenv import load_dotenv
import streamlit as st

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

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]

    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)

    openrouter_embeddings  = OpenAIEmbeddings(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            model="nvidia/nemotron-3-embed-1b:free",       
            check_embedding_ctx_length=False,
            chunk_size=128,
            default_headers={
                "X-Title": "Food Nutrition Agent",
            }
        )

    vector_store = Chroma(
        collection_name="Documents",
        persist_directory="./chroma_db",
        embedding_function=openrouter_embeddings,
    )

    if not documents:
        raise RuntimeError("No RAG documents could be loaded.")

    existing_ids = vector_store.get(limit=1).get("ids", [])

    if not existing_ids:
        vector_store.add_documents(documents=splits)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    @tool
    @traceable
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