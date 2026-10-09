import os
import requests
import numpy as np
from langchain_core.tools import tool

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