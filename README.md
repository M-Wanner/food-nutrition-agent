# Food Nutrition Agent

A conversational AI agent that answers questions about nutrition, diet, plant-based alternatives and other questions related to food.

## Functions

-   **Structured data**: real-time nutrition lookups via the [USDA FoodData Central](https://fdc.nal.usda.gov/) API returning calories, protein, fat, carbohydrates, sugar, fiber, and sodium.
-   **Unstructured data**: Retrieval Augmented Generation (RAG) over a set of public health authority documents on diet, allergies, and vegan/vegetarian eating.
-   **Short-term memory**: the agent remembers what was said earlier in the conversation and can answer follow-up questions about it without you repeating the details.
-   **Short-term memory**: the agent remembers what was said earlier in the conversation and can answer follow-up questions about it without you repeating the details.

The agent uses the ReAct agent pattern and decides, whether it needs to look up nutrition facts, search the document collection, both, or neither.

Everything runs locally via [Ollama](https://ollama.com/).

 **Disclaimer**: This project is for educational/portfolio purposes. It does not provide medical advice. For food allergies, intolerances, or any other health condition, always consult a qualified healthcare professional.

## Setup

### Prerequisites

1.  Python 3.10+
2.  [Ollama](https://ollama.com/) installed and running locally
3.  Pull the two models this project uses:
    
    ```bash
    ollama pull qwen3.5:4b
    ollama pull mxbai-embed-large
    ```
    
4.  A free USDA FoodData Central API key: https://fdc.nal.usda.gov/api-key-signup.html

### Installation

```bash
git clone https://github.com/M-Wanner/food-nutrition-agent
cd food-nutrition-agent
pip install -r requirements.txt
```
Insert your API key into the `.env` file.

### Run

```bash
python agent.py
```

## Example Conversation

>**Q: I had a meal consisting of 150 g chicken breast, 200 g cooked rice, 100 g broccoli, and 30 g ketchup. What nutrients did I consume in total?**

Based on the nutrition data available, I can calculate the nutrients from three of your four meal components. Unfortunately, no nutrition data was found for "cooked rice" in the database, so I cannot provide a complete total for that portion.

Here is the nutritional breakdown for the components with available data:

| Nutrient | Chicken Breast (150g) | Broccoli (100g) | Ketchup (30g) |
|----------|----------------------|-----------------|---------------|
| Energy | 249.1 kcal | 102.0 kcal | 34.0 kcal |
| Protein | 34.4 g | 5.3 g | 0.0 g |
| Fat | 9.9 g | 4.3 g | 0.0 g |
| Carbohydrates | 4.1 g | 10.9 g | 8.0 g |
| Sugar | 0.9 g | 1.3 g | 6.4 g |
| Fiber | 0.2 g | 1.7 g | 0.1 g |
| Sodium | 748.9 mg | 234.7 mg | 279.2 mg |

**Total from available components:**
- Energy: ~385 kcal
- Protein: ~39.7 g
- Fat: ~14.2 g
- Carbohydrates: ~23.0 g
- Sugar: ~8.6 g
- Fiber: ~2.0 g
- Sodium: ~1,262 mg

**Note:** The cooked rice portion (200g) could not be included in this calculation because no nutrition data was found for "cooked rice" in the database. USDA products were used for chicken breast (47 products), broccoli (119 products), and ketchup (187 products).

>**Q: Is this meal vegan? If not, which ingredients would need to be replaced, and what vegan alternatives would you recommend? Also compare the nutritional values of the original and vegan alternatives.**


This meal is **not vegan** because it contains chicken breast, which is an animal product. To make this meal vegan, you would need to replace the chicken breast with a plant-based protein alternative.

### Non-Vegan Ingredients:
- **Chicken breast (150g)** - Contains animal products

### Vegan Alternatives:
Based on available information, good vegan alternatives for chicken breast include:
- **Tofu** - A versatile soy product that can be used in various preparations
- **Tempeh** - Fermented soy product with a firm texture
- Other options mentioned in dietary guidance include mycoprotein (such as Quorn) and textured vegetable protein

### Nutritional Comparison:

| Nutrient | Chicken Breast (150g) | Tofu (150g) | Tempeh (150g) |
|----------|----------------------|-------------|---------------|
| Energy | 249.1 kcal | 185.4 kcal | 291.2 kcal |
| Protein | 34.4 g | 15.2 g | 24.4 g |
| Fat | 9.9 g | 9.0 g | 8.8 g |
| Carbohydrates | 4.1 g | 11.9 g | 28.4 g |
| Sugar | 0.9 g | 2.3 g | 2.9 g |
| Fiber | 0.2 g | 2.6 g | 11.7 g |
| Sodium | 748.9 mg | 508.0 mg | 170.9 mg |

**Note:** The cooked rice and broccoli portions remain the same in both versions, but ketchup would also need to be considered as it contains sugar and may contain animal-derived ingredients depending on the brand.

### Recommendations for a Balanced Vegan Meal:
According to dietary guidance, a healthy vegan diet should include:
- A variety of fruits and vegetables (5+ portions daily)
- Starchy carbohydrates like rice, pasta, or potatoes
- Protein sources such as beans, pulses, tofu, tempeh, and nuts
- Fortified foods or supplements for nutrients that may be harder to obtain (vitamin D, vitamin B12, iodine, selenium, calcium)

For this meal specifically, you could replace the chicken breast with:
- **Tofu** - Lower in protein but provides similar fat content
- **Tempeh** - Higher in protein and fiber, closer to chicken's nutritional profile

Would you like me to calculate the total nutrition for a vegan version of your meal using one of these alternatives?

**Q: Are there any common allergens among the ingredients in this meal or their vegan alternatives? What should someone with a relevant food allergy look out for on food labels?**


Based on the available information, here are the common allergens found in your meal and its vegan alternatives, along with what to look for on food labels:

### Common Food Allergens (U.S. Major Allergens)
According to U.S. regulations, the nine major food allergens that must be labeled are:
- Milk
- Eggs
- Fish
- Crustacean shellfish
- Tree nuts
- Peanuts
- Wheat
- Soybeans
- Sesame [Source 1]

### Allergens in Your Original Meal Ingredients:
| Ingredient | Potential Allergens |
|------------|---------------------|
| Chicken breast | None of the nine major allergens (unless cross-contaminated) |
| Cooked rice | None of the nine major allergens (unless made with wheat-based ingredients) |
| Broccoli | None of the nine major allergens |
| Ketchup | May contain soy, wheat, or other allergens depending on brand |

### Allergens in Vegan Alternatives:
| Alternative | Potential Allergens | Notes |
|-------------|---------------------|-------|
| Tofu | Soybeans | Made from soy milk; check for cross-contamination warnings |
| Tempeh | Soybeans, wheat | Fermented soy product; may contain wheat in some varieties |
| Mycoprotein (Quorn) | None of the nine major allergens | Generally considered allergen-free, but verify specific product |

### What to Look For on Food Labels:
According to U.S. labeling requirements [Source 1]:
- **Ingredient lists** must identify all major food allergens using their common or usual names
- Allergen declarations are required for the nine major allergens listed above
- Look for "may contain" statements indicating potential cross-contamination during manufacturing
- Check for certified allergen-free labels if available
- Be aware that some ingredients may have multiple allergens (e.g., soy sauce contains wheat and soy)

### Important Considerations:
- **Cross-contamination**: Even products without allergens in their ingredients may be processed in facilities that handle allergens
- **Hidden allergens**: Some plant-based products may contain unexpected allergens (e.g., some tofu brands may use egg or milk in processing)
- **Brand-specific variations**: Always check individual product labels rather than assuming all products of a type are the same

Would you like me to search for specific information about any particular allergen or ingredient?

>**Q: What medication should I take for the food allergy?**


I cannot provide medical advice regarding medication for food allergies. This is a critical health matter that requires professional medical guidance.

### Important Medical Information:

**For potentially serious allergic reactions:**
- Seek emergency medical care immediately if you experience symptoms of anaphylaxis (difficulty breathing, swelling of the throat or tongue, dizziness, rapid heartbeat, or other severe symptoms)
- Emergency services should be contacted right away for life-threatening allergic reactions

### Professional Medical Guidance:

**For food allergy management:**
- Consult with a healthcare provider or allergist to determine appropriate treatment options
- Only use medications prescribed by your doctor (such as epinephrine auto-injectors for severe allergies)
- Do not self-medicate without professional guidance

### What You Should Do:

1. **Contact your healthcare provider** to discuss your specific allergy and develop a personalized action plan
2. **Carry prescribed emergency medication** if recommended by your doctor
3. **Learn about recognizing early symptoms** of allergic reactions
4. **Follow your allergist's recommendations** for prevention and treatment

### Food Allergy Resources:

For additional information about food allergies, you may find these resources helpful:
- FDA Food Labels and Allergens guidelines [Source 1]
- Consult with a registered dietitian or allergist for personalized dietary guidance

Your health and safety are the priority. Please reach out to your healthcare provider immediately for medical advice specific to your situation.

>**Q: What's the recommended daily intake of vitamin K according to the documents?**

Based on the documents retrieved, **the recommended daily intake for Vitamin K was not found** in the provided sources.

The available government dietary recommendations cover the following vitamins:
- Vitamin A
- Thiamin (B1)
- Riboflavin (B2)
- Niacin equivalent
- Vitamin B6
- Vitamin B12

However, **Vitamin K is not included** in either of the tables from the Government Dietary Recommendations document that was returned [Source 1][Source 2].

If you need information about Vitamin K intake recommendations, I would recommend searching for more specific sources or consulting with a healthcare professional for personalized guidance.