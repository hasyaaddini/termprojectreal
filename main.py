import json
from classify import classify_image

with open("recipes.json", "r") as f:
    recipes = json.load(f)

with open("harmful_ingredients.json", "r") as f:
    harmful_data = json.load(f)

def check_harmful(food):
    harmful_list = []
    for ingredient, foods in harmful_data.items():
        if food in foods:
            harmful_list.append(ingredient)
    return harmful_list

def main():
    img_path = input("Upload a food image (path): ")

    print("\nClassifying food...\n")
    food, prob = classify_image(img_path)
    print(f"Detected food: {food} (confidence {prob:.2f})\n")

    print("🐾 Pet-safe alternative:")
    print(recipes[food]["pet_safe"], "\n")

    harmful = check_harmful(food)
    if harmful:
        print("⚠️ Harmful ingredients for pets:")
        for ing in harmful:
            print(f"- {ing}")
    else:
        print("No harmful ingredients detected.")

if __name__ == "__main__":
    main()
