# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import json
from io import BytesIO
import math

st.set_page_config(page_title="PetPalChef — Pet-Safe Recipe Recommender", layout="centered")

# ---------------------------
# Helpers
# ---------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

DATA_DIR = "data"
RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")
HARMFUL_PATH = os.path.join(DATA_DIR, "harmful_ingredients.json")
NUTR_PATH = os.path.join(DATA_DIR, "nutrition.json")

recipes = load_json(RECIPES_PATH)
harmful_map = load_json(HARMFUL_PATH)
nutrition_db = load_json(NUTR_PATH)

# Reference images folder (one subfolder per class, optional)
REF_DIR = "reference_images"

def image_to_hist(im: Image.Image, size=(128,128)):
    im = im.convert("RGB").resize(size)
    # 8-bit per channel histogram, but reduce bins to keep vector short
    # Use 16 bins per channel -> 48-dim vector
    arr = np.array(im)
    hist = []
    for c in range(3):
        channel = arr[:,:,c].ravel()
        h, _ = np.histogram(channel, bins=16, range=(0,255))
        hist.append(h)
    hist = np.concatenate(hist).astype(float)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist

def build_reference_histograms(ref_dir=REF_DIR):
    """
    Compute average histogram per class (folder).
    Returns dict: class -> hist vector
    """
    if not os.path.exists(ref_dir):
        return {}
    class_hists = {}
    for cls in os.listdir(ref_dir):
        cls_path = os.path.join(ref_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        hists = []
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    im = Image.open(os.path.join(cls_path, fname))
                    h = image_to_hist(im)
                    hists.append(h)
                except Exception:
                    continue
        if hists:
            avg = np.mean(hists, axis=0)
            if avg.sum() > 0:
                avg = avg / avg.sum()
            class_hists[cls] = avg
    return class_hists

def predict_by_histogram(uploaded_im, class_hists):
    """
    Compare uploaded image histogram to class histograms using cosine similarity.
    Returns best_class, confidence (0-1), and ordered list of (class, score).
    """
    if not class_hists:
        return None, 0.0, []
    h = image_to_hist(uploaded_im)
    sims = []
    for cls, ch in class_hists.items():
        # cosine similarity
        denom = (np.linalg.norm(h) * np.linalg.norm(ch))
        if denom == 0:
            score = 0
        else:
            score = float(np.dot(h, ch) / denom)
        sims.append((cls, score))
    sims.sort(key=lambda x: x[1], reverse=True)
    best, best_score = sims[0]
    # map cosine (0..1) to confidence-like value
    conf = float(best_score)
    return best, conf, sims

def check_harmful(food_key, pet):
    """
    Returns (status, list_of_reasons)
    status: 'safe', 'caution', 'unsafe'
    """
    reasons = []
    for ingredient, foods in harmful_map.items():
        if food_key in foods:
            reasons.append((ingredient, harmful_map[ingredient][food_key] if isinstance(harmful_map[ingredient], dict) and food_key in harmful_map[ingredient] else None))
    # alternative simpler approach: harmful_map maps ingredient -> list of foods
    # but our harmful structure supports vulnerability per pet in recipe reason
    # we derive status by presence in harmful lists for that food
    # We'll check recipes[food_key] for a 'safe_for' field if present
    recipe_entry = recipes.get(food_key)
    if recipe_entry and "safe_for" in recipe_entry:
        safe_for = recipe_entry["safe_for"]
        pet_safe = safe_for.get(pet.lower(), True)
        if pet_safe:
            return "safe", reasons
        else:
            return "unsafe", reasons
    # fallback simple rule:
    # if any harmful ingredient lists include the food, mark unsafe
    for ing, foods in harmful_map.items():
        if isinstance(foods, list) and food_key in foods:
            return "unsafe", reasons
        if isinstance(foods, dict) and food_key in foods:
            # foods might map to dict of pet flags; mark unsafe if True
            return "unsafe", reasons
    return "caution", reasons

def format_reason(status, reasons, food_key):
    if status == "safe":
        return "✅ This food is generally considered safe for the selected pet."
    if status == "unsafe":
        text = "❌ This food appears unsafe. Possible harmful items found: "
        if reasons:
            text += ", ".join([r[0] for r in reasons])
        else:
            # try recipe reason
            rec = recipes.get(food_key)
            if rec and "reason" in rec:
                text += rec["reason"]
            else:
                text += "unknown risk"
        return text
    return "⚠️ Consider caution — check ingredients and serve plain/unseasoned."

# ---------------------------
# UI (pastel)
# ---------------------------
st.markdown(
    """
    <style>
    .main { background-color: #fffaf6; }
    .stApp { background-color: #fffaf6; }
    .title { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .card { background: linear-gradient(135deg, #fff, #ffeef8); border-radius: 12px; padding: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .small { color: #666; font-size:12px;}
    </style>
    """, unsafe_allow_html=True
)

st.image("https://raw.githubusercontent.com/hasyaaddini/termprojectreal/main/logo_placeholder.png" if os.path.exists("logo.png") else "https://i.imgur.com/6RMhx.gif", width=120)

st.title("🍳 PetPalChef — Pet-Safe Recipe Recommender")
st.write("Upload a photo of human food and get pet-safe alternatives, warnings, and a tiny nutrition summary. 💖")

with st.sidebar:
    st.header("Settings")
    pet = st.selectbox("Pet Type", ["Cat", "Dog"])
    st.write("UI theme: Cute Pastel")
    st.markdown("---")
    st.markdown("**Tips**:\n- Upload a clear photo of a single dish\n- If detection fails, try typing the food name below")

st.markdown("### Upload your food photo")
uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

typed_name = st.text_input("Or type food name (fallback)")

# Build reference histograms now (fast for small sets)
class_hists = build_reference_histograms(REF_DIR)

clicked = st.button("Find pet-safe recipe")

if clicked:
    # if user uploaded image, classify by histogram; else fallback to typed name
    if uploaded is not None:
        try:
            img = Image.open(BytesIO(uploaded.read()))
        except Exception:
            st.error("Couldn't read the uploaded file. Try a different image.")
            st.stop()
        st.image(img, caption="Uploaded image", use_column_width=False, width=300)
        best, conf, all_sims = predict_by_histogram(img, class_hists)
        if best is None:
            st.warning("No reference images found for automatic detection — please type the food name below.")
            food_key = typed_name.strip().lower()
            conf = 0.0
        else:
            food_key = best.lower()
            st.markdown(f"**Detected:** {food_key.capitalize()} (confidence {conf:.2f})")
    else:
        if typed_name.strip() == "":
            st.warning("Please upload an image or type a food name.")
            st.stop()
        food_key = typed_name.strip().lower()
        st.markdown(f"**Using typed food name:** {food_key.capitalize()}")
        conf = 0.0

    # normalize food_key mapping (map common synonyms)
    synonyms = {
        "pizza": "pizza", "salad":"salad", "burger":"burger", "ramen":"ramen",
        "pasta":"pasta", "cookie":"cookie", "chocolate cake":"chocolate_cake",
        "chocolate":"chocolate_cake", "cake":"chocolate_cake", "fried chicken":"fried chicken",
        "chicken":"cooked_chicken", "salmon":"salmon", "steak":"steak", "rice":"plain_rice"
    }
    fk = synonyms.get(food_key, food_key)
    food_key = fk

    # Safety check
    status, reasons = check_harmful(food_key, pet)
    reason_text = format_reason(status, reasons, food_key)

    # Show safety badge
    st.markdown("---")
    st.markdown("### Safety")
    if status == "safe":
        st.success(f"Safe for {pet}s ✅")
    elif status == "unsafe":
        st.error(f"Unsafe for {pet}s ❌")
    else:
        st.warning(f"Caution for {pet}s ⚠️")

    st.write(reason_text)

    # Show nutrition snippet (if available)
    st.markdown("### Nutrition (approx.)")
    nut = nutrition_db.get(food_key)
    if nut:
        st.write(f"Calories: {nut.get('calories','N/A')} kcal")
        st.write(f"Protein: {nut.get('protein','N/A')} g")
        st.write(f"Fat: {nut.get('fat','N/A')} g")
        st.write(f"Carbs: {nut.get('carbs','N/A')} g")
    else:
        st.info("No nutrition data available for this item in the demo dataset.")

    # Recipes suggestions
    st.markdown("### Pet-safe alternatives")
    entry = recipes.get(food_key)
    if entry and "alternatives" in entry:
        for alt in entry["alternatives"]:
            r = entry["recipes"].get(alt) if "recipes" in entry else None
            if r is None:
                # maybe alt is a key in global recipes
                r = recipes.get(alt)
            if r:
                st.markdown(f"**{r.get('title','Alternative')}**")
                st.write("Ingredients:")
                for ing in r.get("ingredients", []):
                    st.write("- " + ing)
                st.write("Steps:")
                for s in r.get("steps", []):
                    st.write("- " + s)
                st.markdown("---")
    else:
        # fallback: try to find similar recipe by simple substring match
        found = False
        for k, v in recipes.items():
            if k in food_key or food_key in k:
                found = True
                st.markdown(f"**{v.get('pet_safe','Alternative')}**")
                if "ingredients" in v:
                    st.write("Ingredients:")
                    for ing in v.get("ingredients", []):
                        st.write("- " + ing)
                break
        if not found:
            st.info("No direct alternative recipe found in demo dataset. Please see README for how to add more recipes.")

    st.markdown("---")
    st.caption("Note: This is a demo tool. Always verify pet-safety with a veterinarian for your pet's special needs.")
