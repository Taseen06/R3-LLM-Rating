"""
STEP 1 — Select 15k Valid Screens from RICO
============================================
Filters all screens, keeps only valid ones, randomly samples 15,000.
Output: selected_15k.csv (screen_id, category, total_elements, clickable_elements)

Literature: RICO dataset paper (Deka et al. 2017) — quality filtering criteria
"""

import os
import json
import hashlib
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
RICO_DIR = r"E:\R3_Reserach\rico_dataset"
OUTPUT_CSV = r"E:\R3_Reserach\selected_15k.csv"
TARGET_N = 15000
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def get_root(data):
    """Extract UI root from RICO JSON: data['activity']['root']"""
    try:
        return data["activity"]["root"]
    except (KeyError, TypeError):
        return None


def count_elements(node, counts=None):
    """Count visible and clickable elements. Skip None children."""
    if node is None:
        return counts if counts else {"total": 0, "clickable": 0}
    if counts is None:
        counts = {"total": 0, "clickable": 0}

    if node.get("visible-to-user", False):
        counts["total"] += 1
        if node.get("clickable", False):
            counts["clickable"] += 1

    for child in node.get("children", []):
        if child is not None:
            count_elements(child, counts)

    return counts


def image_hash(path):
    """MD5 hash for duplicate detection."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_category(data):
    """Derive category from activity_name keywords."""
    activity = data.get("activity_name", "").lower()

    keywords = {
        "social": ["social", "chat", "message", "whatsapp", "telegram", "twitter", "facebook", "instagram"],
        "shopping": ["shop", "store", "amazon", "ebay", "market", "buy", "commerce"],
        "productivity": ["productivity", "note", "task", "todo", "calendar", "office"],
        "games": ["game", "play", "puzzle", "arcade", "chess"],
        "travel": ["travel", "flight", "hotel", "booking", "airbnb", "map", "uber"],
        "food": ["food", "recipe", "restaurant", "delivery", "cook"],
        "health": ["health", "fitness", "workout", "medical", "doctor"],
        "news": ["news", "article", "blog", "reader", "feed"],
        "finance": ["bank", "finance", "money", "pay", "wallet", "invest"],
        "utility": ["utility", "tool", "file", "manager", "cleaner", "launcher", "camera", "weather"],
    }

    for category, words in keywords.items():
        if any(w in activity for w in words):
            return category

    return "other"


def filter_all_screens():
    """Run through all screens, collect valid ones."""
    all_files = os.listdir(RICO_DIR)
    all_jsons = [f for f in all_files if f.endswith(".json")]
    valid_screens = []
    seen_hashes = set()
    skipped = defaultdict(int)

    print(f"Total JSON files found: {len(all_jsons):,}")
    print(f"Running quality checks...\n")

    for filename in tqdm(all_jsons, desc="Filtering"):
        screen_id = filename.replace(".json", "")
        json_path = os.path.join(RICO_DIR, filename)
        image_path = os.path.join(RICO_DIR, screen_id + ".jpg")

        # Check 1 — image exists
        if not os.path.exists(image_path):
            skipped["missing_image"] += 1
            continue

        # Check 2 — image not corrupted
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception:
            skipped["corrupted_image"] += 1
            continue

        # Check 3 — JSON valid, has activity.root
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            root = get_root(data)
            if root is None:
                skipped["no_activity_root"] += 1
                continue
        except Exception:
            skipped["bad_json"] += 1
            continue

        # Check 4 — bounds non-zero
        bounds = root.get("bounds", [0, 0, 0, 0])
        if (not bounds or len(bounds) < 4 or bounds == [0, 0, 0, 0] or
            (bounds[2] - bounds[0]) <= 0 or (bounds[3] - bounds[1]) <= 0):
            skipped["zero_bounds"] += 1
            continue

        # Check 5 & 6 — element counts (Min 3 visible, 1 clickable)
        counts = count_elements(root)
        if counts["total"] < 3:
            skipped["too_few_elements"] += 1
            continue
        if counts["clickable"] < 1:
            skipped["no_clickable"] += 1
            continue

        # Check 7 — duplicate image
        h = image_hash(image_path)
        if h in seen_hashes:
            skipped["duplicate"] += 1
            continue
        seen_hashes.add(h)

        # Passed
        valid_screens.append({
            "screen_id": screen_id,
            "category": get_category(data),
            "total_elements": counts["total"],
            "clickable_elements": counts["clickable"],
        })

    print(f"\n{'='*50}")
    print(f"VALID screens found: {len(valid_screens):,}")
    print(f"SKIPPED: {sum(skipped.values()):,}")
    for reason, count in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"  {reason:25s}: {count:,}")
    print(f"{'='*50}\n")

    return valid_screens


def random_sample_15k(valid_screens):
    """Randomly sample 15,000 screens from valid pool."""
    if len(valid_screens) <= TARGET_N:
        print(f"Only {len(valid_screens):,} valid screens available. Using all.")
        return valid_screens

    random.seed(RANDOM_SEED)
    sampled = random.sample(valid_screens, TARGET_N)
    print(f"Randomly sampled {TARGET_N:,} screens from {len(valid_screens):,} valid screens")
    return sampled


def main():
    valid_screens = filter_all_screens()

    if not valid_screens:
        print("No valid screens found. Check your RICO_DIR path.")
        return

    sampled_screens = random_sample_15k(valid_screens)

    df = pd.DataFrame(sampled_screens)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Saved {len(df):,} screens → {OUTPUT_CSV}")
    print(f"\n📊 Category breakdown:")
    print(df["category"].value_counts().to_string())
    print(f"\n📊 Element stats:")
    print(df["total_elements"].describe().round(2).to_string())


if __name__ == "__main__":
    main()