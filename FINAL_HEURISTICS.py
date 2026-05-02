"""
STEP 2 — Heuristic Scorer (Literature-Grounded)
=================================================
Scores each screen on 3 dimensions using weighted continuous metrics.

Literature:
  - Usability: Harrison et al. (2013) — severity-weighted signals
  - Layout Quality: Miniukovich & De Angeli (2015) — normalized continuous metrics
  - Visual Complexity: Deka et al. (2017) — pixel-ratio density
  - Cognitive Load: Miller (1956) — 7±2 items threshold
  - Heuristic Severity: Nielsen (1994)

Output: heuristic_scores_15k.csv
Columns: screen_id, usability, layout_quality, visual_complexity, screen_mean
All values as floats (e.g., 3.75, not 4)
"""

import os
import json
import math
import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
RICO_DIR = r"E:\R3_Reserach\rico_dataset"
SELECTED_CSV = r"E:\R3_Reserach\selected_15k.csv"
OUTPUT_CSV = r"E:\R3_Reserach\heuristic_scores_15k.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_json(screen_id):
    """Load JSON and return UI root."""
    path = os.path.join(RICO_DIR, str(screen_id) + ".json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["activity"]["root"]


def get_all_elements(node, result=None):
    """Flatten UI tree. Skip None nodes."""
    if result is None:
        result = []
    if node is None:
        return result
    result.append(node)
    for child in node.get("children", []):
        if child is not None:
            get_all_elements(child, result)
    return result


def get_visible_elements(elements):
    """Only elements visible to user."""
    return [e for e in elements if e.get("visible-to-user", False)]


def get_nesting_depth(node, depth=0):
    """Maximum depth of UI tree."""
    if node is None:
        return depth
    children = [c for c in node.get("children", []) if c is not None]
    if not children:
        return depth
    return max(get_nesting_depth(c, depth + 1) for c in children)


def bounds_area(bounds):
    """Pixel area from bounds."""
    if not bounds or len(bounds) < 4:
        return 0
    w = abs(bounds[2] - bounds[0])
    h = abs(bounds[3] - bounds[1])
    return w * h


def screen_area(root):
    """Total screen area."""
    b = root.get("bounds", [0, 0, 1080, 1920])
    return bounds_area(b) or (1080 * 1920)


def normalize(value, min_val, max_val):
    """Normalize to [0, 1] range."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def scale_to_5(normalized, invert=False):
    """Convert [0,1] to [1,5]. Returns float with 2 decimals."""
    if invert:
        normalized = 1.0 - normalized
    return round(1.0 + normalized * 4.0, 2)


def count_overlaps(elements):
    """Count overlapping clickable element pairs (Harrison et al. 2013)."""
    clickable = [e for e in elements
                 if e.get("clickable") and e.get("visible-to-user")
                 and e.get("bounds") and len(e["bounds"]) == 4]
    count = 0
    for i in range(len(clickable)):
        for j in range(i + 1, len(clickable)):
            b1, b2 = clickable[i]["bounds"], clickable[j]["bounds"]
            if b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]:
                count += 1
    return count


# ════════════════════════════════════════════════════════════════════════════
# DIMENSION 1: USABILITY — Harrison et al. (2013) severity-weighted signals
# ════════════════════════════════════════════════════════════════════════════

def score_usability(root, all_elements):
    """Usability score [1-5] using severity weights from Harrison et al. 2013."""
    visible = get_visible_elements(all_elements)
    clickable = [e for e in visible if e.get("clickable")]

    # Signal 1: Touch target size (weight 0.30 — severity 3/4)
    if clickable:
        adequate = sum(1 for e in clickable
                       if e.get("bounds") and len(e["bounds"]) == 4
                       and abs(e["bounds"][2] - e["bounds"][0]) >= 48
                       and abs(e["bounds"][3] - e["bounds"][1]) >= 48)
        touch_score = adequate / len(clickable)
    else:
        touch_score = 0.5

    # Signal 2: Cognitive load (weight 0.25 — Miller 1956, 7±2 threshold)
    n_clickable = len(clickable)
    if n_clickable <= 7:
        cognitive_score = 1.0
    else:
        cognitive_score = math.exp(-0.15 * (n_clickable - 7))
        cognitive_score = max(0.0, cognitive_score)

    # Signal 3: Text presence (weight 0.15 — Nielsen heuristic #6)
    has_text = any(
        ("Text" in e.get("class", "") or bool(e.get("text", "").strip()))
        for e in visible
    )
    text_score = 1.0 if has_text else 0.0

    # Signal 4: Element overlap (weight 0.30 — severity 4/4)
    overlaps = count_overlaps(visible)
    if overlaps == 0:
        overlap_score = 1.0
    else:
        overlap_score = max(0.0, 1.0 - (overlaps * 0.25))

    weighted = (0.30 * touch_score + 0.25 * cognitive_score + 
                0.15 * text_score + 0.30 * overlap_score)
    return scale_to_5(weighted)


# ════════════════════════════════════════════════════════════════════════════
# DIMENSION 2: LAYOUT QUALITY — Miniukovich & De Angeli (2015) continuous metrics
# ════════════════════════════════════════════════════════════════════════════

def score_layout_quality(root, all_elements):
    """Layout quality score [1-5] using normalized continuous metrics."""
    visible = get_visible_elements(all_elements)
    total_area = screen_area(root)

    # Signal 1: Whitespace ratio (weight 0.35 — optimal 0.3-0.6)
    elem_area = sum(bounds_area(e.get("bounds", [])) for e in visible if e.get("bounds"))
    raw_whitespace = 1.0 - (elem_area / total_area) if total_area > 0 else 0.5
    optimal = 0.40
    whitespace_score = max(0.0, 1.0 - abs(raw_whitespace - optimal) / optimal)

    # Signal 2: Alignment consistency (weight 0.35)
    left_edges = [e["bounds"][0] for e in visible 
                  if e.get("bounds") and len(e["bounds"]) == 4]
    if len(left_edges) > 1:
        from collections import Counter
        edge_counts = Counter(left_edges)
        aligned = sum(1 for x in left_edges if edge_counts[x] > 1)
        alignment_score = aligned / len(left_edges)
    else:
        alignment_score = 0.5

    # Signal 3: Overlap penalty (weight 0.30)
    overlaps = count_overlaps(visible)
    overlap_score = max(0.0, 1.0 - (overlaps * 0.20))

    weighted = (0.35 * whitespace_score + 0.35 * alignment_score + 0.30 * overlap_score)
    return scale_to_5(weighted)


# ════════════════════════════════════════════════════════════════════════════
# DIMENSION 3: VISUAL COMPLEXITY — Deka et al. (2017) pixel-ratio density
# ════════════════════════════════════════════════════════════════════════════

def score_visual_complexity(root, all_elements):
    """Visual complexity score [1-5] — higher = more complex."""
    visible = get_visible_elements(all_elements)
    total_area = screen_area(root)
    depth = get_nesting_depth(root)

    # Signal 1: Element density (weight 0.40 — Deka et al. 2017 primary metric)
    elem_area = sum(bounds_area(e.get("bounds", [])) for e in visible if e.get("bounds"))
    density = min(1.0, elem_area / total_area) if total_area > 0 else 0.5

    # Signal 2: Element count normalized (weight 0.35)
    n = len(visible)
    count_norm = normalize(n, 4, 150)

    # Signal 3: Structural depth (weight 0.25)
    depth_norm = normalize(depth, 1, 20)

    weighted = (0.40 * density + 0.35 * count_norm + 0.25 * depth_norm)
    return scale_to_5(weighted)  # NOT inverted — higher complexity = higher score


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def score_screen(screen_id):
    """Score one screen, return dict with all values."""
    try:
        root = load_json(screen_id)
        elements = get_all_elements(root)
        return {
            "screen_id": screen_id,
            "usability": score_usability(root, elements),
            "layout_quality": score_layout_quality(root, elements),
            "visual_complexity": score_visual_complexity(root, elements),
        }
    except Exception as e:
        print(f"  ⚠️ Failed {screen_id}: {e}")
        return None


def run_all():
    if not os.path.exists(SELECTED_CSV):
        print(f"❌ ERROR: {SELECTED_CSV} not found. Run step1 first.")
        return

    df_in = pd.read_csv(SELECTED_CSV, dtype={"screen_id": str})
    screen_ids = df_in["screen_id"].tolist()
    print(f"📊 Loading {len(screen_ids):,} screens from {SELECTED_CSV}")
    print(f"📁 Looking for JSON files in: {RICO_DIR}\n")

    results = []
    failed = []

    for sid in tqdm(screen_ids, desc="Heuristic Scorer"):
        r = score_screen(sid)
        if r:
            results.append(r)
        else:
            failed.append(sid)

    df_out = pd.DataFrame(results)

    # Calculate screen mean (average of 3 dimensions)
    df_out["screen_mean"] = df_out[["usability", "layout_quality", "visual_complexity"]].mean(axis=1).round(4)

    # Reorder columns: screen_id first, then dimensions, then mean
    df_out = df_out[["screen_id", "usability", "layout_quality", "visual_complexity", "screen_mean"]]

    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"✅ Scored : {len(results):,} screens")
    print(f"❌ Failed : {len(failed):,} screens")
    print(f"💾 Saved  → {OUTPUT_CSV}")
    print(f"{'='*50}")

    if len(results) > 0:
        print("\n📊 Score Distributions (literature-grounded):")
        print(df_out[["usability", "layout_quality", "visual_complexity", "screen_mean"]].describe().round(4))

        print("\n📊 Precision check (unique values per dimension):")
        for col in ["usability", "layout_quality", "visual_complexity"]:
            print(f"   {col}: {df_out[col].nunique()} unique values")


if __name__ == "__main__":
    run_all()