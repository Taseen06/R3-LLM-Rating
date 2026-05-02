import os
import re
import time
import base64
import json
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
RICO_DIR = "/home/taseen06/Desktop/R3/rico_dataset"
SELECTED_CSV = "/home/taseen06/Desktop/R3/selected_15k.csv"
OUTPUT_JPEG_CSV = "/home/taseen06/Desktop/R3/llm_scores_15k_(gemma3)JPEG.csv"
OUTPUT_JSON_CSV = "/home/taseen06/Desktop/R3/llm_scores_15k_(gemma3)JSON.csv"

# Ollama Settings - Target PORT 11435
OLLAMA_URL = "http://localhost:11435/api/generate"
OLLAMA_MODEL = "gemma3:27b-cloud" 

REQUEST_DELAY = 0.8
BATCH_SAVE_INTERVAL = 5
RETRY_LIMIT = 2

# ============================================================
# PROMPT TEMPLATE
# ============================================================
PROMPT_TEMPLATE = (
    "You are a UI expert. Rate the {target_type} of this mobile app screen.\n"
    "USABILITY: How easily a user interacts with it.\n"
    "LAYOUT QUALITY: Organization, whitespace, and hierarchy.\n"
    "VISUAL COMPLEXITY: Density and overwhelming elements.\n\n"
    "Scale: 1.0 to 5.0. Provide exactly 2 decimal places.\n"
    "Output ONLY in this format: usability:X.XX, layout:X.XX, complexity:X.XX"
)

# ============================================================
# UTILITIES
# ============================================================

def load_image_b64(sid):
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(RICO_DIR, f"{sid}{ext}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return None

def extract_ui_hierarchy(sid):
    path = os.path.join(RICO_DIR, f"{sid}.json")
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        elements = []
        def traverse(node):
            if node and node.get("visible-to-user"):
                name = node.get("class", "").split(".")[-1]
                text = node.get("text", "")
                elements.append(f"{name}({text})" if text else name)
            children = node.get("children", [])
            if children:
                for child in children: traverse(child)
        traverse(data.get("activity", {}).get("root", {}))
        return "UI Structure: " + " -> ".join(elements[:25])
    except: return None

def parse_model_response(text):
    scores = {"usability": None, "layout_quality": None, "visual_complexity": None}
    patterns = {
        "usability": r"usability:?([\d\.]+)",
        "layout_quality": r"layout:?([\d\.]+)",
        "visual_complexity": r"complexity:?([\d\.]+)"
    }
    vals = []
    for key, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            try:
                v = round(float(match.group(1)), 2)
                scores[key] = v
                vals.append(v)
            except: pass
    scores["screen_mean"] = round(sum(vals)/len(vals), 2) if vals else None
    return scores

def call_ollama(prompt, image_b64=None):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 60}
    }
    if image_b64: payload["images"] = [image_b64]
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "")
    except: pass
    return ""

# ============================================================
# MAIN EXECUTION
# ============================================================

def run():
    if not os.path.exists(SELECTED_CSV):
        print("Input CSV not found.")
        return
        
    df = pd.read_csv(SELECTED_CSV, dtype={"screen_id": str})
    
    # --- AUTO-RESUME LOGIC ---
    jpeg_data, json_data = [], []
    processed_ids = set()

    if os.path.exists(OUTPUT_JPEG_CSV):
        try:
            old_df = pd.read_csv(OUTPUT_JPEG_CSV, dtype={"screen_id": str})
            jpeg_data = old_df.to_dict('records')
            processed_ids = set(old_df["screen_id"].unique())
            
            # Sync JSON data if it exists
            if os.path.exists(OUTPUT_JSON_CSV):
                json_data = pd.read_csv(OUTPUT_JSON_CSV, dtype={"screen_id": str}).to_dict('records')
            
            print(f"Resuming from screen {len(processed_ids)}...")
        except: pass

    remaining_df = df[~df['screen_id'].isin(processed_ids)]
    
    print(f"--- Starting Dual Pipeline with {OLLAMA_MODEL} ---")
    
    try:
        for i, (_, row) in enumerate(tqdm(remaining_df.iterrows(), total=len(remaining_df))):
            sid = row['screen_id']
            
            # 1. Process JPEG
            img_b64 = load_image_b64(sid)
            if img_b64:
                prompt_jpeg = PROMPT_TEMPLATE.format(target_type="VISUAL appearance (attached image)")
                resp_jpeg = call_ollama(prompt_jpeg, image_b64=img_b64)
                jpeg_data.append({"screen_id": sid, **parse_model_response(resp_jpeg)})
            
            # 2. Process JSON
            hierarchy = extract_ui_hierarchy(sid)
            if hierarchy:
                prompt_json = PROMPT_TEMPLATE.format(target_type="STRUCTURAL hierarchy") + f"\n\nData: {hierarchy}"
                resp_json = call_ollama(prompt_json)
                json_data.append({"screen_id": sid, **parse_model_response(resp_json)})

            # Batch Save (Fixed variable name: json_data)
            if (i + 1) % BATCH_SAVE_INTERVAL == 0:
                pd.DataFrame(jpeg_data).to_csv(OUTPUT_JPEG_CSV, index=False)
                pd.DataFrame(json_data).to_csv(OUTPUT_JSON_CSV, index=False)
            
            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving current progress...")

    # Final Save
    pd.DataFrame(jpeg_data).to_csv(OUTPUT_JPEG_CSV, index=False)
    pd.DataFrame(json_data).to_csv(OUTPUT_JSON_CSV, index=False)
    print(f"Success. Files saved.")

if __name__ == "__main__":
    run()