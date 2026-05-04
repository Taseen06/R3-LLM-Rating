import os
import re
import asyncio
import json
import base64
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm.asyncio import tqdm
try:
    import google.genai as genai
    NEW_GENAI = True
except ImportError:
    import google.generativeai as genai
    NEW_GENAI = False

# Apply nest_asyncio if running in Jupyter/IPython (detects existing loop)
try:
    asyncio.get_running_loop()
    # If we reach here, there's already a loop running (Jupyter/IPython)
    import nest_asyncio
    nest_asyncio.apply()
except RuntimeError:
    # No running loop, regular script execution
    pass

# ============================================================
# CONFIGURATION
# ============================================================
RICO_DIR = "/home/taseen06/Desktop/R3/rico_dataset"
SELECTED_CSV = "/home/taseen06/Desktop/R3/selected_15k.csv"
OUTPUT_JPEG_CSV = "/home/taseen06/Desktop/R3/llm_scores_15k_(gemini-3-flash)JPEG.csv"
OUTPUT_JSON_CSV = "/home/taseen06/Desktop/R3/llm_scores_15k_(gemini-3-flash)JSON.csv"

API_KEYS = ["AIzaSyD4__tsMMFuoQp7AH_s8Dl9aDE-z8RiQHg"]

MODEL_NAME = "gemini-3-flash-preview" 

MAX_PARALLEL_TASKS = 15  # Maximum concurrent screen tasks
BATCH_SIZE = 5          # Process up to 15 screens per minute
OUTPUT_COLUMNS = ["screen_id", "usability", "layout_quality", "visual_complexity", "screen_mean"]

# ============================================================
# PROMPT
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

class APIRotator:
    def __init__(self, keys, screens_per_minute=15, key_limit_per_minute=5, key_switch_minutes=15):
        self.keys = keys
        self.screens_per_minute = screens_per_minute
        self.key_limit_per_minute = key_limit_per_minute
        self.key_switch_minutes = key_switch_minutes
        self.lock = asyncio.Lock()

        self.period_start = datetime.now(timezone.utc)
        self.period_primary = 0
        self.reset_minute(restart_from_first=True)

    def reset_minute(self, restart_from_first=False):
        self.minute_start = datetime.now(timezone.utc)
        self.minute_end = self.minute_start + timedelta(minutes=1)
        self.total_screens = 0
        self.key_counts = [0] * len(self.keys)
        if restart_from_first:
            self.period_primary = 0
            self.period_start = self.minute_start
        self.primary_index = self.period_primary

    def rotate_period_if_needed(self):
        now = datetime.now(timezone.utc)
        if now - self.period_start >= timedelta(minutes=self.key_switch_minutes):
            self.period_start = now
            self.period_primary = (self.period_primary + 1) % len(self.keys)
            self.primary_index = self.period_primary
            print(f"[API ROTATION] switched primary API key to index {self.period_primary}")

    def _choose_key(self):
        now = datetime.now(timezone.utc)
        if now >= self.minute_end:
            self.reset_minute(restart_from_first=False)

        self.rotate_period_if_needed()

        if self.total_screens >= self.screens_per_minute:
            return None

        for offset in range(len(self.keys)):
            idx = (self.primary_index + offset) % len(self.keys)
            if self.key_counts[idx] < self.key_limit_per_minute:
                self.key_counts[idx] += 1
                self.total_screens += 1
                self.primary_index = idx
                return self.keys[idx]

        return None

    async def get_key(self):
        async with self.lock:
            key = self._choose_key()
            if key is not None:
                return key

            wait_for = max(0, (self.minute_end - datetime.now(timezone.utc)).total_seconds())
            if wait_for > 0:
                print(f"[RATE LIMIT] all API keys exhausted for this minute; idling for {wait_for:.1f}s and restarting from first API key")
                await asyncio.sleep(wait_for)

            self.reset_minute(restart_from_first=True)
            self.rotate_period_if_needed()
            return self._choose_key()

    async def prepare_batch(self, batch_size):
        while True:
            async with self.lock:
                now = datetime.now(timezone.utc)
                if now >= self.minute_end:
                    self.reset_minute(restart_from_first=False)
                self.rotate_period_if_needed()
                available = self.screens_per_minute - self.total_screens
                if batch_size <= available:
                    return
                wait_for = max(0, (self.minute_end - now).total_seconds())

            if wait_for > 0:
                print(f"[BATCH WAIT] waiting {wait_for:.1f}s for next minute window")
                try:
                    await asyncio.sleep(wait_for)
                except RuntimeError:
                    # Handle Python 3.12 context variable issues
                    import time
                    time.sleep(wait_for)
                # Reset for next iteration
                async with self.lock:
                    self.reset_minute(restart_from_first=True)
                    self.rotate_period_if_needed()


def parse_response(text):
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

def get_hierarchy(sid):
    path = os.path.join(RICO_DIR, f"{sid}.json")
    if not os.path.exists(path): return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        elements = []
        def traverse(node):
            if node and node.get("visible-to-user"):
                cls = node.get("class", "").split(".")[-1]
                txt = node.get("text", "")
                elements.append(f"{cls}({txt})" if txt else cls)
            for child in node.get("children", []) or []: traverse(child)
        traverse(data.get("activity", {}).get("root", {}))
        return "UI Structure: " + " -> ".join(elements[:30])
    except: return None

def read_processed_ids(path):
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, dtype={"screen_id": str})["screen_id"].dropna())
    except Exception as e:
        print(f"[WARN] Could not read existing output {path}: {e}", flush=True)
        return set()

def save_batch_results(path, new_results, selected_order, label):
    """Merge a completed batch into the output CSV immediately and safely."""
    if not new_results:
        print(f"\n[SAVE {label}] No new {label} results from this batch", flush=True)
        return 0

    try:
        new_df = pd.DataFrame(new_results)
        for col in OUTPUT_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[OUTPUT_COLUMNS]
        new_df["screen_id"] = new_df["screen_id"].astype(str)

        if os.path.exists(path):
            old_df = pd.read_csv(path, dtype={"screen_id": str})
            for col in OUTPUT_COLUMNS:
                if col not in old_df.columns:
                    old_df[col] = None
            combined = pd.concat([old_df[OUTPUT_COLUMNS], new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.drop_duplicates(subset=["screen_id"], keep="last")
        combined["_order"] = combined["screen_id"].map(selected_order)
        combined = combined.sort_values("_order", kind="stable").drop(columns=["_order"])
        combined.to_csv(path, index=False, columns=OUTPUT_COLUMNS)
        print(f"✓ Saved batch: {len(new_df)} new {label} rows, {len(combined)} total -> {path}", flush=True)
        return len(new_df)
    except Exception as e:
        print(f"✗ Failed to save {label} batch: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 0

# ============================================================
# ASYNC CORE
# ============================================================

async def judge_task(sid, rotator, semaphore, jpeg_list, json_list):
    async with semaphore:
        key = await rotator.get_key()

        if NEW_GENAI:
            # Use asyncio.to_thread for proper async handling, with error handling
            try:
                await asyncio.to_thread(_process_screen_sync, sid, key, jpeg_list, json_list)
            except RuntimeError as e:
                if "cannot enter context" in str(e):
                    # Fallback: run sync in current thread if context error
                    _process_screen_sync(sid, key, jpeg_list, json_list)
                else:
                    raise
        else:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(MODEL_NAME)
            await _process_screen_legacy(sid, model, jpeg_list, json_list)

def _process_screen_sync(sid, api_key, jpeg_list, json_list):
    """Synchronous API calls using thread pool to avoid async context conflicts."""
    import sys
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[ERROR] Failed to create client for {sid}: {e}", flush=True)
        return
    
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        p = os.path.join(RICO_DIR, f"{sid}{ext}")
        if os.path.exists(p):
            img_path = p
            break

    if img_path:
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    PROMPT_TEMPLATE.format(target_type="VISUAL appearance"),
                    {"parts": [{"inline_data": {"data": img_bytes, "mime_type": "image/jpeg"}}]}
                ]
            )
            result = {"screen_id": sid, **parse_response(response.text)}
            jpeg_list.append(result)
        except Exception as e:
            # Only log quota errors, not other API errors
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                pass  # Silent for quota errors
            else:
                print(f"[ERROR JPEG] {sid}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    else:
        print(f"[WARN] No image found for {sid}", flush=True)

    struct = get_hierarchy(sid)
    if struct:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[PROMPT_TEMPLATE.format(target_type="STRUCTURAL hierarchy") + f"\n\n{struct}"]
            )
            result = {"screen_id": sid, **parse_response(response.text)}
            json_list.append(result)
        except Exception as e:
            # Only log quota errors, not other API errors
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                pass  # Silent for quota errors
            else:
                print(f"[ERROR JSON] {sid}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    else:
        print(f"[WARN] No hierarchy found for {sid}", flush=True)

async def _process_screen_legacy(sid, model, jpeg_list, json_list):
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        p = os.path.join(RICO_DIR, f"{sid}{ext}")
        if os.path.exists(p):
            img_path = p
            break

    if img_path:
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            response = await model.generate_content_async([
                PROMPT_TEMPLATE.format(target_type="VISUAL appearance"),
                {'mime_type': 'image/jpeg', 'data': img_data}
            ])
            jpeg_list.append({"screen_id": sid, **parse_response(response.text)})
        except Exception:
            pass

    struct = get_hierarchy(sid)
    if struct:
        try:
            response = await model.generate_content_async(
                PROMPT_TEMPLATE.format(target_type="STRUCTURAL hierarchy") + f"\n\n{struct}"
            )
            json_list.append({"screen_id": sid, **parse_response(response.text)})
        except Exception:
            pass

async def main():
    if not os.path.exists(SELECTED_CSV):
        print("Input CSV not found!")
        return

    df = pd.read_csv(SELECTED_CSV, dtype={"screen_id": str})
    
    # Quick test to verify API is working
    print("\n[TEST] Verifying API with first screen...")
    test_sid = df["screen_id"].iloc[0]
    test_jpeg = []
    test_json = []
    _process_screen_sync(test_sid, API_KEYS[0], test_jpeg, test_json)
    print(f"[TEST] Results: JPEG={len(test_jpeg)}, JSON={len(test_json)}")
    
    if len(test_jpeg) == 0 and len(test_json) == 0:
        print("❌ API TEST FAILED - All calls returned errors (likely quota exceeded)")
        print("   Check your Google Cloud Console quota and billing")
        print("   Continuing anyway, but expect all batches to fail...\n")
    else:
        print("✅ API TEST PASSED - Proceeding with full processing\n")
    
    # Resume logic: a screen is complete only when both outputs already exist.
    jpeg_processed = read_processed_ids(OUTPUT_JPEG_CSV)
    json_processed = read_processed_ids(OUTPUT_JSON_CSV)
    processed = jpeg_processed & json_processed
    
    remaining = [sid for sid in df["screen_id"] if sid not in processed]
    selected_order = {sid: idx for idx, sid in enumerate(df["screen_id"].astype(str))}
    print(f"Processing {len(remaining)} screens...")
    print(f"Resume state: JPEG={len(jpeg_processed)}, JSON={len(json_processed)}, complete={len(processed)}")

    rotator = APIRotator(API_KEYS)
    semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
    saved_jpeg_rows = 0
    saved_json_rows = 0
    
    # Process in batches to save to disk regularly
    for i in range(0, len(remaining), BATCH_SIZE):
        batch_ids = remaining[i:i+BATCH_SIZE]
        jpeg_results = []
        json_results = []

        await rotator.prepare_batch(len(batch_ids))

        tasks = [judge_task(sid, rotator, semaphore, jpeg_results, json_results) for sid in batch_ids]
        await tqdm.gather(*tasks, desc=f"Batch {i//BATCH_SIZE + 1}")

        # Save immediately after every judged batch finishes.
        saved_jpeg_rows += save_batch_results(OUTPUT_JPEG_CSV, jpeg_results, selected_order, "JPEG")
        saved_json_rows += save_batch_results(OUTPUT_JSON_CSV, json_results, selected_order, "JSON")

    # Final summary
    total_processed = len(df) - len(remaining)
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY:")
    print(f"  Total screens in dataset: {len(df)}")
    print(f"  Already processed: {total_processed}")
    print(f"  Remaining to process: {len(remaining)}")
    print(f"  JPEG results saved: {os.path.exists(OUTPUT_JPEG_CSV)}")
    print(f"  JSON results saved: {os.path.exists(OUTPUT_JSON_CSV)}")
    print(f"{'='*60}")
    
    if len(remaining) > 0 and saved_jpeg_rows == 0 and saved_json_rows == 0:
        print("❌ API QUOTA EXCEEDED - All API calls failed with 429 RESOURCE_EXHAUSTED")
        print("   Solutions:")
        print("   1. Check Google Cloud Console for quota limits")
        print("   2. Upgrade your Gemini API plan")
        print("   3. Wait for quota reset (usually monthly)")
        print("   4. Use different API keys")
        print("   5. Reduce BATCH_SIZE or MAX_PARALLEL_TASKS")
        print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
