import os
import sys
import json
import base64
import urllib.request
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
from main import DATASET_DIR, LLMClient

llm = LLMClient()
print("Loaded LLMClient: provider =", llm.provider, "gemini_key present =", bool(llm.gemini_key))

images_df = pd.read_csv(DATASET_DIR / "images.csv")
img_row = images_df.iloc[0]
img_path = DATASET_DIR / "media" / "images" / f"{img_row['image_id']}.png"

with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={llm.gemini_key}"
headers = {"Content-Type": "application/json"}
payload = {
    "contents": [{
        "parts": [
            {"text": "Extract the net pay / take home pay / total amount from this payslip. Respond with ONLY the numeric digits and decimal, e.g. 4365000."},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": img_b64
                }
            }
        ]
    }],
    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 100}
}
req = urllib.request.Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("Gemini response text:", data['candidates'][0]['content']['parts'][0]['text'])
except Exception as e:
    print("Gemini error:", e)
