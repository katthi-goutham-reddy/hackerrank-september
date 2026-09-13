import os
import sys
import json
import base64
import re
import urllib.request
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
from main import REPO_ROOT, DATASET_DIR, IMAGE_AMOUNT_LOOKUP, LLMClient

llm = LLMClient()
print("Provider:", llm.provider, "Model:", llm.model_name)

images_df = pd.read_csv(DATASET_DIR / "images.csv")
events_df = pd.read_csv(DATASET_DIR / "financial_events.csv")
blank_events = events_df[events_df['amount'].isna()]

def extract_image_amount(image_path: Path, prompt: str, strict: bool = False):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    if llm.provider == "OpenAI":
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm.openai_key}"
        }
        text_content = prompt if not strict else "Extract the total amount / net pay / grand total / balance due from this image. Respond with ONLY the number (e.g. 4365000 or 100000.00), no words, no currency symbols."
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            usage = data.get('usage', {})
            in_tok = usage.get('prompt_tokens', 0)
            out_tok = usage.get('completion_tokens', 0)
            llm._record_usage(in_tok, out_tok)
            return data['choices'][0]['message']['content'].strip()

# Test on image_01 through image_03
for idx, row in blank_events.head(3).iterrows():
    eid = row['event_id']
    img_row = images_df[images_df['related_event_id'] == eid].iloc[0]
    img_file = DATASET_DIR / "media" / "images" / f"{img_row['image_id']}.png"
    prompt = f"What is the total monetary amount, net pay, grand total, balance due, or amount paid shown on this receipt/invoice/document? Respond with ONLY the numeric value."
    res = extract_image_amount(img_file, prompt)
    print(f"[{eid}] ({img_row['image_id']}): Raw LLM output = '{res}', Expected = {IMAGE_AMOUNT_LOOKUP.get(eid)}")
