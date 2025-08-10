# scripts/test_client.py
import requests, json

URL = "http://127.0.0.1:8000/predict"

examples = json.load(open("data/examples/sample_histories.json", "r", encoding="utf-8"))
for hist in examples:
    r = requests.post(URL, json=hist)
    print(f"[{hist['sponsor_name']}] {r.status_code} → {r.json()}")
