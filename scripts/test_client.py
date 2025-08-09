import json, requests, sys

API = "http://127.0.0.1:8000/predict"

def run_one(sponsor_payload):
    r = requests.post(API, json=sponsor_payload, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    path = "data/examples/sponsor_samples.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        resp = run_one(item)
        print(f"[{resp.get('sponsor_name')}] prob={resp['prob_calibrated']:.3f} "
              f"label={resp['label']} thr={resp['threshold_used']:.3f}")

if __name__ == "__main__":
    main()
