import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "sponsor_name": "ACME Bio",
    "trials": [
        {"start_date":"2016-01-10","phase":"Phase 1","enrollment":30,"allocation":"Randomized","masking":"Double","primary_purpose":"Treatment","intervention_types":"{Drug}","overall_status":"Completed"},
        {"start_date":"2018-04-15","phase":"Phase 2","enrollment":120,"allocation":"Randomized","masking":"Quadruple","primary_purpose":"Treatment","intervention_types":"{Drug}","overall_status":"Terminated"},
        {"start_date":"2021-09-01","phase":"Phase 2/Phase 3","enrollment":220,"allocation":"Randomized","masking":"Triple","primary_purpose":"Treatment","intervention_types":"{Drug}","overall_status":"Completed"}
    ]
}

try:
    r = requests.post(url, json=payload, timeout=30)
    print("Status:", r.status_code)
    print("JSON:", r.json())
except Exception as e:
    print("Error:", e)
    if hasattr(e, "response") and e.response is not None:
        print("Body:", e.response.text)
