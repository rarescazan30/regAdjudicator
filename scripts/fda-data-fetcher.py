import os
import json
import requests

meds = {
    "aducanumab": "ADUHELM",
    "lecanemab": "LEQEMBI",
    "bevacizumab": "AVASTIN",
    "olaparib": "LYNPARZA",
}

os.makedirs("data/raw", exist_ok=True)

for generic_name, brand_name in meds.items():
    print(f"Fetching FDA data for {generic_name} ({brand_name})...")
    url = f'https://api.fda.gov/drug/label.json?search="{brand_name}"&limit=1'
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        out_path = f"data/raw/{generic_name}_fda.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  -> Saved to {out_path}")
    else:
        print(f"  -> Failed to fetch {generic_name} (HTTP {response.status_code})")
