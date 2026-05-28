# ==============================================================================
# Script: debug_api.py
# Purpose: Sanity check for OpenCorporates API raw responses
# ==============================================================================

import requests
import json
from opencorporates_network_mapper import OpenCorporatesClient

# 1. Initialize client
client = OpenCorporatesClient()

# 2. Pick a "known" entity that definitely has subsidiaries (e.g., Apple)
test_name = "Apple Inc."
test_ticker = "AAPL"
# Mock a set of tax havens (use 'ie' for Ireland, a common Apple base)
test_havens = {"ie", "ky", "bm", "lu"} 

print(f"--- Debugging API response for: {test_name} ---")

# 3. Simulate the search (using the logic from your refactored mapper)
# We manually trigger the search_subsidiaries method
search_term = client.get_core_brand_name(test_name, test_ticker)
print(f"Debug: Searching with keyword: '{search_term}'")

params = {'q': search_term, 'per_page': 5} # Limit to 5 results for readability
response = client.session.get(client.base_url, params=params)

if response.status_code == 200:
    data = response.json()
    companies = data.get('results', {}).get('companies', [])
    
    print(f"Found {len(companies)} companies. Raw data preview:")
    
    # 4. Print the raw structure of the first company returned
    if companies:
        first_comp = companies[0].get('company', {})
        print(json.dumps(first_comp, indent=4))
        
        # Verify jurisdiction code
        jur = first_comp.get('jurisdiction_code', 'N/A')
        print(f"\nVerification -> Name: {first_comp.get('name')}")
        print(f"Verification -> Jurisdiction Code: {jur}")
        
        # Check if this matches your filter logic
        prefix = jur.split('_')[0]
        print(f"Does it match your tax haven list ({test_havens})? {'YES' if prefix in test_havens else 'NO'}")
    else:
        print("Error: No companies found. The search keyword might be too strict.")
else:
    print(f"API Error: {response.status_code}")