import os
import json
import urllib.request
import urllib.error

API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_TOKEN = os.getenv("HF_API_TOKEN", "")

def run_local_llm(prompt: str) -> str:
    if not HF_TOKEN:
        return "Fallback: Please configure a valid HF_API_TOKEN in your environment variables."

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 40,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(API_URL, data=data)
    req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    req.add_header("Content-Type", "application/json")

    try:
        # Using built-in urllib to bypass proxy/dns layers of local libraries
        with urllib.request.urlopen(req, timeout=20.0) as response:
            res_body = response.read().decode("utf-8")
            parsed_data = json.loads(res_body)
            
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                return parsed_data[0].get("generated_text", "").strip()
            return "Error: Unexpected response structure from Hugging Face."
            
    except urllib.error.HTTPError as e:
        error_info = e.read().decode("utf-8")
        print(f"HF API HTTP Error: {e.code} - {error_info}")
        return f"Hugging Face API error status: {e.code}. Details: {error_info}"
    except Exception as e:
        print(f"HF Urllib Exception: {e}")
        return f"Urllib connection failed. Network Error: {str(e)}"