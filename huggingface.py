import os
import httpx

# We will use the Hugging Face Serverless API to query the model for free.
API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_TOKEN = os.getenv("HF_API_TOKEN", "")

def run_local_llm(prompt: str) -> str:
    """
    Queries the Hugging Face Serverless API for text generation.
    This takes 0MB of local RAM, preventing the 'Out of memory' error.
    """
    if not HF_TOKEN:
        return "Fallback: Please configure a valid HF_API_TOKEN in your environment variables."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 40,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Serverless API returns a list containing a dict with 'generated_text'
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "").strip()
                return "Error: Unexpected response structure from Hugging Face."
            else:
                print(f"HF API Error: {response.status_code} - {response.text}")
                return f"Hugging Face API returned error status: {response.status_code}"
                
    except Exception as e:
        print(f"HF Request Exception: {e}")
        return "Failed to communicate with Hugging Face Serverless API."