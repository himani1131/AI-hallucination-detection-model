import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the absolute smallest functional model that can run on a free Render instance.
MODEL_NAME = "distilgpt2"

print(f"Loading local model: {MODEL_NAME}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model, ensuring it runs only on CPU (as free Render has no GPU)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to('cpu')

def run_local_llm(prompt: str) -> str:
    """
    Runs the distilgpt2 model locally on the free Render instance (CPU only).
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
        
        # Adjust generation parameters for better performance on a limited CPU
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,  # Keep responses short to save CPU/RAM
                do_sample=True,
                temperature=0.7,
                top_k=50,
                num_return_sequences=1
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response, as causal models include it
        if full_response.startswith(prompt):
            final_response = full_response[len(prompt):].strip()
        else:
            final_response = full_response.strip()
            
        return final_response
        
    except Exception as e:
        print(f"Error running local LLM: {e}")
        return "Local LLM inference failed due to a system resource issue."

if __name__ == "__main__":
    # Test generation locally before deploying
    test_prompt = "What is the capital of India?"
    print(f"\nPrompt: {test_prompt}")
    response = run_local_llm(test_prompt)
    print(f"Response: {response}")