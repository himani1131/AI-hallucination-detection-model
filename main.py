import os
import math
import httpx
import random
from typing import Literal
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- IMPORT LOCAL HUGGING FACE MODULE ---
# This line imports the function from the 'huggingface.py' file you created.
from huggingface import run_local_llm
# ------------------------------------------

# Hugging Face Configuration for NLI consistency checking
# (This still uses the Hugging Face API, which has a generous free tier, 
#  so a token is required on Render for this specific model, but it is free.)
HF_API_URL = "https://api-inference.huggingface.co/models/roberta-large-mnli"
HF_TOKEN = os.getenv("HF_API_TOKEN", "")  # Add this env var ('HF_API_TOKEN') to Render

class LoginRequest(BaseModel):
    name: str
    email: str

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    hallucination_score: float
    verdict: str
    nli_score: float
    rag_score: float
    uncertainty_score: float

class HistoryItem(BaseModel):
    prompt: str
    response: str
    score: float

class HistoryResponse(BaseModel):
    conversations: list[HistoryItem]

class UserResponse(BaseModel):
    name: str
    email: str

class GenerationResult(BaseModel):
    text: str
    token_confidences: list[float]

class NLIResult(BaseModel):
    label: Literal["entailment", "contradiction", "neutral"]
    confidence: float

class RAGVerificationResult(BaseModel):
    supported: bool
    confidence: float

HALLUCINATION_THRESHOLD = 0.45
SCORE_WEIGHTS = {
    "nli": 0.45,
    "rag": 0.35,
    "uncertainty": 0.20,
}

SESSION_HISTORY = []
ACTIVE_USER = {"name": "", "email": ""}

app = FastAPI(title="Production Hallucination Detection Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    """Serves the index.html file."""
    return FileResponse("index.html")

@app.get("/ChatInterface.css")
async def serve_css():
    """Serves the ChatInterface.css file."""
    return FileResponse("ChatInterface.css")

@app.get("/api/health")
async def health_check():
    """Endpoint for monitoring app status."""
    return {"status": "ok", "model": "local_huggingface_ready"}

@app.post("/api/login", response_model=UserResponse)
async def handle_login(request: LoginRequest):
    """Establishes user session."""
    global SESSION_HISTORY, ACTIVE_USER
    ACTIVE_USER["name"] = request.name
    ACTIVE_USER["email"] = request.email
    SESSION_HISTORY = [] 
    return UserResponse(**ACTIVE_USER)

@app.post("/api/logout")
async def handle_logout():
    """Terminates user session."""
    global SESSION_HISTORY, ACTIVE_USER
    ACTIVE_USER = {"name": "", "email": ""}
    SESSION_HISTORY = []
    return {"message": "Logged out successfully"}

# --- MODIFIED FUNCTION FOR FREE, LOCAL INFERENCE ---
def run_llm_generation(prompt: str) -> GenerationResult:
    """
    Runs generation using the local Hugging Face model loaded in huggingface.py.
    This replaces the paid OpenAI call with a free local call.
    """
    print(f"DEBUG: Running generation for prompt: '{prompt}'")
    
    # Call the local LLM function from huggingface.py
    text = run_local_llm(prompt)
    
    # Because we are running locally in an optimized manner, we don't have 
    # true logprobs (token confidences) from this simple implementation.
    # To satisfy the GenerationResult model, we return a flat score.
    # In a full-blown implementation, you would extract this from the model output.
    token_confidences = [0.65] 
    
    return GenerationResult(text=text, token_confidences=token_confidences)
# ---------------------------------------------------

def run_nli_module(prompt: str, generated_response: str) -> NLIResult:
    """
    Checks for logical entailment using the Hugging Face Serverless API.
    (This is free but requires the HF_API_TOKEN environment variable.)
    """
    if not HF_TOKEN:
        # Fallback if token isn't provided
        return NLIResult(label="neutral", confidence=0.5)
        
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": f"Context: {prompt} Statement: {generated_response}"
        }
        with httpx.Client(timeout=10.0) as client:
            res = client.post(HF_API_URL, headers=headers, json=payload)
            if res.status_code == 200:
                # The model returns prediction lists, so we get the best prediction
                data = res.json()[0]
                top_pred = max(data, key=lambda x: x['score'])
                lbl = top_pred['label'].lower()
                
                final_label = "neutral"
                if "contradict" in lbl:
                    final_label = "contradiction"
                elif "entail" in lbl:
                    final_label = "entailment"
                    
                return NLIResult(label=final_label, confidence=round(top_pred['score'], 2))
    except Exception as e:
        print(f"Hugging Face NLI Error: {e}")
    # Return placeholder if the NLI API call fails
    return NLIResult(label="neutral", confidence=0.5)

def run_rag_verifier(prompt: str, generated_response: str) -> RAGVerificationResult:
    """
    RAG validation stub. 
    In a real system, you would plug in your Vector DB search logic here.
    """
    return RAGVerificationResult(supported=True, confidence=0.85)

def calculate_uncertainty_score(token_confidences: list[float]) -> float:
    """
    Calculates structural uncertainty score based on token confidences.
    Higher score means the model is more uncertain.
    """
    if not token_confidences:
        return 1.0
    average_confidence = sum(token_confidences) / len(token_confidences)
    return round(1.0 - average_confidence, 4)

def nli_to_hallucination_score(nli_result: NLIResult) -> float:
    """Map NLI result to a risk score."""
    if nli_result.label == "contradiction":
        return nli_result.confidence
    if nli_result.label == "entailment":
        return 1.0 - nli_result.confidence
    return 0.5 * nli_result.confidence  # Neutral is medium risk

def rag_to_hallucination_score(rag_result: RAGVerificationResult) -> float:
    """Map RAG verification result to a risk score."""
    if rag_result.supported:
        return 1.0 - rag_result.confidence
    return rag_result.confidence

def aggregate_hallucination_score(nli_result: NLIResult, rag_result: RAGVerificationResult, uncertainty_score: float) -> float:
    """Calculates the weighted hallucination risk score."""
    weighted_score = (
        SCORE_WEIGHTS["nli"] * nli_to_hallucination_score(nli_result) +
        SCORE_WEIGHTS["rag"] * rag_to_hallucination_score(rag_result) +
        SCORE_WEIGHTS["uncertainty"] * uncertainty_score
    )
    return round(weighted_score, 4)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Endpoint for evaluating a prompt for hallucination risk.
    It generates content locally, checks NLI consistency, and verifies RAG.
    """
    global SESSION_HISTORY
    # 1. Run LLM generation (using local model)
    generation = run_llm_generation(request.prompt)
    # 2. Run NLI consistency check (using Hugging Face free API)
    nli_result = run_nli_module(request.prompt, generation.text)
    # 3. Run RAG verification (stub)
    rag_result = run_rag_verifier(request.prompt, generation.text)
    # 4. Calculate uncertainty (simple average for local model implementation)
    uncertainty_score = calculate_uncertainty_score(generation.token_confidences)

    # 5. Aggregate metrics to a single risk score
    hallucination_score = aggregate_hallucination_score(
        nli_result=nli_result,
        rag_result=rag_result,
        uncertainty_score=uncertainty_score,
    )

    # 6. Set the verdict based on the threshold
    verdict = "hallucinated" if hallucination_score >= HALLUCINATION_THRESHOLD else "grounded"

    # 7. Add entry to session history
    SESSION_HISTORY.append({
        "prompt": request.prompt,
        "response": generation.text,
        "score": hallucination_score
    })

    # 8. Return response
    return ChatResponse(
        response=generation.text,
        hallucination_score=hallucination_score,
        verdict=verdict,
        nli_score=nli_to_hallucination_score(nli_result),
        rag_score=rag_to_hallucination_score(rag_result),
        uncertainty_score=uncertainty_score
    )

@app.get("/api/history", response_model=HistoryResponse)
async def get_history() -> HistoryResponse:
    """Retrieves session history."""
    return HistoryResponse(conversations=[HistoryItem(**item) for item in SESSION_HISTORY])

@app.get("/api/users", response_model=UserResponse)
async def get_user() -> UserResponse:
    """Retrieves active user profile."""
    if not ACTIVE_USER["name"]:
        raise HTTPException(status_code=404, detail="No active session logged in")
    return UserResponse(**ACTIVE_USER)

if __name__ == "__main__":
    # Local development server settings
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)