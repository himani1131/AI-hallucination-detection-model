import random
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


class LoginRequest(BaseModel):
    name: str
    email: str


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    hallucination_score: float
    verdict: str


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


app = FastAPI(title="Backend API Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.get("/ChatInterface.css")
async def serve_css():
    return FileResponse("ChatInterface.css")


@app.get("/api/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "loaded",
        "database": "connected",
    }


@app.post("/api/login", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def handle_login(request: LoginRequest) -> UserResponse:
    global SESSION_HISTORY, ACTIVE_USER
    ACTIVE_USER["name"] = request.name
    ACTIVE_USER["email"] = request.email
    SESSION_HISTORY = [] 
    return UserResponse(**ACTIVE_USER)


@app.post("/api/logout", status_code=status.HTTP_200_OK)
async def handle_logout():
    global SESSION_HISTORY, ACTIVE_USER
    ACTIVE_USER = {"name": "", "email": ""}
    SESSION_HISTORY = []
    return {"message": "Logged out successfully"}


def run_llm_generation(prompt: str) -> GenerationResult:
    # Rule 1: "telephone" forces an accurate, highly grounded factual response
    if "telephone" in prompt.lower():
        return GenerationResult(
            text="Alexander Graham Bell is commonly credited with inventing the telephone.",
            token_confidences=[0.96, 0.94, 0.95, 0.93, 0.97, 0.91, 0.94, 0.95],
        )

    # Fallback simulation: Generate wide token-confidence distributions (simulating uncertainty variants)
    return GenerationResult(
        text="This is a generated placeholder response from the future LLM integration.",
        token_confidences=[round(random.uniform(0.15, 0.95), 2) for _ in range(10)],
    )


def run_nli_module(prompt: str, generated_response: str) -> NLIResult:
    if "telephone" in prompt.lower() and "Alexander Graham Bell" in generated_response:
        return NLIResult(label="entailment", confidence=0.94)

    # FIX: Expanded baseline random choices to introduce clear factual conflicts
    label = random.choice(["entailment", "contradiction", "neutral"])
    return NLIResult(label=label, confidence=round(random.uniform(0.40, 0.98), 2))


def run_rag_verifier(prompt: str, generated_response: str) -> RAGVerificationResult:
    if "telephone" in prompt.lower() and "Alexander Graham Bell" in generated_response:
        return RAGVerificationResult(supported=True, confidence=0.92)

    # FIX: Expanded random boolean constraints to simulate weak reference coverage
    return RAGVerificationResult(
        supported=random.choice([True, False]),
        confidence=round(random.uniform(0.35, 0.95), 2),
    )


def calculate_uncertainty_score(token_confidences: list[float]) -> float:
    if not token_confidences:
        return 1.0
    average_confidence = sum(token_confidences) / len(token_confidences)
    return round(1 - average_confidence, 4)


def nli_to_hallucination_score(nli_result: NLIResult) -> float:
    # If there's a structural contradiction, pass a higher risk value
    if nli_result.label == "contradiction":
        return nli_result.confidence
    if nli_result.label == "entailment":
        return 1 - nli_result.confidence
    return 0.5 * nli_result.confidence


def rag_to_hallucination_score(rag_result: RAGVerificationResult) -> float:
    if rag_result.supported:
        return 1 - rag_result.confidence
    # If not supported by RAG contexts, risk score matches verification strength directly
    return rag_result.confidence


def aggregate_hallucination_score(
    nli_result: NLIResult,
    rag_result: RAGVerificationResult,
    uncertainty_score: float,
) -> float:
    weighted_score = (
        SCORE_WEIGHTS["nli"] * nli_to_hallucination_score(nli_result)
        + SCORE_WEIGHTS["rag"] * rag_to_hallucination_score(rag_result)
        + SCORE_WEIGHTS["uncertainty"] * uncertainty_score
    )
    return round(weighted_score, 4)


@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    global SESSION_HISTORY
    generation = run_llm_generation(request.prompt)
    nli_result = run_nli_module(request.prompt, generation.text)
    rag_result = run_rag_verifier(request.prompt, generation.text)
    uncertainty_score = calculate_uncertainty_score(generation.token_confidences)

    hallucination_score = aggregate_hallucination_score(
        nli_result=nli_result,
        rag_result=rag_result,
        uncertainty_score=uncertainty_score,
    )

    verdict = (
        "hallucinated"
        if hallucination_score >= HALLUCINATION_THRESHOLD
        else "grounded"
    )

    SESSION_HISTORY.append({
        "prompt": request.prompt,
        "response": generation.text,
        "score": hallucination_score
    })

    return ChatResponse(
        response=generation.text,
        hallucination_score=hallucination_score,
        verdict=verdict,
    )


@app.get("/api/history", response_model=HistoryResponse, status_code=status.HTTP_200_OK)
async def get_history() -> HistoryResponse:
    return HistoryResponse(conversations=[HistoryItem(**item) for item in SESSION_HISTORY])


@app.get("/api/users", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_user() -> UserResponse:
    if not ACTIVE_USER["name"]:
        raise HTTPException(status_code=404, detail="No active session logged in")
    return UserResponse(**ACTIVE_USER)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)