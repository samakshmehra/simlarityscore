from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas.schemas import TextSimilarityRequest, TextSimilarityResponse
from service.service import get_similarity_score

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/similarity", response_model=TextSimilarityResponse)
def get_similarity(data: TextSimilarityRequest):
    score = get_similarity_score(data.text1, data.text2)
    return TextSimilarityResponse(similarity_score=score)