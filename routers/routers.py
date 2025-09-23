from fastapi import FastAPI
from schemas.schemas import TextSimilarityRequest, TextSimilarityResponse
from service.service import get_similarity_score

app = FastAPI()

@app.post("/similarity", response_model=TextSimilarityResponse)
def get_similarity(data: TextSimilarityRequest):
    score = get_similarity_score(data.text1, data.text2)
    return TextSimilarityResponse(similarity_score=score)