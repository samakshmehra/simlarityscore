from pydantic import BaseModel

#request 
class TextSimilarityRequest(BaseModel):
    text1: str
    text2: str

#response 
class TextSimilarityResponse(BaseModel):
    similarity_score: float