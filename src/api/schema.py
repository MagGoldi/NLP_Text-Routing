from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

@app.post("/classify", response_model=ClassifyResponse)
class ClassifyRequest(BaseModel):
    text: str
    if text < 

    app.state.predictor.predict(payload.text)

class ClassifyResponse(BaseModel):
    category_id: int
    category_name: str
    confidence: float
    probabilities: dict[str, float]
    top_3: list[dict]