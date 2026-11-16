from fastapi import FastAPI
import uvicorn
import random
from pydantic import BaseModel


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": random.randint(1, 20)}

class ClassifyRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify(request: ClassifyRequest):
    # теперь в ответе будет сам текст, а не объект
    user_text = request.text.lower()

    clean_user_text = clean_text_for_rubert(user_text)

    if len(clean_user_text) > 10:
        reply = 'Больше 10'
    else:
        reply = 'Меньше...'

    return {
        "test": "successfully",
        "your_text": request.text,
        "clean_text": clean_user_text,
        "answer": reply
    } 

class ClassifyResponse(BaseModel):
    category_id: int
    category_name: str
    confidence: float
    probabilities: dict[str, float]
    top_3: list[dict]


import re
import html
import pymorphy3
from nltk.corpus import stopwords
from time import perf_counter

def clean_text_for_rubert(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('`', '"')

    return text.strip()




if __name__ == "__main__":
    uvicorn.run("main:app")


# http://127.0.0.1:8000/docs


# {
#   "text": "Привет, мир!"
# }