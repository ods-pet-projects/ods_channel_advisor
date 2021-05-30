from typing import Optional
import uvicorn as uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scripts.baseline import get_tfidf_answer


app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/v1/channel/recommend")
def get_channel_recommend(body: str):
    channels = get_tfidf_answer(body)
    return {"channels": channels}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
