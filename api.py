from typing import Optional
import uvicorn as uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from scripts.baseline import get_tfidf_answer

app = FastAPI()


class InputItem(BaseModel):
    id: int
    message: str


class OutputItem(BaseModel):
    id: int
    channels: List[str]


@app.post('/recommend', response_model=List[OutputItem])
def get_channel_recommend(body: List[InputItem]):
    ans_list = []
    for req in body:
        cur_id = req.id
        msg = req.message
        channels = get_tfidf_answer(msg)
        ans = {'id': cur_id,
               'channels': channels}
        ans_list.append(ans)
    return ans_list


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
