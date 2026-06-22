"""02_pydantic_validator.py
练习:Pydantic 模型 + 自定义校验器。

任务:
    定义一个 CreateBookRequest 模型,要求:
        - title: str,必填,2-100 字
        - author: str,必填
        - year: int,必填,1900-2025
        - isbn: str,可选,必须是 13 位数字(用 field_validator)
        - price: float,必填,大于 0

提示:
    - Field(..., min_length=2, max_length=100)
    - field_validator + 正则或字符串判断
    - ISBN 可以用 `len(v) == 13 and v.isdigit()` 判断
"""

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI

app = FastAPI()


class CreateBookRequest(BaseModel):
    # 在这里定义字段
    pass


@app.post("/books")
def create_book(req: CreateBookRequest) -> dict:
    return {"title": req.title, "author": req.author, "isbn": req.isbn}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)