"""03_response_model.py
练习:response_model 控制响应。

任务:
    写一个"查用户"接口:
        - 数据库里 UserInDB 包含 id / username / password / email
        - 接口返回 UserResponse(不含 password)
        - 用 response_model 自动过滤 password 字段

提示:
    - UserInDB 和 UserResponse 拆成两个 Pydantic 模型
    - @app.get("/users/{user_id}", response_model=UserResponse)
    - 函数返回值是 UserInDB,但 response_model 自动只吐 UserResponse 字段
"""

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()


class UserInDB(BaseModel):
    """数据库里的用户(完整)。"""
    id: int
    username: str
    password: str
    email: Optional[str] = None


class UserResponse(BaseModel):
    """对外暴露的用户(脱敏)。"""
    id: int
    username: str
    email: Optional[str] = None


# 假装是数据库
fake_users: dict[int, UserInDB] = {
    1: UserInDB(id=1, username="alice", password="alice_secret", email="alice@example.com"),
    2: UserInDB(id=2, username="bob", password="bob_secret", email="bob@example.com"),
}


# 在这里写接口:GET /users/{user_id},返回 UserResponse


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)