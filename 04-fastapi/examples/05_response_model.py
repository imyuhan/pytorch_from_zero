"""05_response_model.py
响应类型定义:用 response_model 控制接口"返回什么数据"。

学习目标:
    1. response_model:告诉 FastAPI"这个接口应该返回这种结构的数据"
    2. 自动过滤敏感字段(比如 password 字段不会泄露给前端)
    3. 自动生成 API 文档

运行方式:
    uvicorn examples.05_response_model:app --reload
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

app = FastAPI(title="响应类型定义")


# ============================================================
# 一、数据库里的"原始数据"和"对外暴露的数据"要分开
# ============================================================
# 数据库存的是完整数据(包含密码、token 等敏感信息)
# 但接口返回给前端时,密码这些字段绝不能泄露
# 解决:用两个 Pydantic 模型 —— 一个"内部",一个"外部"


class UserInDB(BaseModel):
    """数据库里的用户(内部用)。"""

    id: int
    username: str
    password: str               # 敏感!
    email: Optional[str] = None
    role: str = "user"


class UserResponse(BaseModel):
    """对外暴露的用户数据(给前端)。"""

    id: int
    username: str
    email: Optional[str] = None
    role: str


# ============================================================
# 二、用 response_model 自动过滤敏感字段
# ============================================================
# 假装这是数据库
fake_db: dict[int, UserInDB] = {
    1: UserInDB(id=1, username="alice", password="secret123", email="alice@example.com", role="admin"),
    2: UserInDB(id=2, username="bob", password="bob_secret", email="bob@example.com"),
}


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int) -> UserInDB:
    """查用户 —— 返回值是 UserInDB,但 FastAPI 只把 UserResponse 的字段吐出去。

    即使函数里返回的是带 password 的 UserInDB,response_model=UserResponse
    会让 FastAPI 自动过滤掉 password 字段,保证不会泄露。
    """
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="用户不存在")
    return fake_db[user_id]


@app.get("/users", response_model=list[UserResponse])
def list_users() -> list[UserInDB]:
    """列出所有用户 —— 自动过滤每个用户的 password。"""
    return list(fake_db.values())


# ============================================================
# 三、自定义响应状态码
# ============================================================
class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str
    email: Optional[str] = None


@app.post(
    "/users",
    response_model=UserResponse,
    status_code=201,           # 创建成功用 201,而不是默认的 200
)
def create_user(req: CreateUserRequest) -> UserInDB:
    """创建用户,返回 201。"""
    new_id = max(fake_db.keys()) + 1
    user = UserInDB(
        id=new_id,
        username=req.username,
        password=req.password,
        email=req.email,
    )
    fake_db[new_id] = user
    return user


# ============================================================
# 四、统一错误响应
# ============================================================
class ErrorResponse(BaseModel):
    """统一的错误响应体。"""

    error: str = Field(..., description="错误代码,如 'user_not_found'")
    message: str = Field(..., description="人类可读的错误说明")


@app.get("/me", response_model=UserResponse)
def get_me() -> UserResponse:
    """演示统一错误体 —— 永远找不到"我"。"""
    raise HTTPException(
        status_code=404,
        detail=ErrorResponse(
            error="user_not_found",
            message="当前用户未找到",
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)