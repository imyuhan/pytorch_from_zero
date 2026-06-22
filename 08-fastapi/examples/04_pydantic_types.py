"""04_pydantic_types.py
Pydantic 类型提示工具:Literal / Optional / Union / 字段校验器。

学习目标:
    1. Literal:把变量锁死在固定几个值里
    2. Optional / Union:允许字段为空或多种类型
    3. field_validator:自定义校验逻辑

运行方式:
    uvicorn examples.04_pydantic_types:app --reload
"""

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Pydantic 类型工具")


# ============================================================
# 一、Literal:把变量锁死在固定几个值里
# ============================================================
# 场景:用户角色只能是 "admin" / "user" / "guest",不能是其他
# 用 Literal 一行搞定,类型不对直接报错
Role = Literal["admin", "user", "guest"]


class CreateAccountRequest(BaseModel):
    """注册请求 —— role 字段只接受三个值。"""

    username: str = Field(..., min_length=3)
    role: Role = "user"   # 默认是 "user",但必须是 Literal 里那三个之一


@app.post("/accounts")
def create_account(req: CreateAccountRequest) -> dict:
    return {"username": req.username, "role": req.role}


# ============================================================
# 二、Optional / Union:字段可以是 None,也可以是多种类型
# ============================================================
class UpdateProfileRequest(BaseModel):
    """更新个人资料 —— 所有字段都可以不传。

    Optional[str] = None 表示:这个字段要么是 str,要么不传(默认 None)。
    FastAPI 收到 null 也会接受,不会报错。
    """

    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    age: Optional[int] = None


@app.patch("/profile/{user_id}")
def update_profile(user_id: int, req: UpdateProfileRequest) -> dict:
    return {"user_id": user_id, "updated": req.model_dump(exclude_none=True)}


# ============================================================
# 三、Union:字段可以是多种类型之一
# ============================================================
class TextContent(BaseModel):
    """纯文本内容。"""
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    """图片内容。"""
    type: Literal["image"]
    url: str
    width: int
    height: int


# 一个"消息"可以是 text,也可以是 image
# Union[TextContent, ImageContent] = "这两个类型里的任意一个"
MessageContent = Union[TextContent, ImageContent]


class SendMessageRequest(BaseModel):
    """发送消息 —— content 可以是文本或图片。"""

    sender: str
    content: MessageContent


@app.post("/messages")
def send_message(req: SendMessageRequest) -> dict:
    """FastAPI 会根据 content.type 字段自动判断走哪个子模型。"""
    if req.content.type == "text":
        return {"type": "text", "preview": req.content.text[:30]}
    return {"type": "image", "url": req.content.url}


# ============================================================
# 四、field_validator:自定义校验逻辑
# ============================================================
class RegisterRequest(BaseModel):
    """注册请求 —— 自定义密码强度校验。"""

    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_must_be_strong(cls, v: str) -> str:
        """自定义校验:密码必须同时包含字母和数字。"""
        if not any(c.isalpha() for c in v):
            raise ValueError("密码必须包含字母")
        if not any(c.isdigit() for c in v):
            raise ValueError("密码必须包含数字")
        return v


@app.post("/register")
def register(req: RegisterRequest) -> dict:
    return {"username": req.username, "msg": "注册成功"}


# ============================================================
# 五、把错误转成统一格式
# ============================================================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """把 ValueError 转成 422 + 统一错误体。"""
    raise HTTPException(status_code=422, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)