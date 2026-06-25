"""08_layered.py
练习:把"用户登录"功能拆成接口层 / 服务层 / 数据层。

任务:
    写一个完整的"用户登录"功能,严格分三层:
    
    数据层:
        - _db_users: 假装是个数据库表(dict)
        - db_get_user_by_username(username) -> dict | None
    
    服务层:
        - service_authenticate(username, password) -> dict
          业务规则:
            - 用户不存在 → raise InvalidCredentialsError
            - 密码错误   → raise InvalidCredentialsError
            - 都对       → return 用户记录(含 id, username, role)
        - 注意:密码存的是 hash("plain"),验证时也要 hash 后比
    
    接口层:
        - POST /login,接收 LoginRequest(username, password)
        - 调 service_authenticate
        - 成功 → 返回 {"user_id": ..., "token": "fake-token-xxx"}
        - 失败(InvalidCredentialsError) → HTTPException(401)
        - response_model 用 LoginResponse

提示:
    - 用 raise 抛业务异常,接口层 try/except 转 HTTP
    - 密码 hash 用最简方式:_hash = lambda s: f"hashed::{s}"
    - 真项目里用 bcrypt / argon2,这里只为演示分层概念
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="分层架构练习")

# ============================================================
# Schemas
# ============================================================
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    user_id: int
    token: str


# ============================================================
# 数据层
# ============================================================
_db_users: dict[str, dict] = {
    "alice": {"id": 1, "username": "alice", "password_hash": "hashed::alice123", "role": "user"},
    "bob":   {"id": 2, "username": "bob",   "password_hash": "hashed::bob_secret", "role": "admin"},
}

def db_get_user_by_username(username: str) -> dict | None:
    # TODO: 从 _db_users 查
    pass


# ============================================================
# 服务层
# ============================================================
class InvalidCredentialsError(Exception):
    pass


def _hash(plain: str) -> str:
    return f"hashed::{plain}"


def service_authenticate(username: str, password: str) -> dict:
    # TODO: 业务规则 —— 用户不存在 / 密码错都抛 InvalidCredentialsError
    pass


# ============================================================
# 接口层
# ============================================================
@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    # TODO: 调 service_authenticate,成功返回 LoginResponse,失败 raise 401
    pass


# 测试用例:
#   # 成功
#   curl -X POST http://127.0.0.1:8000/login \
#        -H "Content-Type: application/json" \
#        -d '{"username":"alice","password":"alice123"}'
#   # 应返回 200 + {"user_id":1,"token":"..."}
#
#   # 失败:用户不存在
#   curl -X POST http://127.0.0.1:8000/login \
#        -H "Content-Type: application/json" \
#        -d '{"username":"nobody","password":"x"}'
#   # 应返回 401
#
#   # 失败:密码错
#   curl -X POST http://127.0.0.1:8000/login \
#        -H "Content-Type: application/json" \
#        -d '{"username":"alice","password":"wrong"}'
#   # 应返回 401


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
