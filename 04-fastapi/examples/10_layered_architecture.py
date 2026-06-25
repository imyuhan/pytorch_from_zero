"""10_layered_architecture.py
接口层 / 服务层 / 数据层:为什么分三层?怎么分?

学习目标:
    1. 理解"分层"在解决什么问题
    2. 看清每层各自的职责
    3. 知道换协议(比如换 gRPC)只动接口层、换数据库只动数据层的好处

运行方式:
    uvicorn examples.10_layered_architecture:app --reload

测试方式:
    curl -X POST http://127.0.0.1:8000/users \\
         -H "Content-Type: application/json" \\
         -d '{"username":"alice","password":"alice123"}'
    curl http://127.0.0.1:8000/users/1
    curl -X POST http://127.0.0.1:8000/login \\
         -H "Content-Type: application/json" \\
         -d '{"username":"alice","password":"alice123"}'
"""

# ============================================================
# 一、为什么需要分层?
# ============================================================
# 把所有代码塞一个文件里,小项目没问题;一旦业务复杂就会:
#   - 接口函数 200 行,HTTP 校验、业务规则、SQL 全混一起
#   - 想加个"管理员批量删用户"接口,发现要复制粘贴一大坨
#   - 想换数据库(MySQL → PostgreSQL),要在 20 个接口里改 SQL
#   - 想换协议(HTTP → gRPC),所有逻辑都要重写
#
# 分层后,每一层只关心自己的事:
#
#   接口层 (api.py)          → "URL 长什么样"
#   ┌──────────────────┐
#   │ @app.post("/x")  │
#   │ 收请求 → 调服务   │      ↕ 调用
#   │ 包装响应          │      ↓
#   └──────────────────┘
#   服务层 (service.py)      → "业务规则是什么"
#   ┌──────────────────┐
#   │ 检查用户名重复    │      ↕ 调用
#   │ 加密密码          │      ↓
#   │ 写数据库          │
#   └──────────────────┘
#   数据层 (db.py)           → "数据从哪儿来"
#   ┌──────────────────┐
#   │ SQL 查询          │
#   │ 连接池            │
#   │ ORM 模型          │
#   └──────────────────┘
#
# 换数据库 → 只动 db.py
# 换协议   → 只动 api.py
# 改业务   → 只动 service.py


# ============================================================
# 二、真实项目里这三层应该是三个文件
# ============================================================
# 这一章为了"一个文件能跑",把三层塞在同一个文件里,用注释分块。
# 实际项目结构:
#
#   app/
#     api.py            ← 接口层(本文件第一块)
#     service.py        ← 服务层(本文件第二块)
#     db.py             ← 数据层(本文件第三块)
#     schemas.py        ← Pydantic 数据模型(接口层用)
#     main.py           ← FastAPI() 实例 + 启动
#
# 下一章做项目时,就按这个结构来组织代码。


# ============================================================
# 三、Pydantic Schemas(数据形状定义)
# ============================================================
# 注:生产项目一般放 schemas.py,这里和 api 层放一起方便阅读
from typing import Optional
from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    """注册请求体。"""
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    """对外暴露的用户数据(故意不含 password)。"""
    id: int
    username: str
    role: str = "user"


class LoginRequest(BaseModel):
    """登录请求体。"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """登录响应。"""
    user_id: int
    token: str


# ============================================================
# 四、数据层(db.py 的角色)—— 假装操作"数据库"
# ============================================================
# 真实项目这里会用 SQLAlchemy / Tortoise / 写原生 SQL
# 这一章为了零依赖,用内存 dict 假装
#
# 数据层职责:
#   - 屏蔽"数据具体存在哪儿"(内存 / MySQL / 文件 / 远程 API)
#   - 只暴露最简单的 CRUD:create / get / list / update / delete
#   - **不做业务校验**(比如"用户名不能重复"是业务规则,不是数据层职责)
# ============================================================

class _UserRecord:
    """假装是 ORM 的一条记录。"""
    def __init__(self, id: int, username: str, password_hash: str, role: str = "user"):
        self.id = id
        self.username = username
        self.password_hash = password_hash   # 永远不存明文密码!
        self.role = role

    def to_dict(self) -> dict:
        # 数据层提供"原始数据"形式,不含业务加工
        return {
            "id": self.id,
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role,
        }


# 假装是数据库表
_db_users: dict[int, _UserRecord] = {}
_next_id = 1


def db_get_user_by_id(user_id: int) -> _UserRecord | None:
    """数据层 —— 按 id 查用户。"""
    return _db_users.get(user_id)


def db_get_user_by_username(username: str) -> _UserRecord | None:
    """数据层 —— 按用户名查用户。"""
    for u in _db_users.values():
        if u.username == username:
            return u
    return None


def db_create_user(username: str, password_hash: str) -> _UserRecord:
    """数据层 —— 插入新用户。返回插入后的记录(含 id)。"""
    global _next_id
    rec = _UserRecord(id=_next_id, username=username, password_hash=password_hash)
    _db_users[_next_id] = rec
    _next_id += 1
    return rec


# ============================================================
# 五、服务层(service.py 的角色)—— 业务规则
# ============================================================
# 服务层职责:
#   - 业务规则:"用户名不能重复""密码要 hash""登录要验密码"
#   - 调用数据层 + 可能调用其他服务(如发邮件、调用外部 API)
#   - **不关心 HTTP**(不知道有 request / response 是什么)
#   - 抛出"业务异常",让接口层决定怎么转 HTTP 状态码
#
# 关键:服务层函数应该是"纯 Python 函数",不依赖 FastAPI 的任何东西
# 这样它可以被:
#   - 接口层调用(HTTP)
#   - CLI 脚本调用(批量任务)
#   - 定时任务调用(cron)
#   - 测试用例直接调用(不用 mock HTTP)
# ============================================================

# 业务异常 —— 服务层用业务异常,接口层翻译成 HTTP 状态码
class UserAlreadyExistsError(Exception):
    pass

class InvalidCredentialsError(Exception):
    pass

class UserNotFoundError(Exception):
    pass


# 假装一个 hash 函数(真实项目用 bcrypt / argon2)
def _hash_password(plain: str) -> str:
    return f"hashed::{plain}"


def _verify_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed


def service_create_user(username: str, password: str) -> _UserRecord:
    """业务:创建用户。
    
    步骤:
        1. 业务校验:用户名是否已存在
        2. 业务加工:密码 hash
        3. 调数据层存
        4. 返回记录
    """
    if db_get_user_by_username(username):
        raise UserAlreadyExistsError(f"用户名 {username!r} 已被占用")
    return db_create_user(username=username, password_hash=_hash_password(password))


def service_authenticate(username: str, password: str) -> _UserRecord:
    """业务:验证登录。
    
    返回用户记录(里面含 id),由接口层生成 token。
    """
    user = db_get_user_by_username(username)
    if user is None:
        raise InvalidCredentialsError("用户名或密码错误")
    if not _verify_password(password, user.password_hash):
        raise InvalidCredentialsError("用户名或密码错误")
    return user


def service_get_user(user_id: int) -> _UserRecord:
    """业务:按 id 查用户。"""
    user = db_get_user_by_id(user_id)
    if user is None:
        raise UserNotFoundError(f"用户 {user_id} 不存在")
    return user


# ============================================================
# 六、接口层(api.py 的角色)—— HTTP 包装
# ============================================================
# 接口层职责:
#   - URL → 函数 的映射
#   - 收请求(用 Pydantic 校验)
#   - 调服务层
#   - 把服务层返回 / 抛出的东西转成 HTTP 响应
#   - **不做业务逻辑**
# ============================================================

from fastapi import FastAPI, HTTPException

app = FastAPI(title="分层架构示例")


@app.post("/users", response_model=UserResponse, status_code=201)
def create_user(req: CreateUserRequest) -> UserResponse:
    """接口:注册用户。
    
    流程:
        1. Pydantic 自动校验请求体(req 已经是合法 CreateUserRequest)
        2. 调 service_create_user(可能抛业务异常)
        3. 把 _UserRecord 转成 UserResponse(对外模型,不含 password)
        4. 返回
    """
    try:
        user = service_create_user(req.username, req.password)
    except UserAlreadyExistsError as e:
        # 业务异常 → HTTP 状态码
        # "用户名已存在" → 409 Conflict
        raise HTTPException(status_code=409, detail=str(e))

    # 手动转成对外模型(也可以用 response_model 自动过滤)
    return UserResponse(id=user.id, username=user.username, role=user.role)


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int) -> UserResponse:
    """接口:查用户。"""
    try:
        user = service_get_user(user_id)
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return UserResponse(id=user.id, username=user.username, role=user.role)


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    """接口:登录。
    
    注意:接口层只生成 token(因为 token 属于"协议层"概念),
          业务规则(验密码)在 service 层。
    """
    try:
        user = service_authenticate(req.username, req.password)
    except InvalidCredentialsError as e:
        # 凭据错 → 401
        raise HTTPException(status_code=401, detail=str(e))
    # 真实项目里 token 用 jwt 库生成
    fake_token = f"token-for-user-{user.id}"
    return LoginResponse(user_id=user.id, token=fake_token)


# ============================================================
# 七、分层的"换协议/换数据库"演示(思想实验)
# ============================================================
# 假设明天要换数据库 MySQL → PostgreSQL:
#   ✗ 不用改 api.py
#   ✗ 不用改 service.py
#   ✓ 只用改 db.py(把 _db_users 换成 SQLAlchemy session)
#
# 假设明天要加一个 gRPC 接口:
#   ✗ 不用改 service.py(它本来就是"纯 Python 函数")
#   ✗ 不用改 db.py
#   ✓ 只用新建 grpc_server.py,调同样的 service_xxx 函数
#
# 这就是分层的价值:**让每一层可以独立替换,不影响其他层**。
#
# 同样的代码,既能给 Web 用(HTTP)、又能给 CLI 用(命令行脚本)、
# 又能给定时任务用(cron),靠的就是 service 层"不依赖任何协议"。


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
