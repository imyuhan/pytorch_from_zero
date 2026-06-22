"""03_request_body.py
POST 请求 + Pydantic 请求体:让前端发 JSON 过来。

学习目标:
    1. POST 是 HTTP 协议里"提交数据"的请求方法(GET 是"看")
    2. 请求体(Body):请求里携带的数据,通常是 JSON
    3. Pydantic BaseModel:用 Python 类描述"请求数据应该长什么样"

运行方式:
    uvicorn examples.03_request_body:app --reload

测试方式(用 curl 或 Postman):
    curl -X POST http://127.0.0.1:8000/users \\
         -H "Content-Type: application/json" \\
         -d '{"name": "张三", "age": 25, "email": "zs@example.com"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="POST 请求与请求体")


# ============================================================
# 一、什么是"请求体"(Request Body)?
# ============================================================
# GET 请求:数据放在 URL 里(路径参数 + Query 参数),适合查询
# POST 请求:数据放在"请求体"里(通常是一段 JSON),适合"新建/提交"
#
# 例:前端调 POST /users 创建用户,会发一段 JSON 过来:
#     {"name": "张三", "age": 25}
# FastAPI 要做的就是:把这串 JSON 解析成 Python 对象 + 校验字段


# ============================================================
# 二、用 Pydantic BaseModel 定义"请求数据应该长什么样"
# ============================================================
class CreateUserRequest(BaseModel):
    """创建用户的请求体结构。

    Pydantic 会自动:
        1. 把 JSON 转成 Python 对象
        2. 校验字段类型(name 必须是 str、age 必须是 int)
        3. 缺字段 → 报错
        4. 多字段 → 直接忽略(默认),需要拒绝可以加额外配置
    """

    name: str = Field(..., min_length=1, max_length=50, description="用户名")
    age: int = Field(..., ge=0, le=150, description="年龄,0-150")
    email: str | None = Field(None, description="邮箱,可选")


class LoginRequest(BaseModel):
    """登录请求体。"""

    username: str
    password: str = Field(..., min_length=6, description="密码至少 6 位")


# ============================================================
# 三、POST 接口:接收请求体
# ============================================================
@app.post("/users")
def create_user(user: CreateUserRequest) -> dict:
    """创建用户。

    FastAPI 看到函数参数 user: CreateUserRequest,就知道:
        "哦,这个接口要收一段 JSON,我帮你按 CreateUserRequest 的定义校验"
    """
    return {
        "id": 1,                          # 假装数据库生成
        "name": user.name,
        "age": user.age,
        "email": user.email,
        "message": f"用户 {user.name} 创建成功",
    }


@app.post("/login")
def login(req: LoginRequest) -> dict:
    """登录接口。

    密码不足 6 位 → Pydantic 直接拦在门外,返回 422
    """
    return {
        "username": req.username,
        "token": "fake-jwt-token-12345",   # 假装签发的 token
    }


# ============================================================
# 四、嵌套模型:模型里套模型
# ============================================================
class Address(BaseModel):
    """地址子模型。"""

    city: str
    street: str


class CreateOrderRequest(BaseModel):
    """创建订单 —— Address 作为 Order 的字段。"""

    user_id: int
    items: list[str] = Field(..., min_length=1, description="至少一个商品")
    address: Address                       # 嵌套


@app.post("/orders")
def create_order(order: CreateOrderRequest) -> dict:
    """创建订单。

    请求体形如:
    {
        "user_id": 42,
        "items": ["苹果", "香蕉"],
        "address": {"city": "北京", "street": "中关村大街"}
    }
    """
    return {
        "order_id": 1,
        "user_id": order.user_id,
        "items": order.items,
        "city": order.address.city,
        "street": order.address.street,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)