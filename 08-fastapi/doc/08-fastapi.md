# 08 - FastAPI 速成(教学文档)

> 本章是给初学者的 FastAPI 入门。每节都从"为什么要这个东西"讲起,所有术语都给生活化比喻,代码逐行注释。
>
> 学完你应该能:听懂"接口、路由、请求体、流式响应、WebSocket"这些词不再犯怵,能用 FastAPI 写出能给别人调用的 HTTP 服务。

## 8.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_hello_world.py` | 最小 FastAPI 应用 + uvicorn 启动 |
| `examples/02_path_query.py` | 路径参数 + Query 参数 + Optional |
| `examples/03_request_body.py` | POST 请求 + Pydantic 请求体 + 嵌套模型 |
| `examples/04_pydantic_types.py` | `Literal` / `Optional` / `Union` / `field_validator` |
| `examples/05_response_model.py` | `response_model` 控制响应 + 状态码 + 统一错误体 |
| `examples/06_sse_stream.py` | SSE 流式响应 + `text/event-stream` + 浏览器演示页 |
| `examples/07_websocket.py` | WebSocket 双向通信 + 浏览器演示页 |

## 8.2 基础知识

> 这一节是本章的"概念地基"。读完再去翻 examples,会觉得所有代码都在重复这些概念。

### 8.2.0 准备知识:Web 是怎么"一问一答"的

在学 FastAPI 之前,先搞清楚"Web 服务器到底在干嘛"。这部分不会写代码,只需要建立画面感。

#### 1) 客户端和服务端

想象你去餐厅吃饭:
- **你** = 客户端(浏览器、手机 App、curl 命令)
- **后厨** = 服务端(我们的 FastAPI 程序)
- **服务员** = 网络(HTTP 协议)

你点菜(请求) → 服务员把点菜单送到后厨 → 后厨做完端回来(响应)。

**客户端**:发起请求的一方。在我们项目里通常是浏览器、curl 命令、或者别的程序。
**服务端**:响应请求的一方。在我们项目里就是 FastAPI 跑起来的进程。

#### 2) HTTP 协议——"一问一答"的格式

HTTP 是 HyperText Transfer Protocol 的缩写,翻译过来就是"超文本传输协议"。你可以把它理解成:
**一种规定好的"对话格式",让客户端和服务员(服务员指网络)能听懂彼此在说什么**。

一次 HTTP 交互长这样:

```
[客户端发的请求]                          [服务端回的响应]
POST /login HTTP/1.1                      HTTP/1.1 200 OK
Host: example.com                         Content-Type: application/json
Content-Type: application/json            Content-Length: 43

{"username": "alice",                     {"status": "ok",
 "password": "secret"}                     "token": "abc123"}
```

请求和响应都是"一坨文本",按固定格式写。FastAPI 和浏览器都已经帮你处理了这些格式,你只关心**请求里带了什么数据、响应里返回什么数据**。

#### 3) HTTP 请求方法(verb)

最常见的几种:

| 方法 | 语义 | 典型用途 |
|------|------|---------|
| **GET** | "看" | 查询、读数据 |
| **POST** | "提交" | 新建、提交表单 |
| **PUT** | "更新" | 整体更新一个资源 |
| **PATCH** | "修改" | 部分更新 |
| **DELETE** | "删" | 删除资源 |

> 在 FastAPI 里,这些动词对应 `@app.get(...)` / `@app.post(...)` / `@app.put(...)` 等装饰器。

#### 4) 一次 HTTP 请求的三个"放数据的位置"

请求里"想带数据过去",有三种位置:

| 位置 | 写法举例 | 适用场景 |
|------|---------|---------|
| **URL 路径** | `/users/42` | "要哪个资源" |
| **URL 查询(Query)** | `/search?q=fastapi&limit=10` | "筛选/分页/排序" |
| **请求体(Body)** | 一段 JSON | "新建/提交大块数据" |

FastAPI 用三种不同的函数参数来接收这三种数据——这个我们 8.2.3 详细讲。

#### 5) TCP / UDP / WebSocket 是什么

这些都是"传输层"的协议,可以这样理解:

- **TCP**:打电话,先接通、再说话、说完了挂。**可靠**但**慢**。HTTP 就跑在 TCP 上。
- **UDP**:发短信,只管发,不管对方收没收到。**快**但**不可靠**。视频通话、网游常用。
- **WebSocket**:像 TCP,但**保持连接不断**,两边可以随时互相说话。8.3.7 节会讲。

> 本章不用你直接和 TCP/UDP 打交道,只需要知道:"HTTP 是基于 TCP 的、WebSocket 是另一种保持连接不断的方式"。

#### 6) 一个完整的访问流程

浏览器访问 `http://127.0.0.1:8000/users/42?role=admin` 时,发生了什么:

```
浏览器                    网络                    FastAPI 服务
  |                       |                          |
  |--- GET 请求 -------->|                          |
  |    路径:/users/42     |                          |
  |    Query:role=admin   |                          |
  |                       |--- 转发请求 ----------->|
  |                       |                          |
  |                       |                          | 处理逻辑
  |                       |                          |
  |                       |<--- 200 OK + JSON ------|
  |                       |                          |
  |<--- 渲染页面 --------|                          |
```

你的 FastAPI 代码只关心中间那个方框里的事:**收请求 → 处理 → 返回**。

---

### 8.2.1 FastAPI 是什么,为什么用它

**FastAPI 是一个 Python Web 框架**,专门用来快速写"接口"——也就是那种别人用 URL 调用、能返回数据的服务。

为什么学它而不是别的(Django / Flask)?
- **类型提示驱动**:写 Python 类型注解,它帮你自动校验、自动生成文档
- **性能好**:和 Node.js、Go 一个水平
- **自带 Swagger 文档**:访问 `/docs` 就有可视化接口页面,**免费送**
- **生态好**:背后是 Starlette(性能) + Pydantic(校验)

一句话:**用 Python 写接口,FastAPI 是当下最舒服的选择**。

---

### 8.2.2 一个最小 FastAPI 应用长什么样

`examples/01_hello_world.py`:

```python
from fastapi import FastAPI

app = FastAPI()                          # ① 创建应用实例

@app.get("/")                            # ② 注册一个 GET 接口
def home():
    return {"message": "你好!"}            # ③ 返回的数据会自动变成 JSON
```

启动:
```bash
uvicorn examples.01_hello_world:app --reload
```

浏览器访问 http://127.0.0.1:8000/ 会看到 `{"message": "你好!"}`。
访问 http://127.0.0.1:8000/docs **白送你一个可视化接口文档**。

几个关键词解释:

| 关键词 | 意思 |
|--------|------|
| `FastAPI()` | 创建应用实例,相当于"一台刚接电源的服务器" |
| `@app.get("/")` | **装饰器**:给这个函数"装"到路径 `/` 上。访问 `/` 就执行它 |
| `uvicorn` | 一个 ASGI 服务器,**负责把 Python 应用跑起来**接受网络请求 |
| `app --reload` | `--reload` 表示"代码改了自动重启",开发时很方便 |

> **接口 = 路由 = endpoint**,这三个词在不同场合指同一个东西——一个 URL 路径 + 处理它的函数。

---

### 8.2.3 URL 路径的三种"参数"

URL 里除了"路径"和"查询字符串",还有第三种"路径参数"。这是初学者最容易混的地方。

#### 1) 路径参数(Path Parameter)

URL 路径里的"变量",用 `{}` 占位:

```
/users/{user_id}      ← {user_id} 就是路径参数
```

访问 `/users/42`,user_id 就是 42;访问 `/users/abc`,user_id 就是 `"abc"`。

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):              # ← 写 int 类型,自动校验
    return {"user_id": user_id}
```

**类型自动校验**:`/users/abc` 会返回 422 错误,因为 int 拿不到 "abc"。

#### 2) Query 参数(Query String)

URL 里 `?` 后面的 `key=value`:

```
/search?q=fastapi&limit=10
└──┘ └────────┴────────┘
路径  查询字符串
```

```python
@app.get("/search")
def search(q: str, limit: int = 10):     # 没出现在路径里的、有默认值的 = Query 参数
    return {"q": q, "limit": limit}
```

**识别诀窍**:函数参数里
- 在路径里有 `{xxx}` 占位 → **路径参数**(必填)
- 在路径里没有,但**有默认值** → **可选 Query 参数**
- 在路径里没有,**没有默认值** → **必填 Query 参数**

#### 3) 请求体(Body)

放在"请求正文"里的数据,通常是一段 JSON。**只用 POST/PUT/PATCH 请求才会有 Body**。

```python
class CreateUser(BaseModel):
    name: str
    age: int

@app.post("/users")
def create_user(user: CreateUser):       # ← Pydantic 模型 = 请求体
    return {"name": user.name, "age": user.age}
```

调用方式(curl):
```bash
curl -X POST http://127.0.0.1:8000/users \
     -H "Content-Type: application/json" \
     -d '{"name": "张三", "age": 25}'
```

> `-H "Content-Type: application/json"` 告诉服务器"我发的是 JSON"。
> `-d` 后面的就是请求体。

#### 三种参数的"分工"

| 参数类型 | 放哪儿 | 适合什么 | FastAPI 怎么识别 |
|---------|--------|---------|-----------------|
| **路径参数** | URL 路径里 | "定位哪个资源" | 函数参数 + 路径里有 `{}` |
| **Query 参数** | URL `?` 后面 | "筛选/分页/排序" | 函数参数 + 没默认值或有默认值 |
| **Body** | 请求正文 | "新建/提交大块数据" | 函数参数是 Pydantic 模型 |

> 一句话记忆:**路径参数定位资源,Query 参数筛资源,Body 提交数据**。

---

### 8.2.4 Pydantic:数据校验的"守门员"

#### 1) 为什么需要数据校验

想象你写了一个"创建用户"的接口,前端可能发过来:
- 用户名是个数字(`123` 而不是字符串)
- 年龄是个负数(`-5`)
- 该有的字段没传

如果不校验,这些"脏数据"直接进数据库,后面 debug 想哭。**Pydantic 就是那个帮你拦在门口的守门员**。

#### 2) 用 Pydantic 定义"数据应该长什么样"

```python
from pydantic import BaseModel, Field

class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)   # 必填,1-50 字
    age: int = Field(..., ge=0, le=150)                   # 必填,0-150
    email: str | None = None                               # 可选
```

三件事 Pydantic 自动帮你做:
1. **类型校验**:age 必须能转成 int,转不了就报错
2. **范围校验**:age 必须在 0-150 之间
3. **必填校验**:name 和 age 缺一不可

> 字段名后的 `= Field(...)` 里的 `...` 表示"必填";`= None` 表示"可选"。

#### 3) 嵌套模型:数据里有数据

```python
class Address(BaseModel):
    city: str
    street: str

class CreateOrder(BaseModel):
    user_id: int
    address: Address                    # 嵌套一个 Address
```

Pydantic 会自动帮你解析嵌套结构,你的代码里 `order.address.city` 直接能用。

#### 4) `model_dump()` 把对象转成字典

```python
user = CreateUserRequest(name="张三", age=25, email="zs@example.com")
user.model_dump()
# {'name': '张三', 'age': 25, 'email': 'zs@example.com'}
```

存数据库、序列化成 JSON、log 输出,经常要用。

---

### 8.2.5 typing 工具:`Literal` / `Optional` / `Union`

这三种类型注解工具解决"字段值需要更精确约束"的问题。

#### 1) `Literal`:字段只能是固定几个值之一

```python
from typing import Literal

Role = Literal["admin", "user", "guest"]

class CreateAccount(BaseModel):
    role: Role = "user"
```

`role` 字段只能是 `"admin"` / `"user"` / `"guest"` 三个之一,其他值直接报错。

> 适合:用户角色、订单状态、性别、支付方式这种"枚举"场景。

#### 2) `Optional`:字段可以是 None

```python
from typing import Optional

class UpdateProfile(BaseModel):
    nickname: Optional[str] = None       # 不传也行,传了必须是 str
```

等价写法:`str | None = None`(Python 3.10+ 推荐)。

#### 3) `Union`:字段可以是多种类型之一

```python
from typing import Union

TextContent = {"type": "text", "text": str}
ImageContent = {"type": "image", "url": str}

MessageContent = Union[TextContent, ImageContent]
```

`content` 可以是文本消息,也可以是图片消息。FastAPI 根据字段里的 `type` 字段自动判断走哪个子模型。

#### 4) `field_validator`:自定义校验规则

Pydantic 内置的校验不够用时,可以自己写:

```python
from pydantic import field_validator

class Register(BaseModel):
    password: str

    @field_validator("password")
    @classmethod
    def password_strong(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("密码必须包含数字")
        if not any(c.isalpha() for c in v):
            raise ValueError("密码必须包含字母")
        return v
```

> 这条规则会在每次构造对象时自动跑,密码不达标直接报错。

---

### 8.2.6 分层架构:接口层 / 服务层 / 数据层

项目一大,把所有代码塞一个文件里就会乱。一般项目分成三层:

```
┌────────────────────────────┐
│  接口层 (api.py)            │  ← 只关心 URL → 函数 的映射
│  - @app.get / @app.post    │
│  - 收请求、调服务、返回响应  │
└────────────┬───────────────┘
             │ 调用
             ↓
┌────────────────────────────┐
│  服务层 (service.py)        │  ← 业务逻辑写在这里
│  - 查数据库、调用外部 API   │
│  - 数据加工、规则校验       │
└────────────┬───────────────┘
             │ 调用
             ↓
┌────────────────────────────┐
│  数据层 (db.py / model.py)  │  ← 和存储打交道
│  - SQL 查询、ORM 模型       │
│  - 文件读写                 │
└────────────────────────────┘
```

**为什么分层?**
- 接口层只管"URL 长什么样",换协议(比如换 gRPC)只动接口层
- 服务层是核心业务,改规则只动服务层
- 数据层只管"数据从哪儿取",换数据库只动数据层

> 小项目可以不分,但养成"把逻辑放在服务层"的习惯,项目大了才不乱。

---

### 8.2.7 鉴权基础:让接口知道"你是谁"

很多接口不能随便让人调——比如查"我的订单"必须先登录。

#### 1) Cookie:最传统的"身份标识"

服务器发给浏览器的一小段文本,浏览器每次请求都自动带上。
**适合**:浏览器场景(web 后台、博客)
**不适合**:移动 App、第三方 API

#### 2) Token:现代 API 的标准做法

前端登录成功后,服务端返回一个 token(一段无意义的字符串),前端把它存起来,之后每次请求都放在 `Authorization` 头里:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

**适合**:移动 App、第三方 API、前后端分离

#### 3) JWT:一种"自带信息"的 Token

普通 token 只是"一把钥匙",要查数据库才知道你是谁。
**JWT** 把"你是谁 + 过期时间"编码进 token 里,服务端不用查数据库就能验证。

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYWxpY2UiLCJleHAiOjE3MzAwMDB9.signature
└────────── header ──────────┘ └────────── payload ─────────┘ └── sig ──┘
```

> 进阶细节(API Key、OAuth 2.0)在 8.4 节展开。

---

### 8.2.8 CORS 跨域基础(为什么前端调不通后端)

**CORS**(Cross-Origin Resource Sharing)是浏览器的安全机制:网页在 `a.com` 上,不能直接调 `b.com` 的接口,除非 `b.com` 明确允许。

报错长这样:
```
Access to fetch at 'http://api.b.com/...' from origin 'http://a.com' has been blocked by CORS policy
```

**开发场景**几乎一定会遇到——前端跑在 `localhost:3000`,后端跑在 `localhost:8000`,浏览器觉得"两个不同的 origin"。

**解决**:在 FastAPI 里加 CORS 中间件:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 生产环境别写 *,写具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

详细配置在 8.4 节。

---

### 8.2.9 流式响应 vs 一次性响应(SSE 入门)

这一节是脑图里 K14 的"流式响应 响应数据编码方式:text/event-stream"三要点的解释。

#### 一次性响应

服务端把内容**全部算完**,再一次性发给前端。客户端等着。

```
客户端: GET /result
服务端: ............(算 10 秒)............
服务端: 200 OK + {"result": "全部内容"}   ← 一次性
```

#### 流式响应(SSE)

服务端算一段发一段,前端**边收边显示**。

```
客户端: GET /stream
服务端: start
服务端: 第1个字
服务端: 第2个字
服务端: ...
服务端: done
```

**ChatGPT 的打字机效果**就是 SSE。

#### 为什么要用流式?

1. **用户体验好**:不用等十几秒黑屏,内容立刻开始出现
2. **服务器内存省**:不用把全部内容攒在内存里,算一段发一段
3. **容错率高**:第一次连接断了,前端可以重连,服务端从头再来;普通响应断了就全完了

#### SSE 协议格式

每条消息长这样:

```
event: <事件名>
data: <数据,JSON 字符串>
<空行>          ← 必须有空行,前端才知道这条消息结束
```

Python 生成:
```python
def generate():
    yield "event: start\n"
    yield "data: {\"text\": \"你好\"}\n"
    yield "\n"
```

前端监听(浏览器原生 EventSource):
```javascript
const es = new EventSource("/stream");
es.addEventListener("delta", e => {
    const data = JSON.parse(e.data);
    console.log(data.text);
});
```

> 实现细节和完整代码在 `examples/06_sse_stream.py`。

---

### 8.2.10 WebSocket 入门

#### HTTP 是什么

一次 HTTP 请求 = 一次"打电话"。客户端说一句话,服务端回一句话,**连接就关了**。

```
客户端: "你好"
服务端: "你好"
(挂断)
```

#### WebSocket 是什么

连接不断,**两边可以随时说话**。

```
客户端: "你好"
服务端: "你好"
客户端: "在吗?"
服务端: "在"
客户端: "今天天气如何?"
服务端: "..."
(两边都可以随时发,直到某一边说"挂")
```

#### URL 协议

| HTTP | WebSocket |
|------|-----------|
| `http://` | `ws://` |
| `https://` | `wss://` |

#### 适用场景

| 场景 | 为什么用 WebSocket |
|------|------------------|
| 实时聊天 | 双方都要发消息 |
| 实时游戏 | 状态实时同步 |
| 实时数据大屏 | 服务端主动推数据 |
| 协作工具 | 多人同时编辑一个文档 |

> 实现细节和完整代码在 `examples/07_websocket.py`。

---

### 8.2.11 Content-Type 4 种常见编码方式

脑图里提到的"post 请求里面 4 种常见请求编码格式",展开讲。

POST 请求发数据时,数据用什么"格式"组织,要看 `Content-Type` 头:

| Content-Type | 数据长什么样 | 常见场景 |
|-------------|------------|---------|
| `application/json` | `{"name": "张三"}` | **现代 API 首选** |
| `application/x-www-form-urlencoded` | `name=%E5%BC%A0%E4%B8%89&age=25` | 老 HTML 表单提交 |
| `multipart/form-data` | 包含文件二进制 | 上传文件、图片 |
| `text/plain` | 纯文本 | 调试用 |

**FastAPI 默认**就是 `application/json`,发过来的 Body 会被自动解析成 Python 对象。**我们这一章的所有例子都用 JSON**。

文件上传用 `multipart/form-data`,需要用 FastAPI 的 `UploadFile`:

```python
from fastapi import UploadFile, File

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}
```

> 文件上传细节 8.4 节展开。

---

## 8.3 逐个 example 讲解

> 详细讲解见 `examples/0X_*.py` 各脚本顶部的 docstring。本节把每个例子的"重点"和"易错点"拎出来。

### 8.3.1 `01_hello_world.py` — 最小 FastAPI 应用

**重点**:
- `FastAPI()` 创建应用实例
- `@app.get("/")` 注册一个接口
- 函数返回值自动转 JSON

**易错点**:
- 启动时用 `uvicorn 文件名:app` 不是 `python 文件名.py`(`if __name__` 块只是备用)
- 改完代码没加 `--reload` 不会自动重启

### 8.3.2 `02_path_query.py` — 路径参数 + Query 参数

**重点**:
- 路径参数写在 `@app.get("/users/{user_id}")` 里,函数参数名要和 `{user_id}` 一致
- 函数参数**不在路径里**且**没默认值** = 必填 Query 参数
- 函数参数**不在路径里**且**有默认值** = 可选 Query 参数
- `Optional[类型] = None` = "可以是这个类型,也可以不传"

**易错点**:
- 路径参数和函数参数名不一致 → 报错
- 想做"全可选",但忘了给默认值 → 必填报错

### 8.3.3 `03_request_body.py` — POST 请求 + 请求体

**重点**:
- 函数参数是 Pydantic 模型 = 这个接口收请求体
- `Field(...)` 的 `...` 表示必填
- `Field(..., min_length=1)` 等内置校验规则
- 嵌套模型:一个模型的字段是另一个模型

**易错点**:
- `Content-Type` 头忘了设 → FastAPI 解析不了
- 嵌套模型忘了写子模型 → 报 `NameError`

### 8.3.4 `04_pydantic_types.py` — 类型工具

**重点**:
- `Literal["a", "b", "c"]` 限定取值范围
- `Optional[X]` 和 `X | None` 等价
- `Union[A, B]` 表示"两种类型都行",FastAPI 会按字段内容自动判别
- `field_validator` 写自定义校验

**易错点**:
- `Union` 子模型里忘了加 `type` 字段做"区分标记"
- `field_validator` 没写 `@classmethod` 装饰器

### 8.3.5 `05_response_model.py` — 响应控制

**重点**:
- `response_model=Model` 告诉 FastAPI"返回这个模型的结构",多余字段自动过滤
- 数据库存完整数据,接口返回"脱敏"数据,用两个模型分离
- `status_code=201` 自定义响应码
- `HTTPException(status_code=..., detail=...)` 抛错

**易错点**:
- `response_model` 写成"内部数据模型" → 敏感字段泄露
- 抛错时 `status_code` 写成 200 → 客户端不知道出错了

### 8.3.6 `06_sse_stream.py` — SSE 流式响应

**重点**:
- `StreamingResponse(生成器, media_type="text/event-stream")`
- 生成器函数用 `yield` 持续产出内容
- SSE 消息格式:`event: 事件名\ndata: 数据\n\n`
- `Cache-Control: no-cache` 必须加,否则中间代理会缓存,流式效果出不来

**易错点**:
- `media_type` 写成 `text/plain` → 前端 EventSource 收不到
- 忘了 `Cache-Control: no-cache` → 内容一次性出现,看不到流式效果

### 8.3.7 `07_websocket.py` — WebSocket 双向通信

**重点**:
- `@app.websocket("/ws/...")` 注册 WS 接口
- `await websocket.accept()` 同意连接
- `receive_text()` / `send_text()` 收发文本
- `WebSocketDisconnect` 异常处理客户端断开

**易错点**:
- 忘了 `accept()` → 连接建不起来
- 死循环忘了 break / 没异常处理 → 服务端卡死

---

## 8.4 进阶知识

### 8.4.1 鉴权深度:API Key / JWT / OAuth 2.0

**API Key**:最简单的鉴权——给每个调用方发一个固定字符串作为"通行证"。

```python
from fastapi import Header, HTTPException

API_KEY = "secret-key-123"

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")

@app.get("/protected", dependencies=[Depends(verify_key)])
def protected():
    return {"message": "你通过了鉴权"}
```

调用:
```bash
curl http://127.0.0.1:8000/protected -H "X-API-Key: secret-key-123"
```

**JWT**:服务端签发一段"带身份信息的 token",前端存起来每次请求带上:

```python
from jose import jwt

def create_token(user_id: int) -> str:
    return jwt.encode({"sub": user_id, "exp": ...}, SECRET_KEY, algorithm="HS256")

def verify_token(token: str = Header(...)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload["sub"]
```

**OAuth 2.0**:三方授权标准,比如"用 GitHub 账号登录第三方网站"。

### 8.4.2 CORS 跨域深度

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",         # 开发环境前端
        "https://my-frontend.com",       # 生产环境前端
    ],
    allow_credentials=True,             # 允许带 cookie
    allow_methods=["GET", "POST"],      # 允许的方法
    allow_headers=["*"],                # 允许的请求头
)
```

| 配置 | 含义 |
|------|------|
| `allow_origins=["*"]` | 允许所有来源(**仅限开发**,生产危险) |
| `allow_credentials=True` | 允许前端发 cookie |
| `allow_methods=["*"]` | 允许所有 HTTP 方法 |
| `allow_headers=["*"]` | 允许所有请求头 |

### 8.4.3 中间件(Middleware)

中间件是"每个请求都会过一遍"的钩子,可以加日志、改请求、统计耗时。

```python
import time

@app.middleware("http")
async def add_timing_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Duration"] = str(duration)
    return response
```

每个响应都会带 `X-Duration` 头,告诉客户端这次请求耗时多少。

### 8.4.4 依赖注入:把"重复的事"抽出来

```python
from fastapi import Depends

def get_db():
    db = "fake-connection"
    try:
        yield db
    finally:
        # 关闭连接
        pass

@app.get("/users")
def list_users(db=Depends(get_db)):
    return {"db": db}
```

> `Depends(get_db)` 告诉 FastAPI:"这个接口需要先用 get_db 拿个 db,接口结束时自动清理"。

### 8.4.5 部署:从 uvicorn 到生产

```bash
# 开发:单进程 + 自动重启
uvicorn app.api:app --reload

# 生产:多 worker
gunicorn app.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

**反代**:
- Nginx 转发请求到 uvicorn
- Cloudflare 做 HTTPS
- Docker 打包整个应用

---

## 8.5 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| 422 Unprocessable Entity | 请求数据不满足 Pydantic 模型 | 看错误体里的 `detail` 字段,照着改 |
| 启动报 `ModuleNotFoundError: No module named 'fastapi'` | 没装依赖 | `pip install fastapi uvicorn pydantic` |
| 浏览器访问 `/` 看到 404 | URL 路径写错 | 看接口注册时的路径,核对大小写 |
| CORS 报错 | 前端和后端 origin 不同 | 加 CORSMiddleware |
| SSE 接不到数据 | `media_type` 写错 | 必须是 `text/event-stream` |
| WebSocket 连不上 | 忘了 `await websocket.accept()` | 加 accept |
| `response_model` 改了不生效 | 函数返回值是 dict 而不是模型实例 | 让函数 `return Model实例` |

---

## 8.6 学习自检

能口述:
- [ ] HTTP 是什么?一次请求包含哪些部分?
- [ ] GET 和 POST 的区别?
- [ ] 路径参数、Query 参数、Body 怎么区分?各适合什么场景?
- [ ] Pydantic 帮你做了哪些事?
- [ ] `response_model` 有什么用?为什么要拆"内部模型"和"对外模型"?
- [ ] 一次性响应 vs 流式响应,什么时候用哪个?
- [ ] SSE 消息格式是什么?
- [ ] WebSocket 和 HTTP 的本质区别?
- [ ] CORS 跨域是什么?为什么会报错?

能写:
- [ ] 一个支持 GET + POST 的小接口
- [ ] 一个用 Pydantic 校验的登录接口
- [ ] 一个 SSE 流式接口,前端用 EventSource 收
- [ ] 一个 WebSocket 聊天接口

---

## 8.7 下一步

- 想接大模型 API:看 [DeepSeek](https://platform.deepseek.com/) / OpenAI / Anthropic 的官方文档
- 想做 LLM Agent:把 FastAPI 当后端,前面挂一个 Agent 框架(比如 LangChain / LlamaIndex)
- 想做更复杂的项目:加数据库(SQLAlchemy + SQLite/PostgreSQL)、加认证(JWT)、加前端(React / Vue)