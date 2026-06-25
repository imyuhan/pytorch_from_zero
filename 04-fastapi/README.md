# 08 - FastAPI 速成 (Web 后端入门)

> **难度**: ⭐⭐
> **前置**: 无(只要会 Python 基础)
> **预计耗时**: 90 - 120 分钟
> **硬件**: 不需要 GPU,任何机器都行
>
> 📖 详细讲解看 [`doc/04-fastapi.md`](doc/04-fastapi.md)

学完这一章你能:
- 听懂"接口"、"路由"、"请求体"、"流式响应"这些词不再犯怵
- 用 FastAPI 写出能"给别人调用"的 HTTP 服务
- 处理 JSON 请求、自动校验字段、流式推送、WebSocket 双向通信

## 核心概念(脑图速记)

| 术语 | 一句话解释 |
|------|-----------|
| **HTTP** | 浏览器/前端和服务端之间的"一问一答"协议 |
| **路径参数** | URL 路径里的变量,比如 `/users/42` 的 `42` |
| **Query 参数** | URL 里 `?` 后面的 `key=value`,比如 `?q=fastapi&limit=10` |
| **Pydantic** | Python 的数据校验库,FastAPI 用它定义请求/响应的"数据形状" |
| **请求体(Body)** | POST 请求里携带的数据,通常是 JSON |
| **`response_model`** | 告诉 FastAPI"这个接口返回的数据长什么样",自动过滤敏感字段 |
| **SSE 流式响应** | 服务端"边算边发",前端实时看到结果(ChatGPT 的打字机效果) |
| **WebSocket** | 浏览器和服务器之间的"双向通话",两边可以随时互发消息 |

## 课程大纲

| 序号 | 主题 | 重点 | 难点 |
|------|------|:----:|:----:|
| 1 | Hello World:最小 FastAPI 应用 | ✅ |  |
| 2 | 路径参数 / Query 参数 | ✅ |  |
| 3 | POST 请求 + Pydantic 请求体 | ✅ |  |
| 4 | `Literal` / `Optional` / `Union` / 自定义校验器 |  | ✅ |
| 5 | `response_model` 控制响应结构 | ✅ |  |
| 6 | SSE 流式响应(`text/event-stream`) | ✅ | ✅ |
| 7 | WebSocket 双向通信 |  | ✅ |

## 最小示例(Hello World)

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "你好,FastAPI!"}
```

启动:
```bash
uvicorn examples.01_hello_world:app --reload
```
浏览器访问 http://127.0.0.1:8000/ 看到 `{"message": "你好,FastAPI!"}`。
访问 http://127.0.0.1:8000/docs **免费获得自动生成的接口文档**。

## 安装

```bash
pip install fastapi uvicorn[standard] pydantic
```

- `fastapi`:框架本体
- `uvicorn`:启动服务的"引擎"
- `pydantic`:数据校验

## 跑例子

每个 `examples/` 下的脚本都能独立跑。建议顺序:

```bash
# 装依赖(只需要装一次)
pip install fastapi "uvicorn[standard]" pydantic

# 启动任一例子(把文件名换成你想跑的那个)
uvicorn examples.01_hello_world:app --reload
```

浏览器打开 http://127.0.0.1:8000/ 看效果,http://127.0.0.1:8000/docs 看接口文档。

## 详细讲解请看 `examples/` 下 7 个脚本

每个脚本顶部的 docstring 写了:
- 这个例子讲什么
- 怎么跑
- 怎么测

## 练习

`exercises/` 下 5 道题,做完就基本掌握 FastAPI 日常用法。