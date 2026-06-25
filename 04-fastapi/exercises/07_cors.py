"""07_cors.py
练习:配置 CORS 放行本地前端。

任务:
    1. 用 CORSMiddleware 配置:
       - 放行 http://localhost:3000 和 http://localhost:5173(Vite 默认)
       - 允许带 cookie(allow_credentials=True)
       - 允许 GET/POST/PUT/DELETE
       - 允许所有自定义头
    2. 写一个 GET /api/hello 接口,返回 {"hello": "world"}
    3. 用 curl 模拟浏览器,加 -H "Origin: http://localhost:3000" 看响应头里
       是不是有 Access-Control-Allow-Origin

提示:
    - from fastapi.middleware.cors import CORSMiddleware
    - app.add_middleware(...)
    - 中间件要在所有路由注册之前 add(本文件结构已经把 add 写在最上面)
    - 真实项目里 allow_origins 别用 ["*"],会有安全风险
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CORS 练习")

# 1) 在这里加 CORS 中间件
#    (注意:中间件要先于路由注册)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# 2) 在这里写 /api/hello
@app.get("/api/hello")
def hello() -> dict:
    return {"hello": "world"}


# 验证方式(用 curl 模拟浏览器):
#   curl http://127.0.0.1:8000/api/hello -H "Origin: http://localhost:3000" -v
#   观察响应头,应看到:
#     < Access-Control-Allow-Origin: http://localhost:3000
#     < Access-Control-Allow-Credentials: true


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
