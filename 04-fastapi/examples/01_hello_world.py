"""01_hello_world.py
你的第一个 FastAPI 应用。

学习目标:
    1. 理解 FastAPI 的"两件套":FastAPI() 实例 + @app.get 装饰器
    2. 学会用 uvicorn 把应用跑起来
    3. 知道什么是"路由",什么是"接口"

运行方式:
    uvicorn examples.01_hello_world:app --reload
    浏览器访问 http://127.0.0.1:8000/
    浏览器访问 http://127.0.0.1:8000/docs  ← FastAPI 自动生成的接口文档,免费送你的!
"""

from fastapi import FastAPI

# 第一步:创建一个 FastAPI 应用实例
# 你可以把它理解成"一台刚接通电源的服务器",现在还没装任何"业务"
app = FastAPI(
    title="我的第一个 API",
    description="这一章所有例子的基础模板",
    version="0.1.0",
)


# 第二步:给这台服务器装一个"业务"——一个接口(也叫"路由")
# @app.get("/") 表示:当有人用浏览器访问根路径 "/" 时,执行下面的函数
# "GET" 是 HTTP 协议里的"请求方法"之一,你可以先记成"看"
@app.get("/")
def home() -> dict:
    """根路径接口:返回一句欢迎语。

    浏览器访问 http://127.0.0.1:8000/ 会看到这个函数的返回值(JSON 格式)。
    """
    return {"message": "你好,这是你的第一个 FastAPI 接口!"}


# 再装一个接口,这次路径是 "/about"
@app.get("/about")
def about() -> dict:
    """/about 接口:返回应用信息。"""
    return {"name": "hello-world", "version": "0.1.0"}


# ---- 直接运行这个文件也能跑(下面这行是为了方便"看效果") ----
# 命令行下推荐用: uvicorn examples.01_hello_world:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)