"""01_path_query.py
练习:路径参数 + Query 参数。

任务:
    实现 /users/{user_id}/orders 接口,要求:
        - 路径参数 user_id(int)
        - Query 参数 status(可选,默认 "pending",取值 "pending"/"shipped"/"delivered")
        - Query 参数 page(int,可选,默认 1)
        - 返回:{"user_id": ..., "status": ..., "page": ..., "orders": [...]}

提示:
    - 用 Literal 限制 status 取值
    - 路径参数写法:@app.get("/users/{user_id}/orders")
    - Query 参数就是函数参数 + 默认值
"""

from typing import Literal
from fastapi import FastAPI

app = FastAPI()


# 在这里写你的代码


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)