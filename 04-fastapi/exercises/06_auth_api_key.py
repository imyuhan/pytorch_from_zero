"""06_auth_api_key.py
练习:用 Depends + Header 抽出 API Key 鉴权。

任务:
    1. 定义一个固定的 API_KEY = "my-secret-2026"
    2. 写一个 verify_api_key 函数:
       - 用 Header(...) 从请求头 X-API-Key 读取
       - 校验通过 → 返回 key
       - 校验失败 → raise HTTPException(401, detail="无效的 API Key")
    3. 写两个接口:
       - GET /admin  —— 用 dependencies=[Depends(verify_key)] 鉴权
       - GET /me     —— 用 x_api_key: str = Depends(verify_key) 拿到 key
                       返回 {"your_key_prefix": api_key[:4] + "***"}
    4. 用 curl 测试三种场景:不带 key / 错 key / 对 key

提示:
    - from fastapi import Header, HTTPException, Depends
    - 函数参数名建议叫 x_api_key,FastAPI 自动对应到 X-API-Key 头
    - 401 = Unauthorized,表示"你没权限"
"""

from fastapi import FastAPI, Header, HTTPException, Depends

app = FastAPI(title="API Key 鉴权练习")

# 1) 在这里定义 API_KEY
API_KEY = "my-secret-2026"


# 2) 在这里写 verify_api_key 函数
def verify_api_key(x_api_key: str = Header(...)) -> str:
    pass


# 3) 在这里写两个接口
@app.get("/admin", dependencies=[Depends(verify_api_key)])
def admin() -> dict:
    return {"message": "管理员接口,通过鉴权"}


@app.get("/me")
def me(api_key: str = Depends(verify_api_key)) -> dict:
    return {"your_key_prefix": api_key[:4] + "***"}


# 测试用例(运行后用 curl 验证)
#   curl http://127.0.0.1:8000/admin                          # 应 401 或 422
#   curl http://127.0.0.1:8000/admin -H "X-API-Key: wrong"   # 应 401
#   curl http://127.0.0.1:8000/admin -H "X-API-Key: my-secret-2026"  # 应 200
#   curl http://127.0.0.1:8000/me    -H "X-API-Key: my-secret-2026"  # 应 200


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
