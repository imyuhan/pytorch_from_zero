"""08_auth_api_key.py
用 Depends + Header 抽出 API Key 鉴权逻辑。

学习目标:
    1. Header(...) 怎么从 HTTP 请求头里拿值
    2. Depends 怎么把"鉴权/取 db/读配置"这类重复逻辑抽出来
    3. 为什么把鉴权写成一个独立函数,而不是塞进每个接口里

运行方式:
    uvicorn examples.08_auth_api_key:app --reload

测试方式:
    # 不带 Key,应该 422(Header 必填)
    curl http://127.0.0.1:8000/secret

    # 带错 Key,应该 401
    curl http://127.0.0.1:8000/secret -H "X-API-Key: wrong-key"

    # 带对 Key,应该 200
    curl http://127.0.0.1:8000/secret -H "X-API-Key: secret-key-123"
"""

from fastapi import FastAPI, Header, HTTPException, Depends

# ============================================================
# 一、为什么需要"鉴权抽出来"?
# ============================================================
# 假设你有 10 个接口都要鉴权,每个接口都写一遍:
#
#     @app.get("/a")
#     def a(x_api_key: str = Header(...)):
#         if x_api_key != "secret-key-123":
#             raise HTTPException(401)
#         ...
#
#     @app.get("/b")
#     def b(x_api_key: str = Header(...)):
#         if x_api_key != "secret-key-123":
#             raise HTTPException(401)
#         ...
#
# 问题:
#   1. 重复 —— 同一段校验逻辑写了 10 遍
#   2. 改 Key 要改 10 个地方
#   3. 哪天想换成 JWT,所有接口都要重写
#
# 解决:把鉴权抽成一个独立函数,用 Depends 注入到需要的接口里。


# ============================================================
# 二、配置(实际项目里会放在 config.py / 环境变量里)
# ============================================================
API_KEY = "secret-key-123"   # 假设数据库/配置中心里存的"真 Key"


# ============================================================
# 三、鉴权函数:核心就三行
# ============================================================
def verify_api_key(x_api_key: str = Header(..., description="从请求头 X-API-Key 读取")) -> str:
    """校验请求头里的 API Key 是否正确。

    几个关键点:
    1. 函数参数 `x_api_key` 默认值是 Header(...),FastAPI 看到 Header 就会
       从 HTTP 请求头里取名字对应的字段(自动小写匹配,下划线转中划线)
    2. Header(...) 里的 `...` 表示"必填",没带这个头 → 直接 422
    3. 返回值:验证通过的话把 key 返回出去,后续接口可以用(比如"查到这是哪个调用方")
    4. 验证失败 → raise HTTPException,FastAPI 会自动转成 JSON 错误响应
    """
    if x_api_key != API_KEY:
        # 401 Unauthorized = "你没权限"
        # 403 Forbidden = "你登录了但没权限做这个事"
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return x_api_key   # 返回的 value 会传给依赖它的接口


# ============================================================
# 四、方式 1:用 dependencies=[...] —— 只验证,接口拿不到 key
# ============================================================
app = FastAPI(title="API Key 鉴权示例")

@app.get("/secret", dependencies=[Depends(verify_api_key)])
def read_secret() -> dict:
    """受保护的接口 —— 只想校验权限,不需要在函数里用 key 本身。

    dependencies=[Depends(verify_api_key)] 告诉 FastAPI:
        "这个接口被调用前,先跑 verify_api_key 验证一下,通过才能进"
    verify_api_key 的返回值这里用不到,所以 dependencies 写法最干净。
    """
    return {"message": "恭喜,你通过了鉴权,这是秘密数据"}


# ============================================================
# 五、方式 2:用 x_api_key: str = Depends(verify_api_key) —— 拿返回值
# ============================================================
# 适合"鉴权后还想知道你是哪个调用方"的场景
@app.get("/whoami")
def whoami(api_key: str = Depends(verify_api_key)) -> dict:
    """鉴权 + 拿到 key 本身(演示 Depends 传值)。

    Depends(verify_api_key) 做了两件事:
        1. 调用 verify_api_key
        2. 把它的返回值赋给参数 api_key
    """
    return {"your_key": api_key, "key_prefix": api_key[:6] + "***"}


# ============================================================
# 六、Depends 的进阶:链式依赖(底层机制)
# ============================================================
# Depends 还可以嵌套 —— 一个 Depends 调用另一个 Depends。
# 实际项目里常见的链:
#
#   接口 → Depends(get_current_user)    ← 验证 token,返回 user 对象
#           → Depends(get_db)            ← 拿数据库连接
#               → Depends(get_settings)  ← 读配置
#
# 每层各管一摊,接口函数本身很干净。

# 演示:一个"配置依赖",被"鉴权依赖"依赖
from typing import Annotated

def get_settings() -> dict:
    """假装从环境变量读配置。"""
    return {"env": "dev", "version": "1.0.0"}

def verify_api_key_with_settings(
    x_api_key: str = Header(...),
    settings: dict = Depends(get_settings),  # ← 嵌套 Depends
) -> dict:
    """鉴权 + 顺便拿到配置。"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    if settings["env"] == "dev":
        # 开发环境放水,任何 key 都通过(演示用,生产别这么写!)
        return {"key": x_api_key, "env": settings["env"], "note": "dev mode: 跳过严格校验"}
    return {"key": x_api_key, "env": settings["env"]}

@app.get("/profile")
def profile(info: dict = Depends(verify_api_key_with_settings)) -> dict:
    """链式依赖示例 —— 鉴权函数又依赖配置函数。"""
    return info


# ============================================================
# 七、为什么参数名是 x_api_key,Header 名是 X-API-Key?
# ============================================================
# HTTP Header 不区分大小写,但约定用"X-前缀 + 中划线"风格:
#   X-API-Key / X-Request-ID / X-User-Token
#
# Python 函数参数不能用中划线,所以 FastAPI 做了"自动转换":
#   函数参数  x_api_key  →  HTTP Header  x-api-key
#   转换规则:下划线 → 中划线,字母全小写
#
# 所以你写 Header(...) 时,只要参数名是合法的 Python 标识符就行,
# FastAPI 帮你对应到正确的 Header 名。
#
# 想精确指定 Header 名?用 alias:
#   def f(api_key: str = Header(..., alias="X-API-Key")):
#       ...
# 默认转换规则几乎覆盖所有情况,大多数项目不用 alias。


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
