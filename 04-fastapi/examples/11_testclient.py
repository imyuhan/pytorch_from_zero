"""11_testclient.py
用 TestClient 在不起 TCP 端口的情况下测试 FastAPI 接口。

学习目标:
    1. TestClient 是什么、为什么不需要起端口
    2. 怎么测正常路径、错误路径、鉴权、流式响应
    3. 怎么和 pytest 结合 / 怎么纯 Python 跑

运行方式:
    # 方式 1:直接 python 跑(看输出)
    python -m examples.11_testclient

    # 方式 2:用 pytest 跑(更专业的做法)
    pip install pytest
    pytest examples/11_testclient.py -v
"""

# ============================================================
# 一、TestClient 是什么?(底层机制)
# ============================================================
# TestClient 来自 fastapi.testclient,内部其实是用 httpx + ASGI Transport。
#
# 关键事实:
#   - 不起 TCP 端口(不需要 listen socket)
#   - 不通过操作系统的网络栈
#   - 直接把 HTTP 请求"喂"给 ASGI 应用,FastAPI 处理完返回响应
#
# 流程对比:
#   浏览器/curl 测试:
#     浏览器 → TCP socket → uvicorn → FastAPI app
#                         (8000 端口)        ↑
#                                         你的代码
#
#   TestClient 测试:
#     TestClient → httpx → ASGI transport → FastAPI app
#                 (内存里直接传)              ↑
#                                          你的代码
#
# 好处:
#   - 不用起服务 → 测试更快、CI 里不用管理端口冲突
#   - 不依赖网络 → 离线也能跑
#   - 可以直接拿响应对象(r.status_code / r.json() / r.text)做断言
#
# 限制:
#   - 真正"起服务"的边界场景测不到(比如 nginx 反代、负载均衡)
#   - WebSocket 测试需要用 TestClient 的特殊上下文(本文件不展开)


# ============================================================
# 二、先定义一个待测的 FastAPI 应用
# ============================================================
# 生产项目里,这个 app 通常在 main.py,我们 import 进来就行。
# 写在同一文件里是为了"一个文件能跑"。
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Iterator

app = FastAPI(title="TestClient 演示应用")


# -- Schemas --
class CreateItem(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)


# -- 鉴权依赖 --
API_KEY = "test-key-456"

def verify_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid key")
    return x_api_key


# -- 数据(内存) --
_items: dict[int, dict] = {}
_id_seq = 0


# -- 接口 --
@app.get("/ping")
def ping() -> dict:
    return {"pong": True}

@app.get("/items/{item_id}")
def get_item(item_id: int) -> dict:
    if item_id not in _items:
        raise HTTPException(status_code=404, detail="not found")
    return _items[item_id]

@app.post("/items", response_model=dict, status_code=201)
def create_item(req: CreateItem) -> dict:
    global _id_seq
    _id_seq += 1
    _items[_id_seq] = {"id": _id_seq, "name": req.name, "price": req.price}
    return _items[_id_seq]

@app.get("/secret", dependencies=[Depends(verify_key)])
def secret() -> dict:
    return {"secret": "you passed"}

@app.get("/stream")
def stream() -> StreamingResponse:
    def gen() -> Iterator[str]:
        yield "event: start\ndata: {}\n\n"
        for ch in "hello":
            yield f"event: delta\ndata: {ch!r}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ============================================================
# 三、导入 TestClient + 写测试
# ============================================================
from fastapi.testclient import TestClient

client = TestClient(app)   # ← 关键:把 FastAPI app 包成 client


# -- 测试 1:最简单的 GET --
def test_ping():
    r = client.get("/ping")
    # 断言风格:断言失败时 pytest 会显示哪一行失败
    assert r.status_code == 200
    assert r.json() == {"pong": True}
    print("[test_ping] OK")


# -- 测试 2:路径参数 --
def test_get_item_found():
    # 先创建一个 item
    create = client.post("/items", json={"name": "book", "price": 9.9})
    item_id = create.json()["id"]

    # 再查
    r = client.get(f"/items/{item_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "book"
    print(f"[test_get_item_found] OK, id={item_id}")


# -- 测试 3:404 错误 --
def test_get_item_not_found():
    r = client.get("/items/9999")
    assert r.status_code == 404
    assert r.json()["detail"] == "not found"
    print("[test_get_item_not_found] OK")


# -- 测试 4:Pydantic 校验失败(422) --
def test_create_item_validation_error():
    # price 传负数,违反 gt=0
    r = client.post("/items", json={"name": "x", "price": -1})
    assert r.status_code == 422   # Unprocessable Entity
    # 错误体长这样:[{"loc": ["body", "price"], "msg": "...", "type": "value_error"}]
    body = r.json()
    assert "detail" in body
    print(f"[test_create_item_validation_error] OK, detail={body['detail']}")


# -- 测试 5:测试请求体缺字段(也是 422) --
def test_create_item_missing_field():
    r = client.post("/items", json={"name": "x"})   # 缺 price
    assert r.status_code == 422
    print("[test_create_item_missing_field] OK")


# -- 测试 6:鉴权缺失 → 注意:Header(...) 必填时缺 Header 返回 422(不是 401) --
def test_secret_no_key():
    """注意一个常见误区:Header(...) 必填时如果不带这个头,
    FastAPI 在参数校验阶段就会报 422(Pydantic 的"缺参数"错误),
    鉴权函数 verify_key 根本没被调用,所以不会返回 401。

    真正的 401 是"带了 key 但 key 是错的"——见 test_secret_wrong_key。
    """
    r = client.get("/secret")
    assert r.status_code == 422
    # 422 的错误体里 detail 是 list,标着 loc=["header","x-api-key"]
    body = r.json()
    assert any("x-api-key" in str(d.get("loc", [])) for d in body["detail"])
    print("[test_secret_no_key] OK (注意:这是 422,不是 401)")


# -- 测试 7:鉴权带错 key --
def test_secret_wrong_key():
    r = client.get("/secret", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    print("[test_secret_wrong_key] OK")


# -- 测试 8:鉴权带对 key --
def test_secret_ok():
    r = client.get("/secret", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert r.json() == {"secret": "you passed"}
    print("[test_secret_ok] OK")


# -- 测试 9:Query 参数 --
def test_query_params():
    # 这个接口没有专门演示,可以临时在 app 里加
    # 演示:用 TestClient 传 query
    r = client.get("/items/1")  # 实际项目里你可以测 /search?q=...
    print(f"[test_query_params] status={r.status_code}")


# -- 测试 10:流式响应 --
def test_streaming_response():
    r = client.get("/stream")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    # TestClient 会把流"攒起来"返回完整 text
    text = r.text
    assert "event: start" in text
    assert "event: delta" in text
    assert "event: done" in text
    print(f"[test_streaming_response] OK, text={text[:80]!r}...")


# -- 测试 11:用 with 做上下文(用于需要清理资源的场景) --
def test_context_manager():
    # TestClient 支持 with 语句,退出时会清理
    with TestClient(app) as c:
        r = c.get("/ping")
        assert r.status_code == 200
    print("[test_context_manager] OK")


# ============================================================
# 四、用 main 把所有测试串起来跑(不依赖 pytest 也能跑)
# ============================================================
if __name__ == "__main__":
    # 顺序跑,任何一个 assert 失败会抛 AssertionError
    test_ping()
    test_get_item_found()
    test_get_item_not_found()
    test_create_item_validation_error()
    test_create_item_missing_field()
    test_secret_no_key()
    test_secret_wrong_key()
    test_secret_ok()
    test_query_params()
    test_streaming_response()
    test_context_manager()
    print("\n所有测试通过!")


# ============================================================
# 五、用 pytest 跑(更专业的方式)
# ============================================================
# 把本文件所有 def test_xxx() 抽出来:
#
#   # tests/test_api.py
#   from fastapi.testclient import TestClient
#   from app.main import app     # ← 你的 FastAPI app
#
#   client = TestClient(app)
#
#   def test_ping():
#       assert client.get("/ping").status_code == 200
#
# 跑测试:
#   pytest tests/ -v
#
# pytest 会自动:
#   - 找所有 def test_xxx() 函数
#   - 各自独立跑(一个失败不影响其他)
#   - 显示哪一行 assert 失败
#   - 支持 fixture / parametrize / mock 等高级功能


# ============================================================
# 六、TestClient 不会测到的场景(避坑)
# ============================================================
# 1) CORS 头 —— TestClient 不会触发浏览器同源策略
#    想测 CORS,得真的用浏览器或加 Origin 头手动检查响应
#
# 2) 真实的网络层(超时、丢包、TLS) —— TestClient 走内存,测不到
#
# 3) uvicorn 启动阶段的错误(比如端口被占) —— TestClient 不起服务,测不到
#
# 4) 多进程 / 多 worker 的并发问题 —— TestClient 是单进程
#
# 这些"测不到"的,要么靠手动 smoke test,要么靠 e2e 测试(用 Playwright 等)。
