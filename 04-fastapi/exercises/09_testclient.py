"""09_testclient.py
练习:用 TestClient 写接口测试。

任务:
    给定下面的 FastAPI app(已经写好,不用动),用 TestClient 写 4 个测试:
    
    1. test_create_item_ok:
       POST /items,body={"name":"book","price":9.9}
       断言 status==201,响应里的 id 字段 >= 1
    
    2. test_create_item_bad_price:
       POST /items,body={"name":"book","price":-1}
       断言 status==422(Pydantic 校验失败)
    
    3. test_get_item_404:
       GET /items/9999
       断言 status==404
    
    4. test_list_items:
       GET /items
       断言 status==200,响应是 list

提示:
    - from fastapi.testclient import TestClient
    - client = TestClient(app)
    - client.post("/items", json={...})  ← json= 自动转 JSON 并加 Content-Type 头
    - r.status_code / r.json()
    - 写完用 python -m examples.09_testclient_exercise 验证(或 pytest 跑)
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


# ---- 待测的 app(已写好,不用改) ----
class CreateItem(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)


_items: list[dict] = []
_id_seq = 0


@app.post("/items", status_code=201)
def create_item(req: CreateItem) -> dict:
    global _id_seq
    _id_seq += 1
    item = {"id": _id_seq, "name": req.name, "price": req.price}
    _items.append(item)
    return item


@app.get("/items/{item_id}")
def get_item(item_id: int) -> dict:
    for it in _items:
        if it["id"] == item_id:
            return it
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="not found")


@app.get("/items")
def list_items() -> list[dict]:
    return _items


# ---- 在这里写测试 ----
from fastapi.testclient import TestClient

client = TestClient(app)


def test_create_item_ok():
    # TODO
    pass


def test_create_item_bad_price():
    # TODO
    pass


def test_get_item_404():
    # TODO
    pass


def test_list_items():
    # TODO
    pass


if __name__ == "__main__":
    # 顺序跑,看输出
    test_create_item_ok()
    test_create_item_bad_price()
    test_get_item_404()
    test_list_items()
    print("\n所有测试通过!")
