"""02_path_query.py
路径参数(Path Parameter)和 Query 参数(Query String)。

学习目标:
    1. 路径参数:URL 路径里的"变量",用 {参数名} 占位
    2. Query 参数:URL 里 ? 后面跟着的 key=value 形式
    3. 二者的区别:路径参数定位"哪个资源",Query 参数用来"筛选/分页/排序"

运行方式:
    uvicorn examples.02_path_query:app --reload

测试链接:
    http://127.0.0.1:8000/users/42                       ← 路径参数
    http://127.0.0.1:8000/users/42?role=admin            ← 路径参数 + Query
    http://127.0.0.1:8000/search?q=fastapi&limit=5       ← 多个 Query
"""

from fastapi import FastAPI

app = FastAPI(title="路径参数与 Query 参数")


# ============================================================
# 一、路径参数:URL 路径里的"变量"
# ============================================================
# /users/{user_id} 里的 {user_id} 就是"路径参数"占位符
# 访问 /users/42,user_id 就是 42;访问 /users/alice,user_id 就是 "alice"
@app.get("/users/{user_id}")
def get_user(user_id: int) -> dict:
    """根据用户 id 查用户信息。

    注意:函数签名里写了 user_id: int —— FastAPI 会自动做类型校验:
        - 访问 /users/42    → user_id=42 (int),正常
        - 访问 /users/abc   → 422 错误,告诉你"期望 int 但拿到 str"
    """
    return {"user_id": user_id, "name": f"用户-{user_id}"}


# ============================================================
# 二、Query 参数:URL 里 ? 后面的 key=value
# ============================================================
# 函数签名里,没有出现在路径里的、且有"默认值"的形参 = Query 参数
# 访问 /search?q=fastapi&limit=5 时:
#   - q = "fastapi"(必填,因为没默认值)
#   - limit = 5(可选,默认 10)
@app.get("/search")
def search(q: str, limit: int = 10) -> dict:
    """搜索接口。

    q: 搜索关键词(必填)
    limit: 返回多少条结果(可选,默认 10)
    """
    return {"query": q, "limit": limit, "results": [f"结果-{i}" for i in range(limit)]}


# ============================================================
# 三、混合:路径参数 + Query 参数
# ============================================================
@app.get("/users/{user_id}/posts")
def user_posts(user_id: int, status: str = "published", page: int = 1) -> dict:
    """查某个用户的文章列表。

    /users/42/posts?status=draft&page=2 表示:
        - 路径参数:user_id = 42
        - Query 参数:status="draft"(默认 published),page=2(默认 1)
    """
    return {
        "user_id": user_id,
        "status": status,
        "page": page,
        "posts": [f"文章-{i}" for i in range(page * 2 - 1, page * 2 + 1)],
    }


# ============================================================
# 四、可选 Query 参数:用 Optional
# ============================================================
from typing import Optional

@app.get("/filter")
def filter_items(category: Optional[str] = None, min_price: Optional[float] = None) -> dict:
    """筛选商品 —— 演示"可选 Query 参数"。

    不传任何参数也可以访问;传了就按传的值过滤。
    Optional[类型] = None 意味着"可以是这个类型,也可以是 None"
    """
    return {
        "filters": {
            "category": category,
            "min_price": min_price,
        },
        "matched": [],  # 假装匹配结果
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)