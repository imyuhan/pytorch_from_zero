"""09_cors_frontend.py
CORS 跨域:为什么浏览器会拦,以及怎么放行本地前端。

学习目标:
    1. CORS 是什么、为什么只在浏览器场景有
    2. 浏览器发请求时多出来的"预检 OPTIONS"是什么
    3. 怎么用 CORSMiddleware 放行 localhost:3000 这种本地前端
    4. 生产环境应该怎么配

运行方式:
    uvicorn examples.09_cors_frontend:app --reload

测试方式:
    # 1) 直接 curl,不带 Origin,不会触发 CORS 检查
    curl http://127.0.0.1:8000/api/data

    # 2) 模拟浏览器,带 Origin,会触发 CORS
    curl http://127.0.0.1:8000/api/data -H "Origin: http://localhost:3000" -v
    # 观察响应头里有没有: Access-Control-Allow-Origin: ...

    # 3) 浏览器演示页:打开 http://127.0.0.1:8000/demo
    #    页面里有"假装是另一个端口的前端"的 JS,fetch 调 /api/data
    #    在没有 CORS 配置时,浏览器 Console 会红字报 "blocked by CORS policy"
    #    加上 CORS 配置后,就能正常拿到数据
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="CORS 跨域示例")


# ============================================================
# 一、CORS 是什么?为什么只在浏览器场景常见?
# ============================================================
# CORS = Cross-Origin Resource Sharing(跨域资源共享)
#
# 浏览器有一道"同源策略"的安全门:
#   协议 + 域名 + 端口 三个全一样才算"同源"
#   例:http://localhost:8000 和 http://localhost:3000
#       ↑同协议  ↑同域名  ↑不同端口
#       → 浏览器判定为"跨域",默认禁止读响应
#
# **关键事实**:CORS 是浏览器的限制,不是后端的限制。
#   - 用 curl / Postman / Python requests 调,不会触发 CORS 检查
#   - 只有浏览器里的 fetch / axios / jQuery.ajax 会触发
#   - 服务端只要正确返回 CORS 头,浏览器才"放行"
#
# 为什么有这个限制?
#   防止你在 a.com 登录后,被 b.com 的恶意脚本偷数据
#   (没有 CORS,b.com 页面里 JS 能直接 fetch a.com 的接口拿你的信息)


# ============================================================
# 二、什么时候会触发 CORS 检查?(底层机制)
# ============================================================
# 浏览器发现"跨域 fetch"时,分两种情况:
#
# 1) 简单请求(GET/HEAD/POST + 简单 Header):
#    浏览器直接发请求,看响应头里有没有:
#        Access-Control-Allow-Origin: <你的 origin>
#    没有 → 浏览器拦截响应,JS 拿不到数据
#
# 2) 复杂请求(自定义 Header、非简单 Content-Type 等):
#    浏览器先发一个 OPTIONS "预检请求"(preflight)问服务端:
#        "我要发 POST + Content-Type: application/json,跨域,行吗?"
#    服务端回 200 + 几个 CORS 头表示"行"
#    浏览器才发真正的请求
#
# 实战里只要前端用了 axios 的 JSON POST,几乎一定会触发 preflight


# ============================================================
# 三、配置 CORSMiddleware
# ============================================================
# 加中间件(必须放在所有路由之前!否则可能不生效)
app.add_middleware(
    CORSMiddleware,
    # 允许的"来源"列表 —— 写 * = 允许所有(开发用,生产危险)
    # 生产环境必须写具体域名,例如:
    #   allow_origins=["https://my-frontend.com", "https://admin.my-frontend.com"]
    allow_origins=[
        "http://localhost:3000",      # React / Vue 常见 dev 端口
        "http://localhost:5173",      # Vite 默认端口
        "http://127.0.0.1:5500",      # VSCode Live Server 端口
    ],
    # 是否允许带 cookie
    # True 时,上面 allow_origins 不能是 *,必须写具体域名
    allow_credentials=True,
    # 允许的 HTTP 方法
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    # 允许的 Header(Authorization / X-API-Key 这类自定义头要列上)
    allow_headers=["*"],   # * = 允许所有自定义头
    # 预检结果缓存多久(秒)—— 期间不再发 OPTIONS
    max_age=600,
)


# ============================================================
# 四、几个业务接口
# ============================================================
@app.get("/api/data")
def get_data() -> dict:
    """一个普通的 GET 接口 —— 测试 CORS 用。"""
    return {"data": [1, 2, 3], "source": "fastapi"}


@app.post("/api/echo")
def echo(payload: dict) -> dict:
    """一个 POST 接口 —— 复杂请求,会触发 preflight OPTIONS。"""
    return {"received": payload}


# ============================================================
# 五、浏览器演示页:模拟"另一个端口的前端"
# ============================================================
# 重点:这个页面是后端提供的(也是 :8000),但下面的 JS 会用 fetch
# 假装是 :3000 的前端来请求 —— 用 127.0.0.1 替代自己端口就行,
# 在浏览器眼里,"不同端口" = 跨域。
#
# 想真正测跨域,需要起两个不同端口:
#   - 终端 1: uvicorn examples.09_cors_frontend:app --reload --port 8000
#   - 终端 2: 找一个简单 index.html 用 python -m http.server 3000
#     然后浏览器打开 http://localhost:3000 那个页面
DEMO_PAGE = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>CORS 演示</title></head>
<body style="font-family:sans-serif; max-width:800px; margin:40px auto;">
    <h1>CORS 跨域演示</h1>

    <h2>测试 1:GET 请求</h2>
    <button onclick="testGet()">调 GET /api/data</button>
    <pre id="r1">(未请求)</pre>

    <h2>测试 2:POST 请求(会触发 preflight)</h2>
    <button onclick="testPost()">调 POST /api/echo</button>
    <pre id="r2">(未请求)</pre>

    <h3>怎么真正测跨域?</h3>
    <ol>
        <li>启本服务在 8000 端口</li>
        <li>另开终端,跑 <code>python -m http.server 3000</code> 在某个空目录</li>
        <li>在那个空目录放一个 index.html,内容就是上面这些 JS,fetch 改成 http://127.0.0.1:8000/api/data</li>
        <li>浏览器打开 http://localhost:3000,打开 Console 点按钮</li>
        <li>不加 CORS 配置时:红字 "blocked by CORS policy"</li>
        <li>加上配置后:正常拿到数据</li>
    </ol>

    <script>
        async function testGet() {
            try {
                // fetch 默认不发 cookie,同源/跨域都会受 CORS 限制
                const r = await fetch("http://127.0.0.1:8000/api/data");
                const j = await r.json();
                document.getElementById("r1").textContent = JSON.stringify(j, null, 2);
            } catch (e) {
                document.getElementById("r1").textContent = "错误: " + e.message;
            }
        }

        async function testPost() {
            try {
                const r = await fetch("http://127.0.0.1:8000/api/echo", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({msg: "hello"}),
                });
                const j = await r.json();
                document.getElementById("r2").textContent = JSON.stringify(j, null, 2);
            } catch (e) {
                document.getElementById("r2").textContent = "错误: " + e.message;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    """演示页 —— 提示:真正测跨域需要起两个不同端口。"""
    return DEMO_PAGE


# ============================================================
# 六、生产环境 CORS 配置建议
# ============================================================
# 1) allow_origins 永远不要写 ["*"] + allow_credentials=True
#    这是个矛盾组合,FastAPI 会报错
#
# 2) allow_origins 写具体域名,不要写通配
#    反面:allow_origins=["*"]  → 任何网站都能调你的接口(只要不带 cookie)
#    正面:allow_origins=["https://my-app.com"]
#
# 3) 生产环境的常见组合:
#
#    app.add_middleware(
#        CORSMiddleware,
#        allow_origins=["https://my-app.com"],
#        allow_credentials=True,    # 登录态走 cookie 才需要 True
#        allow_methods=["GET", "POST", "PUT", "DELETE"],
#        allow_headers=["Authorization", "Content-Type"],
#        max_age=600,
#    )
#
# 4) 部署在 Nginx/Cloudflare 后面时,有时可以直接在 Nginx 层加 CORS 头
#    (避免每个服务都配),参考:
#    add_header Access-Control-Allow-Origin $http_origin;


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
