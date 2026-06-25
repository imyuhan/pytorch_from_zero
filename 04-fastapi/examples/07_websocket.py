"""07_websocket.py
WebSocket:浏览器和服务器之间的"双向通话",两边可以随时发消息。

学习目标:
    1. 理解 WebSocket vs HTTP 的本质区别
    2. FastAPI 用 @app.websocket 装饰器实现
    3. 双向消息的收发

运行方式:
    uvicorn examples.07_websocket:app --reload

测试方式:
    浏览器打开 http://127.0.0.1:8000/  ← 有可视化聊天界面
"""

import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="WebSocket 双工连接")


# ============================================================
# 一、HTTP vs WebSocket
# ============================================================
# HTTP:    客户端请求 → 服务端响应 → 连接关闭
#          像"打骚扰电话":你说一句就挂,下次再打
#
# WebSocket:连接建立后,两边可以**随时**互发消息
#          像"打电话":接通后,双方随便聊,直到有一边说"挂了"
#
# URL 协议:
#   - HTTP:  http://  /  https://
#   - WS:    ws://    /  wss://  (wss 是加密版,类比 https)
#
# 适用场景:
#   - 实时聊天
#   - 实时游戏
#   - 实时数据大屏
#   - 协作工具(多人同时编辑文档)


# ============================================================
# 二、最小 WebSocket 接口
# ============================================================
@app.websocket("/ws/echo")
async def websocket_echo(websocket: WebSocket):
    """回声接口 —— 服务端把客户端发来的每条消息原样回传。

    三步:
        1. accept()   —— 同意连接
        2. send_text  / receive_text  —— 发消息 / 收消息(可以反复)
        3. 异常处理 —— 客户端断开时优雅退出
    """
    await websocket.accept()
    try:
        while True:
            # 等客户端发消息
            data = await websocket.receive_text()
            # 把消息原样回传
            await websocket.send_text(f"你说: {data}")
    except WebSocketDisconnect:
        # 客户端主动断开,正常退出
        print("客户端断开连接")


# ============================================================
# 三、双向交互:服务端也主动发消息
# ============================================================
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """聊天接口 —— 服务端每收到一条消息,先确认收到,再回一条。"""
    await websocket.accept()

    # 服务端主动打个招呼
    await websocket.send_json({
        "role": "system",
        "content": "连接成功!开始聊天吧",
    })

    try:
        while True:
            # 收客户端消息(JSON 格式)
            msg = await websocket.receive_json()
            user_text = msg.get("content", "")

            # 1) 先 ack
            await websocket.send_json({
                "role": "system",
                "content": f"收到:{user_text}",
            })

            # 2) 模拟"思考中"
            await websocket.send_json({
                "role": "assistant",
                "content": f"你说的是「{user_text}」,我听懂了一半。",
            })
    except WebSocketDisconnect:
        print("聊天连接断开")


# ============================================================
# 四、浏览器演示页面
# ============================================================
WS_PAGE = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>WebSocket 演示</title></head>
<body>
    <h2>WebSocket 聊天演示</h2>
    <div id="log" style="height:300px;overflow:auto;border:1px solid #ccc;padding:8px"></div>
    <input id="in" placeholder="说点什么..." style="width:70%">
    <button id="btn">发送</button>

    <script>
        const log = document.getElementById("log");
        const input = document.getElementById("in");
        const btn = document.getElementById("btn");

        // 连接到 /ws/chat
        const ws = new WebSocket("ws://" + location.host + "/ws/chat");

        ws.onopen = () => log.innerHTML += "<p><i>已连接</i></p>";
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            log.innerHTML += `<p>[${msg.role}] ${msg.content}</p>`;
            log.scrollTop = log.scrollHeight;
        };
        ws.onclose = () => log.innerHTML += "<p><i>连接断开</i></p>";

        function send() {
            if (!input.value) return;
            ws.send(JSON.stringify({content: input.value}));
            input.value = "";
        }
        btn.onclick = send;
        input.onkeypress = (e) => { if (e.key === "Enter") send(); };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """浏览器访问根路径看演示。"""
    return WS_PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)