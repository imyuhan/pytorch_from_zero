"""05_websocket.py
练习:WebSocket 双向通信。

任务:
    实现一个 /ws/echo 接口:
        - 客户端发:"hello" → 服务端回:"你说了:hello"
        - 客户端发:"bye"   → 服务端回:"再见",然后主动关闭连接
        - 客户端发其他内容  → 服务端回:"听不懂"

提示:
    - @app.websocket("/ws/echo")
    - await websocket.accept() 同意连接
    - while True: await websocket.receive_text()
    - 用 websocket.close() 关闭连接
    - try / except WebSocketDisconnect 处理断开
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws/echo")
async def websocket_echo(websocket: WebSocket):
    # 在这里写 WebSocket 逻辑
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)