"""04_sse.py
练习:SSE 流式响应。

任务:
    实现一个 /stream/countdown 接口,用 SSE 推送倒计时:
        - 第 1 个事件:event=start, data={"total": 10}
        - 接下来 10 个事件:event=delta, data={"remaining": i},每秒一个
        - 最后 1 个事件:event=done, data={"message": "倒计时结束"}

提示:
    - StreamingResponse(生成器, media_type="text/event-stream")
    - 用 time.sleep(1) 模拟 1 秒间隔
    - 复用 examples/06_sse_stream.py 里的 _sse_format 思路
"""

import json
import time
from typing import Iterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


def _sse_format(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def countdown(total: int) -> Iterator[str]:
    # 在这里写生成器逻辑
    pass


@app.get("/stream/countdown")
def stream_countdown(total: int = 10) -> StreamingResponse:
    # 在这里返回 StreamingResponse
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)