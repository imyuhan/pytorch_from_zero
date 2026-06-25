"""06_sse_stream.py
SSE 流式响应(Server-Sent Events):让接口"边算边发",前端实时看到结果。

学习目标:
    1. 理解什么是流式响应:和"一次性返回"有什么区别
    2. text/event-stream 协议格式
    3. FastAPI 用 StreamingResponse 实现流式

运行方式:
    uvicorn examples.06_sse_stream:app --reload

测试方式:
    浏览器打开 http://127.0.0.1:8000/test  ← JS EventSource 自动演示
    或:
    curl -N http://127.0.0.1:8000/stream  ← 终端能看到"打字机效果"
"""

import json
import time
from typing import Iterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

app = FastAPI(title="SSE 流式响应")


# ============================================================
# 一、流式响应 vs 一次性响应
# ============================================================
# 一次性响应:服务端把所有内容算完,再一次性发给前端(像发完整邮件)
# 流式响应:服务端算一个字发一个字,前端实时显示(像打电话)
#
# 适用场景:
#   - 大模型对话(ChatGPT 那种一个字一个字吐出来的)
#   - 长时间任务(进度条)
#   - 实时数据推送


# ============================================================
# 二、SSE 协议格式
# ============================================================
# 每条消息长这样:
#
#   event: <事件名>
#   data: <数据,通常是 JSON 字符串>
#   <空行>          ← 必须有两个换行,前端才知道这条消息结束了
#
# 前端用 EventSource('http://...') 自动监听,按 event 名触发回调


def _sse_format(event: str, data) -> str:
    """生成一条符合 SSE 协议的消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ============================================================
# 三、生成器函数:流式接口的核心
# ============================================================
def generate_story() -> Iterator[str]:
    """一个会"边生成边推送"的生成器。

    每次 yield 一段内容,前端就收到一段。
    """
    story = (
        "从前有一只小企鹅,它住在一个叫 PyTorch 的冰原上。"
        "它每天都练习用卷积神经网络识别企鹅同伴。"
        "有一天,它学会了反向传播,开心地滑了一跤。"
    )

    # 推送开始事件
    yield _sse_format("start", {"ts": time.time()})

    # 逐字推送
    for i, ch in enumerate(story):
        yield _sse_format("delta", {"text": ch})
        time.sleep(0.05)             # 模拟"打字机效果"

    # 推送结束事件
    yield _sse_format("done", {"total": len(story)})


@app.get("/stream")
def stream_story() -> StreamingResponse:
    """流式讲故事接口。

    StreamingResponse 第一个参数是"一个能持续产出内容的生成器"
    media_type 必须是 "text/event-stream"
    """
    return StreamingResponse(
        generate_story(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",      # 禁用缓存,不然流式效果出不来
            "X-Accel-Buffering": "no",          # 禁用 nginx 缓冲
        },
    )


# ============================================================
# 四、流式数字(进度条场景)
# ============================================================
def generate_progress(total: int) -> Iterator[str]:
    """推送任务进度 0 → 100%"""
    yield _sse_format("start", {"total": total})

    for i in range(1, total + 1):
        yield _sse_format("delta", {"percent": i * 100 // total})
        time.sleep(0.2)

    yield _sse_format("done", {"percent": 100})


@app.get("/progress")
def progress(total: int = 10) -> StreamingResponse:
    """进度条接口 —— total 表示一共多少步。"""
    return StreamingResponse(
        generate_progress(total),
        media_type="text/event-stream",
    )


# ============================================================
# 五、给浏览器看的演示页面
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>SSE 演示</title></head>
<body>
    <h2>流式故事(一个字一个字出来)</h2>
    <div id="story" style="font-size:18px;line-height:1.8"></div>

    <h2>进度条</h2>
    <progress id="bar" max="100" value="0"></progress>
    <span id="pct"></span>

    <script>
        // 监听故事流
        const es1 = new EventSource("/stream");
        const storyEl = document.getElementById("story");
        es1.addEventListener("delta", e => {
            storyEl.textContent += JSON.parse(e.data).text;
        });
        es1.addEventListener("done", () => es1.close());

        // 监听进度流
        const es2 = new EventSource("/progress?total=10");
        const bar = document.getElementById("bar");
        const pct = document.getElementById("pct");
        es2.addEventListener("delta", e => {
            const v = JSON.parse(e.data).percent;
            bar.value = v;
            pct.textContent = v + "%";
        });
        es2.addEventListener("done", () => es2.close());
    </script>
</body>
</html>
"""


@app.get("/test", response_class=HTMLResponse)
def test_page() -> str:
    """浏览器访问 /test 看演示。"""
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)