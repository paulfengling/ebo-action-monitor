"""Local VLM proposes clip labels. Proposals are never automatically approved."""
import base64
import json
import os
import urllib.request
import cv2
from .common import rows, write_rows
from .data import resolve_video
from .video import read_window


def propose(manifest, output, cfg, endpoint, model):
    if not endpoint or not model:
        raise ValueError("请在 .env 设置 VLM_URL/VLM_MODEL，或指定服务参数")
    result = []
    labels = [c["id"] for c in cfg["classes"]]
    for item in rows(manifest):
        row = dict(item)
        video = resolve_video(row, manifest)
        row["video"] = str(video.resolve())
        row["reviewed"] = False
        description = json.dumps(cfg["classes"], ensure_ascii=False)
        content = [{"type": "text", "text": f"以下帧按时间排列。判断整段的主要动作。类别定义：{description}。"
                   '只输出 JSON {"label":"类别id或unknown","reason":"可见证据"}；'
                   '动作混杂、看不清或不属于这些类别时输出unknown，不要猜测。'}]
        for frame in read_window(video, {**cfg["model"], "frames": 8}):
            h, w = frame.shape[:2]
            scale = min(1, 768 / max(h, w))
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("图像编码失败")
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
        payload = {"model": model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": 256}
        headers = {"Content-Type": "application/json"}
        if os.getenv("VLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VLM_API_KEY"]
        try:
            request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
            text = body["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            answer = json.loads(text)
            proposed = answer.get("label")
            if proposed not in labels + ["unknown"]:
                raise ValueError("invalid model label")
            row["proposed_label"] = proposed
            row["proposal_reason"] = str(answer.get("reason", ""))
        except Exception as error:
            row["proposed_label"] = "unknown"
            row["proposal_reason"] = "服务/解析错误：" + type(error).__name__
        result.append(row)
        # Persist progress on every clip; a service failure cannot lose completed proposals.
        write_rows(output, result)
        print(f"已生成 {len(result)} 条待人工确认标签", flush=True)
    return len(result)
