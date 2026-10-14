"""Optional real RTSP loopback test; starts only local child processes and cleans them up."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import imageio_ffmpeg
import torch
from ebo.common import config, save_json
from ebo.video import Stream
from ebo.recording import record
from ebo.inference import run


def wait_until(predicate, seconds=12):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise RuntimeError("RTSP verification timed out")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="本机 MediaMTX 可执行文件")
    parser.add_argument("--smoke-dir", default="runs/smoke_pretrained")
    parser.add_argument("--out", default="runs/rtsp_verification")
    parser.add_argument("--port", type=int, default=18554)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    server_cfg = out / "mediamtx.yml"
    server_cfg.write_text(f"rtspAddress: 127.0.0.1:{args.port}\nrtspTransports: [tcp]\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others: {{}}\n")
    url = f"rtsp://127.0.0.1:{args.port}/test"
    smoke = Path(args.smoke_dir).resolve()
    cfg = config(smoke / "config.yaml")
    server, publisher, reader = None, None, None
    log = open(out / "server.log", "w")

    def publish():
        return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                                 "-re", "-stream_loop", "-1", "-i", str(smoke/"test_0.mp4"),
                                 "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                                 "-g", "8", "-keyint_min", "8",
                                 "-pix_fmt", "yuv420p", "-an", "-f", "rtsp", "-rtsp_transport", "tcp", url],
                                stdout=subprocess.DEVNULL, stderr=log)

    try:
        server = subprocess.Popen([str(Path(args.server).resolve()), str(server_cfg)], stdout=log, stderr=log)
        time.sleep(.5)
        if server.poll() is not None:
            raise RuntimeError("本地 RTSP 服务未启动，请检查端口与服务配置")
        publisher = publish()
        reader = Stream(url, cfg).start()
        wait_until(lambda: reader.window() is not None)
        first_epoch = reader.window()[0]
        publisher.terminate()
        publisher.wait(timeout=5)
        publisher = None
        wait_until(lambda: reader.snapshot()[2] > first_epoch and reader.window() is None)
        publisher = publish()
        wait_until(lambda: reader.window() is not None and reader.window()[0] > first_epoch)
        reader.close()
        reader = None
        recorded = record(url, cfg, out / "recordings", seconds=8, clip_seconds=4, fps=8)
        events = []
        inference = run(str(smoke/"model/best.pt"), url, cfg, "cpu", str(out/"inference.jsonl"),
                        duration=12, allow_demo=True, on_result=events.append)
        if recorded["frames"] == 0 or inference["predictions"] == 0:
            raise RuntimeError("RTSP 录制或推理未产生结果")
        result = {"loopback_rtsp": True, "tcp_capture": True, "disconnect_clears_window": True,
                  "reconnected_new_epoch": True, "recording": recorded, "inference": inference,
                  "note": "本机真实 RTSP 协议测试，不代表现场摄像头与 RTX 5080 性能"}
        save_json(out / "verification.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if reader:
            reader.close()
        for process in (publisher, server):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        log.close()


if __name__ == "__main__":
    main()
