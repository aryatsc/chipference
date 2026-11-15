"""Record media/demo.mp4 from the live page, frame by frame, with headless Chrome.

    python3 tools/record.py            # needs Google Chrome and ffmpeg

Serves the repo locally, opens tools/frame.html (the page inside a browser-window frame) over the
Chrome DevTools Protocol, steps the page's animation clock 1/30 s at a time, screenshots each
frame, and encodes media/demo.mp4 and media/demo.gif. Standard library only.
"""
import base64
import functools
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FPS, SECONDS = 30, 8.0
W, H, SCALE = 1440, 810, 4 / 3        # CSS px and device scale -> 1920x1080 output
HTTP_PORT, CDP_PORT = 8767, 9333


class WebSocket:
    """Just enough of RFC 6455 for a local CDP connection."""

    def __init__(self, url):
        u = urllib.parse.urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.sock.recv(1)
        if b" 101 " not in resp.split(b"\r\n")[0]:
            raise RuntimeError(resp.decode(errors="replace"))

    def _read(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return bytes(buf)

    def send(self, text):
        data, mask = text.encode(), os.urandom(4)
        n, hdr = len(data), bytearray([0x81])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
        self.sock.sendall(bytes(hdr) + mask + masked)

    def recv(self):
        msg = b""
        while True:
            b1, b2 = self._read(2)
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            if b1 & 0x0F in (0x9, 0xA):   # ping/pong
                continue
            msg += payload
            if b1 & 0x80:
                return msg.decode()


class CDP:
    def __init__(self, ws):
        self.ws, self.next_id = ws, 0

    def call(self, method, **params):
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")


def main():
    frames = tempfile.mkdtemp(prefix="chipference-frames-")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    profile = tempfile.mkdtemp(prefix="chipference-chrome-")
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={profile}",
                               "--hide-scrollbars", "--no-first-run", "--disable-gpu", f"--window-size={W},{H}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        cdp = CDP(WebSocket(page["webSocketDebuggerUrl"]))
        cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=SCALE, mobile=False)
        cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        cdp.call("Page.navigate", url=f"http://127.0.0.1:{HTTP_PORT}/tools/frame.html")
        for _ in range(200):
            if cdp.js("window.__ready === true"):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("frame.html never became ready")

        n = round(SECONDS * FPS)
        for i in range(n):
            cdp.js(f"window.__frame({i / FPS})")
            shot = cdp.call("Page.captureScreenshot", format="png", fromSurface=True)
            with open(os.path.join(frames, f"f{i:04d}.png"), "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            if i % 30 == 0:
                print(f"frame {i}/{n}", flush=True)
    finally:
        chrome.terminate()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)

    media = os.path.join(ROOT, "media")
    os.makedirs(media, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames, "f%04d.png"),
                    "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(media, "demo.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(media, "demo.mp4"), "-vf",
                    "fps=15,scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                    os.path.join(media, "demo.gif")], check=True)
    print("frames in", frames)
    print("wrote media/demo.mp4 and media/demo.gif")


if __name__ == "__main__":
    sys.exit(main())
