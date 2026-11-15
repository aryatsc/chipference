"""Static server for the repo that also accepts PNG frames from tools/record.html.

    python3 tools/record_server.py <frames_dir> [port]

Then open http://localhost:<port>/tools/record.html and encode, e.g.:

    ffmpeg -framerate 30 -i <frames_dir>/f%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p media/demo.mp4
"""
import http.server
import os
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "frames")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
os.makedirs(OUT, exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/frame":
            i = int(urllib.parse.parse_qs(u.query)["i"][0])
            with open(os.path.join(OUT, f"f{i:04d}.png"), "wb") as f:
                f.write(body)
        elif u.path == "/done":
            print("done", flush=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
