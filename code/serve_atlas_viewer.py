"""
Serve an atlas viewer folder (from build_atlas_viewer.py) with the
Neuroglancer app, so any station on the network can open it.

  uv run python serve_atlas_viewer.py "<viewer folder>" ^
      --client "C:\\Users\\stuberadmin\\neuroglancer-catalog\\catalog\\neuroglancer_client"

Then open http://<this computer's IP>:8091/ (the index lists the views).
Ctrl+C stops it. Read-only: only files under the viewer folder and the
Neuroglancer app are served.
"""

import argparse
import functools
import http.server
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

DEFAULT_CLIENT = Path(__file__).resolve().parents[2] / "neuroglancer-catalog" / "catalog" / "neuroglancer_client"


class Handler(http.server.SimpleHTTPRequestHandler):
    client_dir = None

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/ng" or clean.startswith("/ng/"):
            rel = clean[len("/ng"):].lstrip("/") or "index.html"
            return str(Path(self.client_dir) / rel)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "range, x-requested-with")
        self.send_header("Access-Control-Expose-Headers", "content-range, content-length, etag")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):  # quiet: only errors
        if args and str(args[1]).startswith(("4", "5")) and not str(args[0]).endswith((".zarray", ".zattrs", ".zgroup")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


Handler.extensions_map.update({".json": "application/json", ".wasm": "application/wasm", ".js": "text/javascript"})


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="viewer folder made by build_atlas_viewer.py")
    ap.add_argument("--client", default=str(DEFAULT_CLIENT), help="Neuroglancer app folder (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--bind", default="0.0.0.0", help="0.0.0.0 = reachable from other stations; 127.0.0.1 = this PC only")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    folder = Path(args.folder.rstrip("/\\"))
    client = Path(args.client)
    if not (folder / "views.json").is_file():
        raise SystemExit(f"{folder} is not an atlas viewer folder (no views.json)")
    if not (client / "index.html").is_file():
        raise SystemExit(f"Neuroglancer app not found in {client}; pass --client")
    Handler.client_dir = str(client)
    handler = functools.partial(Handler, directory=str(folder))
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    host = lan_ip() if args.bind == "0.0.0.0" else "127.0.0.1"
    url = f"http://{host}:{args.port}/"
    print(f"Serving {folder}\n  open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
