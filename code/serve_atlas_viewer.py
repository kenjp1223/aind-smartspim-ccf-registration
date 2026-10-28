"""
Serve one or more atlas viewer folders (from build_atlas_viewer.py) with the
Neuroglancer app, so any station on the network can open them.

One viewer:
  uv run python serve_atlas_viewer.py "<viewer folder>" ^
      --client "C:\\Users\\stuberadmin\\neuroglancer-catalog\\catalog\\neuroglancer_client"

Several viewers behind one server (a landing page links them):
  uv run python serve_atlas_viewer.py ^
      --viewer KOR="\\\\...\\LINCS_series\\atlas_viewer_KOR" ^
      --viewer "Calcr+Pdyn=\\\\...\\LINCS_series\\atlas_viewer_main" ^
      --client "C:\\Users\\stuberadmin\\neuroglancer-catalog\\catalog\\neuroglancer_client"
  (a bare folder with no NAME= is named after its folder)

Then open http://<this computer's IP>:8091/. Ctrl+C stops it. Read-only:
only files under the viewer folders and the Neuroglancer app are served.
"""

import argparse
import functools
import html
import http.server
import os
import posixpath
import re
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote

DEFAULT_CLIENT = Path(__file__).resolve().parents[2] / "neuroglancer-catalog" / "catalog" / "neuroglancer_client"


def safe_join(base, url_rest):
    """Map the URL portion after a mount prefix to a file under `base`, with no escape.

    Decodes %xx, collapses '.'/'..' against a leading '/' so nothing can climb above
    the mount, then confirms the real path stays inside `base` (also stops symlinks
    inside the folder from pointing out). Returns None if it would escape.
    """
    rel = unquote(url_rest).split("?", 1)[0].split("#", 1)[0]
    rel = posixpath.normpath("/" + rel.lstrip("/")).lstrip("/") or "index.html"
    base_real = os.path.realpath(base)
    full = os.path.realpath(os.path.join(base_real, *rel.split("/")))
    if full != base_real and not full.startswith(base_real + os.sep):
        return None
    return full


class Handler(http.server.SimpleHTTPRequestHandler):
    client_dir = None
    viewers = {}        # slug -> folder path (str)
    landing = b""       # landing-page HTML (empty when only one viewer)

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/ng" or clean.startswith("/ng/"):
            return safe_join(self.client_dir, clean[len("/ng"):]) or os.devnull
        # /<slug>/rest  -> that viewer's folder
        m = re.match(r"/([^/]+)(/.*)?$", clean)
        if m and m.group(1) in self.viewers:
            return safe_join(self.viewers[m.group(1)], m.group(2) or "") or os.devnull
        return super().translate_path(path)

    def do_GET(self):
        clean = self.path.split("?", 1)[0].split("#", 1)[0]
        if self.landing and clean in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.landing)))
            self.end_headers()
            self.wfile.write(self.landing)
            return
        super().do_GET()

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


def slugify(name, taken):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "viewer"
    base, n = s, 2
    while s in taken:
        s, n = f"{base}-{n}", n + 1
    return s


def landing_page(viewers):
    rows = "\n".join(
        f'<li><a href="/{html.escape(slug)}/">{html.escape(name)}</a> '
        f'<span class=p>{html.escape(str(folder))}</span></li>'
        for slug, (name, folder) in viewers.items())
    return (f"""<!doctype html><html><head><meta charset="utf-8"><title>Atlas viewers</title>
<style>body{{font:15px system-ui,sans-serif;margin:2em;max-width:60em;background:#111;color:#ddd}}
a{{color:#7cf;font-size:1.1em}}li{{margin:.6em 0}}.p{{color:#666;font-size:.8em;margin-left:.5em}}</style>
</head><body><h2>Atlas-space viewers</h2><ul>{rows}</ul>
<p style="color:#999">Each link opens that cohort's views (overlay, side by side, each brain).</p>
</body></html>""").encode("utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="a single viewer folder (or use --viewer for several)")
    ap.add_argument("--viewer", action="append", default=[], metavar="NAME=FOLDER",
                    help="a named viewer folder; repeat for several (NAME= optional)")
    ap.add_argument("--client", default=str(DEFAULT_CLIENT), help="Neuroglancer app folder (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--bind", default="0.0.0.0", help="0.0.0.0 = reachable from other stations; 127.0.0.1 = this PC only")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    client = Path(args.client)
    if not (client / "index.html").is_file():
        raise SystemExit(f"Neuroglancer app not found in {client}; pass --client")

    specs = list(args.viewer)
    if args.folder:
        specs.insert(0, args.folder)
    if not specs:
        raise SystemExit("give a viewer folder, or one or more --viewer NAME=FOLDER")

    viewers = {}   # slug -> (display name, folder Path)
    for spec in specs:
        name, sep, raw = spec.partition("=")
        folder = Path((raw if sep else name).rstrip("/\\"))
        display = name if sep else folder.name
        if not (folder / "views.json").is_file():
            raise SystemExit(f"{folder} is not an atlas viewer folder (no views.json)")
        viewers[slugify(display, viewers)] = (display, folder)

    host = lan_ip() if args.bind == "0.0.0.0" else "127.0.0.1"
    Handler.client_dir = str(client)

    if len(viewers) == 1:
        # single viewer served at the root, as before
        (_name, folder), = viewers.values()
        Handler.viewers = {}
        Handler.landing = b""
        handler = functools.partial(Handler, directory=str(folder))
        start_url = f"http://{host}:{args.port}/"
        print(f"Serving {folder}\n  open {start_url}  (Ctrl+C to stop)")
    else:
        Handler.viewers = {slug: str(folder) for slug, (_n, folder) in viewers.items()}
        Handler.landing = landing_page(viewers)
        handler = Handler
        start_url = f"http://{host}:{args.port}/"
        print("Serving viewers:")
        for slug, (name, folder) in viewers.items():
            print(f"  {name:<20} http://{host}:{args.port}/{slug}/   ({folder})")
        print(f"  landing page         {start_url}   (Ctrl+C to stop)")

    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(start_url)).start()
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
