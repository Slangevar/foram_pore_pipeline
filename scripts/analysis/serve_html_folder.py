#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os

PORT = 8765
ROOT = Path("data/visualizations/image_volumes/MOM_7_01").resolve()
FILE = "MOM_7_01_3d_volume.html"

os.chdir(ROOT)
print(f"Serving {ROOT}")
print(f"Forward with: ssh -N -L {PORT}:127.0.0.1:{PORT} <user>@<cluster-login-host>")
print(f"Open locally: http://127.0.0.1:{PORT}/{FILE}")
ThreadingHTTPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler).serve_forever()
