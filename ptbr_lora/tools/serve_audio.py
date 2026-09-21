"""serve_audio.py — servidor web leve para OUVIR os WAVs gerados (mobile-friendly).

Lista diretorios e mostra um <audio controls> para cada arquivo de audio, entao
da para navegar (samples/, checkpoints/, reference/) e ouvir pelo celular via
Tailscale. Somente leitura.

Uso:
  python ptbr_lora/tools/serve_audio.py --root "C:\\IA\\Breeze-tts\\training\\runs" --host 0.0.0.0 --port 8081
"""
from __future__ import annotations

import argparse
import html
import io
import os
import sys
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac")
CSS = """
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 10px; background: #101214; color: #e8e8e8; }
  h1 { font-size: 1.15rem; margin: 6px 0 12px; }
  a { color: #6cc6ff; text-decoration: none; }
  .item { margin: 8px 0; padding: 10px; background: #1b1f24; border-radius: 10px; }
  .d { font-weight: 600; font-size: 1.02rem; word-break: break-all; }
  audio { width: 100%; margin-top: 8px; }
  small { color: #93a1ad; }
  .crumb { margin-bottom: 10px; }
</style>
"""


class AudioHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def list_directory(self, path):  # noqa: ANN001
        try:
            entries = os.listdir(path)
        except OSError:
            self.send_error(404, "Cannot list directory")
            return None

        rel = os.path.relpath(path, self.directory)
        dirs = sorted((e for e in entries if os.path.isdir(os.path.join(path, e))),
                      key=str.lower)
        files = sorted((e for e in entries if not os.path.isdir(os.path.join(path, e))),
                       key=str.lower)

        out = ["<!doctype html><html><head><meta charset='utf-8'>",
               "<meta name='viewport' content='width=device-width, initial-scale=1'>",
               "<title>Audios</title>", CSS, "</head><body>"]
        out.append(f"<h1>🎧 {html.escape('' if rel == '.' else rel)}</h1>")
        if rel != ".":
            out.append("<div class='item crumb'><a href='../'>⬆ subir</a></div>")
        for d in dirs:
            out.append(f"<div class='item'><a class='d' href='{urllib.parse.quote(d)}/'>"
                       f"📁 {html.escape(d)}/</a></div>")
        for f in files:
            q = urllib.parse.quote(f)
            if f.lower().endswith(AUDIO_EXT):
                out.append(f"<div class='item'><div class='d'>{html.escape(f)}</div>"
                           f"<audio controls preload='none' src='{q}'></audio></div>")
            else:
                out.append(f"<div class='item'><a href='{q}'>📄 {html.escape(f)}</a></div>")
        out.append("</body></html>")

        body = "".join(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return io.BytesIO(body)


def main() -> None:
    ap = argparse.ArgumentParser(description="Servidor de audio (somente leitura).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"[audio] root invalido: {root}")
    handler = partial(AudioHandler, directory=root)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[audio] servindo {root} em http://{args.host}:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
