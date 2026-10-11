"""Screenshot built MkDocs pages and every <figure class="arch"> in light and dark mode.

Usage:
    python screenshot_docs.py <site_dir> <out_dir> [page ...]

    page defaults to the Visual Guide pages, e.g. "visual/" "visual/vae/".
    Writes <page>_<scheme>.png (full page) and fig_<page>_<i>_<scheme>.png (one per figure),
    and prints console errors / failed requests per scheme. The GitHub releases API 404
    from Material's repo widget is expected and harmless offline.

Needs: pip install playwright && python -m playwright install chromium
"""

import functools
import http.server
import os
import socketserver
import sys
import threading

from playwright.sync_api import sync_playwright

DEFAULT_PAGES = ["visual/", "visual/vae/", "visual/vqvae/", "visual/pulse2pulse/"]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main() -> None:
    site, out = sys.argv[1], sys.argv[2]
    pages = sys.argv[3:] or DEFAULT_PAGES
    os.makedirs(out, exist_ok=True)

    srv = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=site))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme in ["light", "dark"]:
            ctx = browser.new_context(viewport={"width": 1300, "height": 900}, color_scheme=scheme)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.on("response", lambda r: errs.append("404 " + r.url) if r.status == 404 else None)
            pg.on("pageerror", lambda e: errs.append(str(e)))
            for url in pages:
                pg.goto(f"http://127.0.0.1:{port}/{url}")
                pg.wait_for_timeout(4500)  # let draw/snap animations progress
                name = url.strip("/").replace("/", "_") or "index"
                pg.screenshot(path=f"{out}/{name}_{scheme}.png", full_page=True)
                for i, fig in enumerate(pg.locator("figure.arch").all()):
                    fig.screenshot(path=f"{out}/fig_{name}_{i}_{scheme}.png")
            print(scheme, "errors:", errs)
        browser.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
