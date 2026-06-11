"""Live dashboard — zero-dependency HTTP server over local state.

Stdlib-only by design (no FastAPI/uvicorn): one ThreadingHTTPServer
serving a single-page UI (Chart.js from CDN) and a JSON API that reads
runner.db, the monitor state files and the permanent memory — all
read-only, all local. Protected by HTTP basic auth from .env.
"""

from trading.dashboard.app import build_summary, serve

__all__ = ["build_summary", "serve"]
