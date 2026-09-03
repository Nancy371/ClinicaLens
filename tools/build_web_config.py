"""Write the static site's public API base without exposing any secret."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "config.js"


def main() -> int:
    value = os.getenv("CARE_API_BASE_URL", os.getenv("DEMO_API_BASE_URL", "")).strip().rstrip("/")
    if value:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SystemExit("CARE_API_BASE_URL must be an absolute http(s) URL")
    payload = "window.CLINICALENS_CONFIG = " + json.dumps(
        {
            "apiBaseUrl": value,
            "vapidPublicKey": os.getenv("CARE_VAPID_PUBLIC_KEY", "").strip(),
        },
        ensure_ascii=False,
        indent=2,
    ) + ";\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote public web config: apiBaseUrl={value or '<same-origin>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
