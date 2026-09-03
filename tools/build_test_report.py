"""Run the full unittest suite and publish a truthful static build artifact."""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "data" / "test-report.json"

# Running ``python tools/build_test_report.py`` sets sys.path[0] to ``tools``.
# Add the project root explicitly so discovery imports the same packages as CI.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    status = "passed" if result.wasSuccessful() else "failed"
    payload = {
        "schema_version": "test-report.v1",
        "status": status,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures) + len(result.errors),
        "skipped": len(result.skipped),
        "commit": os.getenv("RENDER_GIT_COMMIT", os.getenv("GIT_COMMIT", "local-workspace"))[:12],
        "generated_on": datetime.now(timezone.utc).isoformat(),
        "source": "Python unittest full-suite artifact",
        "reason": (
            f"Python {sys.version_info.major}.{sys.version_info.minor} 环境执行 unittest 全量回归。"
            if result.wasSuccessful()
            else "当前构建存在失败测试，不得展示全部通过。"
        ),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
