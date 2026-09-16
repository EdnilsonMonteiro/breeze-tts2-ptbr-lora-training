import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

tot = 0.0
for f in sorted((paths.SCRAPING_WORK / "diar").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    hs = {k: v["h_clean"] for k, v in d["speakers"].items()}
    tot += sum(hs.values())
    print(f"{d['tag'][:12]:<12} {d['elapsed_s']:>4}s  vozes={len(hs)}  "
          + " ".join(f"{k[-2:]}:{v:.2f}h" for k, v in sorted(hs.items(), key=lambda x: -x[1])))
print(f"TOTAL fala exclusiva: {tot:.2f}h")
