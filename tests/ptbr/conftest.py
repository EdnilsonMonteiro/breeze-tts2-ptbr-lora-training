import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "ptbr_lora" / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
