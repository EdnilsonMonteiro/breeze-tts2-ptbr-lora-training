"""Fase 5 - transcricao X.ai em lote + QC -> datasets/podcast/texts.csv."""
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import num2words
import requests

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

ROOT = paths.ARTIFACTS
WORK = ROOT / "dataScrapping" / "work"
WAVS = ROOT / "datasets" / "podcast" / "wavs"
WORDS_DIR = WORK / "api_words"
WORDS_DIR.mkdir(parents=True, exist_ok=True)
TEXTS_CSV = ROOT / "datasets" / "podcast" / "texts.csv"
STATE = WORK / "transcribe_state.jsonl"

API_URL = "https://api.x.ai/v1/stt"
WORKERS = 4
CHARS_PER_S = (6.0, 25.0)
MAX_DIGIT_RATIO = 0.30
DUR_TOL_S = 0.75
ALLOWED = re.compile(r"[^A-Za-zÀ-ÿ0-9 .,!?'’\-:;()\"]+")
NUM_RE = re.compile(r"(?<![\w])(\d{1,4})(?![\w])")

key = None
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("XAI_TRANSCRIBE_KEY"):
        key = line.split("=", 1)[1].strip()
if not key:
    sys.exit("[stt] XAI_TRANSCRIBE_KEY ausente")


def clean_text(t: str, dur: float) -> tuple[str, str | None]:
    t = unicodedata.normalize("NFC", t)
    t = t.replace("…", "...").replace("–", "-").replace("—", "-")
    t = ALLOWED.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" -")
    if not t:
        return "", "vazio"
    if not (CHARS_PER_S[0] <= len(t) / dur <= CHARS_PER_S[1]):
        return "", f"densidade {len(t)/dur:.1f} c/s"
    if sum(c.isdigit() for c in t) / len(t) > MAX_DIGIT_RATIO:
        return "", "excesso de digitos"
    t = NUM_RE.sub(lambda m: num2words.num2words(int(m.group(1)), lang="pt-BR"), t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, None


local = threading.local()


def sess() -> requests.Session:
    if not hasattr(local, "s"):
        local.s = requests.Session()
    return local.s


def transcribe(chunk: str, dur: float) -> dict:
    for attempt in range(3):
        try:
            r = sess().post(API_URL, headers={"Authorization": f"Bearer {key}"},
                            files={"file": (chunk, (WAVS / chunk).read_bytes(), "audio/wav")},
                            data={"format": "true", "language": "pt"}, timeout=180)
            if r.status_code == 200:
                j = r.json()
                if abs(float(j.get("duration", dur)) - dur) > DUR_TOL_S:
                    return {"chunk": chunk, "status": "rejected", "reason": "dur_mismatch"}
                text, why = clean_text(j.get("text", ""), dur)
                if why:
                    return {"chunk": chunk, "status": "rejected", "reason": why}
                (WORDS_DIR / f"{chunk}.json").write_text(
                    json.dumps(j, ensure_ascii=False), encoding="utf-8")
                return {"chunk": chunk, "status": "ok", "text": text}
            if r.status_code in (429, 403) or r.status_code >= 500:
                wait = [2, 5, 10][attempt] if r.status_code != 403 else [30, 60, 120][attempt]
                time.sleep(wait)
                continue
            return {"chunk": chunk, "status": "rejected", "reason": f"http {r.status_code}"}
        except requests.RequestException:
            time.sleep([2, 5, 10][attempt])
    return {"chunk": chunk, "status": "rejected", "reason": "timeout/rede"}


done = {}
if STATE.exists():
    for ln in STATE.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            done[d["chunk"]] = d

index = [json.loads(l) for l in (WORK / "chunks_index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
todo = [r for r in index if r["chunk"] not in done]
print(f"[stt] total={len(index)} feitos={len(done)} a_transcrever={len(todo)}", flush=True)

state_f = STATE.open("a", encoding="utf-8")
lock = threading.Lock()
t0 = time.time()
cnt = {"ok": 0, "rejected": 0, "reasons": {}}

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(transcribe, r["chunk"], r["dur"]): r for r in todo}
    for fut in as_completed(futs):
        rec = fut.result()
        with lock:
            state_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state_f.flush()
            cnt[rec["status"]] += 1
            if rec["status"] == "rejected":
                cnt["reasons"][rec["reason"]] = cnt["reasons"].get(rec["reason"], 0) + 1
            if (cnt["ok"] + cnt["rejected"]) % 100 == 0:
                print(f"[stt] {cnt['ok'] + cnt['rejected']}/{len(todo)} "
                      f"({(time.time() - t0) / 60:.1f}min)", flush=True)
state_f.close()

ok = [json.loads(l) for l in STATE.read_text(encoding="utf-8").splitlines() if l.strip()]
ok = [o for o in ok if o["status"] == "ok"]
with TEXTS_CSV.open("w", encoding="utf-8") as f:
    for o in sorted(ok, key=lambda x: x["chunk"]):
        f.write(f"wavs/{o['chunk']}=={o['text']}\n")
print(f"[stt] OK {cnt['ok']} aceitos / {cnt['rejected']} rejeitados | motivos: {cnt['reasons']}",
      flush=True)
print(f"[stt] texts.csv: {len(ok)} linhas -> {TEXTS_CSV}", flush=True)
