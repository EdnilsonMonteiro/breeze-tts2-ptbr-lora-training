"""Fase 6 - relatorio QC + sorteio de 30 pares (wav,texto) p/ audicao."""
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

ROOT = paths.ARTIFACTS
WORK = ROOT / "dataScrapping" / "work"
WAVS = ROOT / "datasets" / "podcast" / "wavs"
AUDIT = ROOT / "dataScrapping" / "pilot_audit"
if AUDIT.exists():
    shutil.rmtree(AUDIT)
AUDIT.mkdir(parents=True)

index = {r["chunk"]: r for r in (json.loads(l) for l in
         (WORK / "chunks_index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
state = [json.loads(l) for l in (WORK / "transcribe_state.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ok = [s for s in state if s["status"] == "ok"]
rej = Counter(s["reason"].split(" ")[0] for s in state if s["status"] == "rejected")

durs = [index[s["chunk"]]["dur"] for s in ok]
spx = [index[s["chunk"]]["speaker"] for s in ok]
tags = [index[s["chunk"]]["tag"] for s in ok]
cps = [len(s["text"]) / index[s["chunk"]]["dur"] for s in ok]

print("=== RELATORIO QC PILOTO ===")
print(f"aceitos: {len(ok)}/{len(state)} ({100*len(ok)/len(state):.1f}%) | rejeicoes: {dict(rej)}")
print(f"duracao: min={min(durs):.1f}s med={sorted(durs)[len(durs)//2]:.1f}s max={max(durs):.1f}s | "
      f"total={sum(durs)/3600:.2f}h")
print(f"chars/s: p5={sorted(cps)[len(cps)//20]:.1f} med={sorted(cps)[len(cps)//2]:.1f} "
      f"p95={sorted(cps)[-len(cps)//20]:.1f}")
print("por episodio:", {t: sum(1 for x in tags if x == t) for t in set(tags)})
print("por locutor :", {s: sum(1 for x in spx if x == s) for s in sorted(set(spx))})
h_speaker = {}
for s in ok:
    h_speaker[index[s["chunk"]]["speaker"]] = h_speaker.get(index[s["chunk"]]["speaker"], 0) + \
        index[s["chunk"]]["dur"] / 3600
print("horas por locutor:", {k: round(v, 2) for k, v in sorted(h_speaker.items(), key=lambda x: -x[1])})

random.seed(42)
sample = random.sample(ok, 30)
pl = []
for i, s in enumerate(sample):
    r = index[s["chunk"]]
    dst = AUDIT / s["chunk"]
    shutil.copy(WAVS / s["chunk"], dst)
    pl.append(f"{s['chunk']} | {r['tag']} | {r['speaker']} | {r['dur']:.1f}s\n  {s['text']}\n")
(AUDIT / "playlist.txt").write_text("".join(pl), encoding="utf-8")
print(f"\naudicao: 30 pares -> {AUDIT}\\playlist.txt")
