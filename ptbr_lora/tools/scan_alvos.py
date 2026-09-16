import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import paths  # noqa: E402
NUM = re.compile(
    r"\b(zero|um|dois|tres|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|"
    r"catorze|quatorze|quinze|dezesseis|dezessete|dezoito|dezenove|vinte|trinta|"
    r"quarenta|cinquenta|sessenta|setenta|oitenta|noventa|cem|cento|duzentos|"
    r"trezentos|quatrocentos|quinhentos|seiscentos|setecentos|oitocentos|"
    r"novecentos|mil(?:h(?:ao|ão|oes|ões))?|bilh(?:ao|ão|oes|ões))\b", re.I)
SIGLA = re.compile(r"\b[A-Z]{2,7}\b")
LETRA = re.compile(r"\b[a-zA-Z]\b")

def stats(texts, durs=None):
    n = len(texts)
    h = sum(durs) / 3600 if durs else None
    out = {"n": n}
    if h:
        out["h"] = round(h, 2)
    for name, rx in [("numeros", NUM), ("siglas", SIGLA), ("letras", LETRA)]:
        idxs = [i for i, t in enumerate(texts) if rx.search(t)]
        out[name] = len(idxs)
        if durs:
            out[name + "_h"] = round(sum(durs[i] for i in idxs) / 3600, 2)
    return out

# podcast
pod = [json.loads(l) for l in (paths.TRAINING / "dataset_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
pod = [r for r in pod if "_" in r["idx"]]
print("PODCAST:", stats([r["text"] for r in pod], [r["dur_proc_s"] for r in pod]))

# tata
tata = [l.split("==", 1)[1] for l in (paths.DATASETS_ROOT / "TTS-Portuguese-Corpus" / "texts.csv").read_text(encoding="utf-8").splitlines() if "==" in l]
print("TATA   :", stats(tata))

# exemplos podCAST de cada categoria
for name, rx in [("numero", NUM), ("sigla", SIGLA), ("letra", LETRA)]:
    ex = [r["text"][:90] for r in pod if rx.search(r["text"])][:3]
    print(f"\n{name} exemplos:")
    for e in ex:
        print("  -", e)
