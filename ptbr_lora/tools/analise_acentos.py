"""Analise de acentos: (1) NFC/NFD no corpus, (2) fragmentacao do tokenizer."""

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import common_breeze as CB

# ---------- 1) Corpus: tem NFD / combining marks?
recs = [json.loads(l) for l in open(CB.TRAINING / "dataset_meta.jsonl", encoding="utf-8")]
texts = [r["text"] for r in recs]
n_nfd = 0
nfc_diffs = []
marks = Counter()
for t in texts:
    if any(0x0300 <= ord(c) <= 0x036F for c in t):
        n_nfd += 1
        marks.update(c for c in t if 0x0300 <= ord(c) <= 0x036F)
    if unicodedata.normalize("NFC", t) != t:
        nfc_diffs.append(t)

print("=== 1) UNICODE NO CORPUS (3512 textos) ===")
print(f"textos com combining marks (NFD): {n_nfd}")
print(f"textos que mudariam com NFC    : {len(nfc_diffs)}")
if marks:
    print("marques encontrados:", {unicodedata.name(k, hex(ord(k))): v for k, v in marks.most_common(5)})
if nfc_diffs[:3]:
    for d in nfc_diffs[:3]:
        print("  exemplo:", d[:90])

# contagem geral de letras acentuadas pre-compostas (NFC legitimo)
acc = Counter()
for t in texts:
    for c in t:
        if c in "áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ":
            acc[c] += 1
print("acentos NFC no corpus (top):", dict(acc.most_common(12)))

# ---------- 2) Tokenizer: fatiamento de palavras acentuadas
tok = CB.load_text_tokenizer()
palavras = [
    ("café", "cafe"), ("vovô", "vovo"), ("saudade", "saudade"),
    ("quentinho", "quentinho"), ("coração", "coracao"), ("ônibus", "onibus"),
    ("à tarde", "a tarde"), ("após", "apos"), ("vinte e três", "vinte e tres"),
    ("previsão", "previsao"), ("número", "numero"), ("olá", "ola"),
]
print("\n=== 2) TOKENIZACAO (palavra -> ids) ===")
total_acc, total_plain = 0, 0
for w_acc, w_plain in palavras:
    ia = tok(w_acc, add_special_tokens=False)["input_ids"]
    ip = tok(w_plain, add_special_tokens=False)["input_ids"]
    toks_a = tok.convert_ids_to_tokens(ia)
    total_acc += len(ia); total_plain += len(ip)
    print(f"{w_acc:<14} {len(ia)} toks {toks_a}   | sem acento {w_plain:<14} {len(ip)}")
ratio = total_acc / total_plain
print(f"\nratio medio tokens acentado/plaino: {ratio:.2f}x")

# versao NFD artificial
w = unicodedata.normalize("NFD", "café coração vovô")
iw = tok(w, add_special_tokens=False)["input_ids"]
print(f"NFD 'café coração vovô': {len(iw)} toks {tok.convert_ids_to_tokens(iw)}")
w_nfc = "café coração vovô"
iw2 = tok(w_nfc, add_special_tokens=False)["input_ids"]
print(f"NFC idem             : {len(iw2)} toks {tok.convert_ids_to_tokens(iw2)}")

# ---------- 3) friquencia das formas nas transcrições mantidas p/ treino
alvo_words = ["café", "vovô", "saudade", "quentinho"]
cnt = Counter()
for t in texts:
    tl = t.lower()
    for w0 in alvo_words:
        if w0 in tl:
            cnt[w0] += 1
print("\n=== 3) FREQUENCIA DAS PALAVRAS-CRITICAS NO TREINO ===")
for w0 in alvo_words:
    print(f"  {w0}: {cnt[w0]} ocorrencias")

# vocab do tokenizer reconhece palavras inteiras?
print("\n=== 4) PALAVRA INTEIRA NO VOCAB? ===")
for w0 in [" café", " vovô", " saudade", " quentinho"]:
    tid = tok.convert_tokens_to_ids(w0)
    print(f"  {w0!r}: id_unk={tid == tok.unk_token_id if tok.unk_token_id else 'sem unk'} ({tid})")
