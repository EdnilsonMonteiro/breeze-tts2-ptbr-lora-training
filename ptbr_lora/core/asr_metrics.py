"""asr_metrics.py — WER/CER, deteccao de falhas catastroficas e estatistica. Modulo puro.

Substitui as 3 copias divergentes de `wer_cer` (eval_wer, tools/metrics, gerar_em_blocos).
Decisoes:
  * normalizacao ANTES de minusculizar e nos DOIS lados (ref e hyp): `text_norm` precisa
    ver maiusculas para reconhecer siglas; sem isso "1512" vs "quinze doze" vira erro falso;
  * `lang="en"` nao aplica text_norm pt-BR (o texto de regressao em ingles);
  * `wer_detail` devolve S/D/I: WER alto por insercao (Whisper alucinando cauda) e falha
    de geracao diferente de WER alto por substituicao (pronuncia ruim);
  * `catastrophic` marca amostras quebradas (repeticao, duracao absurda, WER > 0,5) — a
    media de WER sozinha esconde uma cauda de falhas.
"""
from __future__ import annotations

import random
import re
import statistics
import unicodedata

try:                                            # modulo irmao (core/)
    from text_norm import normalize as _tn
except Exception:  # noqa: BLE001  (num2words ausente etc.)
    def _tn(s: str) -> str:
        return s

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_wer(s: str, lang: str = "pt", normalize: bool = True) -> str:
    s = s or ""
    if normalize and lang.lower().startswith("pt"):
        s = _tn(s)
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("-", " ").replace("_", " ")
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_ops(a: list, b: list) -> tuple[int, int, int]:
    """(substituicoes, delecoes, insercoes) minimas de a(ref) -> b(hyp)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            s += int(a[i - 1] != b[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dl, ins


def _lev(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer_detail(ref: str, hyp: str, lang: str = "pt", normalize: bool = True) -> dict:
    r = normalize_for_wer(ref, lang, normalize)
    h = normalize_for_wer(hyp, lang, normalize)
    rw, hw = r.split(), h.split()
    s, dl, ins = edit_ops(rw, hw)
    rc, hc = list(r.replace(" ", "")), list(h.replace(" ", ""))
    return {
        "wer": (s + dl + ins) / max(1, len(rw)),
        "cer": _lev(rc, hc) / max(1, len(rc)),
        "S": s, "D": dl, "I": ins, "n_ref": len(rw), "n_hyp": len(hw),
    }


def wer_cer(ref: str, hyp: str, lang: str = "pt", normalize: bool = True) -> tuple[float, float]:
    d = wer_detail(ref, hyp, lang, normalize)
    return d["wer"], d["cer"]


def repetition_ratio(text: str, n: int = 3) -> float:
    """Fracao de n-gramas de palavras que sao repeticao de um ja visto (0 = sem loop)."""
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def catastrophic(wer: float, dur_s: float | None, n_words: int, hyp: str = "",
                 wps: float = 2.6, wer_max: float = 0.5, dur_lo: float = 0.4,
                 dur_hi: float = 2.5, rep_max: float = 0.3) -> list[str]:
    """Motivos pelos quais a amostra e considerada quebrada ([] = ok)."""
    why: list[str] = []
    if wer > wer_max:
        why.append("wer")
    if dur_s is not None and n_words > 0:
        exp = max(2.0, n_words / wps)
        if dur_s < dur_lo * exp:
            why.append("curto")
        elif dur_s > dur_hi * exp:
            why.append("longo")
    if hyp and repetition_ratio(normalize_for_wer(hyp, "pt", False)) > rep_max:
        why.append("repeticao")
    return why


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(media, lo, hi) por bootstrap percentil. Vazio -> (nan, nan, nan)."""
    v = [float(x) for x in values if x == x]
    if not v:
        nan = float("nan")
        return nan, nan, nan
    rng = random.Random(seed)
    n = len(v)
    means = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(v) / n, lo, hi


def summarize(values: list[float], fails: list[bool] | None = None) -> dict:
    v = [float(x) for x in values if x == x]
    if not v:
        return {"n": 0}
    mean, lo, hi = bootstrap_ci(v)
    sv = sorted(v)
    out = {
        "n": len(v), "mean": mean, "ci95": [lo, hi], "median": statistics.median(v),
        "p90": sv[min(len(sv) - 1, int(0.9 * len(sv)))],
        "sd": statistics.pstdev(v) if len(v) > 1 else 0.0,
    }
    if fails is not None:
        out["fail_rate"] = sum(1 for f in fails if f) / max(1, len(fails))
    return out


def micro_wer(details: list[dict]) -> float:
    """WER agregado (soma de erros / soma de palavras) — pondera por comprimento."""
    err = sum(d["S"] + d["D"] + d["I"] for d in details)
    return err / max(1, sum(d["n_ref"] for d in details))
