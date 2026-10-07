"""text_blocks.py — fatiamento de texto longo em blocos falaveis (<= ~10 s). Modulo puro.

Compartilhado entre `tools/gerar_em_blocos.py` (producao) e a UI (auto-chunk). O treino
usa clipes de ~6 s (max 10 s): gerar 20-30 s num unico passe e extrapolacao de comprimento.
"""
from __future__ import annotations

import re

WPS = 2.6            # palavras por segundo (~155 palavras/min), medido no corpus
MAX_BLOCK_S = 10.0   # teto de duracao do bloco (= teto do treino)

# abreviacoes que terminam em ponto mas NAO encerram a frase
_ABBREV = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "eng", "av", "r", "ex", "exa",
    "obs", "etc", "pág", "pag", "p", "fig", "cap", "vol", "nº", "no", "tel", "cia",
    "ltda", "min", "seg", "aprox", "dep", "sen", "gen", "cel", "cap", "ten", "sto", "sta",
    "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?;:…])\s+")


def _ends_with_abbrev(s: str) -> bool:
    m = re.search(r"(\w+)\.$", s)
    if not m:
        return False
    w = m.group(1).lower()
    return w in _ABBREV or (len(w) == 1 and w.isalpha())   # inicial: "J. Silva"


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    out: list[str] = []
    for s in raw:
        if out and _ends_with_abbrev(out[-1]):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def max_words_for(seconds: float = MAX_BLOCK_S, wps: float = WPS) -> int:
    return max(6, int(seconds * wps))


def split_blocks(text: str, max_words: int, min_words: int = 5) -> list[str]:
    """Quebra `text` em blocos de <= max_words palavras, cortando em pontuacao forte.

    Uma frase ate ~15 % acima do orcamento fica inteira; frases gigantes sao fatiadas por
    virgula e, se preciso, por palavras. Blocos < min_words sao grudados no vizinho.
    """
    sents = split_sentences(text)
    soft = max(max_words, int(round(max_words * 1.15)))
    blocks: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > soft:
            blocks.append(" ".join(cur))
            cur, n = [], 0
        if w > soft:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            chunks: list[str] = []
            for pi, p in enumerate(parts):
                suffix = "," if pi < len(parts) - 1 else ""
                pw = p.split()
                if len(pw) <= max_words:
                    chunks.append(p + suffix)
                    continue
                for i in range(0, len(pw), max_words):
                    is_last = i + max_words >= len(pw)
                    chunks.append(" ".join(pw[i:i + max_words]) + (suffix if is_last else ""))
            for ch in chunks:
                cw = len(ch.split())
                if cur and n + cw > soft:
                    blocks.append(" ".join(cur))
                    cur, n = [], 0
                cur.append(ch)
                n += cw
            continue
        cur.append(s)
        n += w
    if cur:
        blocks.append(" ".join(cur))

    out: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if len(b.split()) < min_words and i + 1 < len(blocks):
            blocks[i + 1] = b + " " + blocks[i + 1]
            i += 1
            continue
        out.append(b)
        i += 1
    if len(out) > 1 and len(out[-1].split()) < min_words:
        out[-2] = out[-2] + " " + out[-1]
        out.pop()
    return out


def blocks_for_seconds(text: str, max_s: float = MAX_BLOCK_S, wps: float = WPS) -> list[str]:
    return split_blocks(text, max_words_for(max_s, wps))


def expected_duration(words: int, wps: float = WPS, floor: float = 2.0) -> float:
    return max(floor, words / max(wps, 0.1))


def dur_ok(dur: float, words: int, lo: float = 0.55, hi: float = 2.2, wps: float = WPS) -> bool:
    """A duracao gerada e plausivel para o texto? Ancora no TEXTO (pega o caso de todos os
    candidatos arrastados, que o gate relativo entre vizinhos nao pega)."""
    exp_s = expected_duration(words, wps)
    return lo * exp_s <= dur <= hi * exp_s
