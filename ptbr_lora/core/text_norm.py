"""text_norm.py — normaliza texto para a FALA (pt-BR). Modulo puro (so `num2words`).

Fonte unica: vive em `ptbr_lora/core/` do repo de treino e e copiado para o repo da UI
(`scripts/sync_shared.py`). Mesma funcao no treino-eval (WER) e na inferencia.

O que faz (na ordem):
  0. protege URLs / e-mails (nao mexe);
  1. CPF / CNPJ / telefone / sequencias longas de digitos -> digito a digito;
  2. datas dd/mm/aaaa, horas (23h30, 14:05, 23h), moeda (R$ 1.234,56), porcentagem;
  3. ordinais (1º, 2ª), numerais romanos em contexto (século XX, Pedro II);
  4. numeros: milhar (1.234), decimal BR (1,5), zeros a esquerda, negativos;
  5. SIGLAS: so expande letra a letra o que e sigla de fato (dicionario ou sem vogal);
     CAIXA-ALTA "pronunciavel" (ATENCAO, NAO) e enfase -> minuscula (nao soletra).

Uso:
  from text_norm import normalize
  normalize("O CPF 123.456.789-00 é da TV às 23h30")
"""
from __future__ import annotations

import re

from num2words import num2words

_LETTERS = {
    "a": "a", "b": "bê", "c": "cê", "d": "dê", "e": "é", "f": "efe", "g": "gê",
    "h": "agá", "i": "i", "j": "jota", "k": "cá", "l": "ele", "m": "eme",
    "n": "ene", "o": "ó", "p": "pê", "q": "quê", "r": "erre", "s": "esse",
    "t": "tê", "u": "u", "v": "vê", "w": "dáblio", "x": "xis", "y": "ípsilon",
    "z": "zê",
}
_DIGITS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"]

# siglas -> leitura (letra a letra ou leitura propria)
_ACRONYMS = {
    "CPF": "cê pê efe", "CNPJ": "cê ene pê jota", "RG": "erre gê",
    "IBGE": "i bê gê é", "INSS": "i ene esse esse", "SUS": "esse u esse",
    "PIX": "pics", "TV": "tê vê", "CEP": "cê é pê", "UF": "u efe",
    "IPVA": "i pê vê a", "IPTU": "i pê tê u", "CNH": "cê ene agá",
    "PIB": "pê i bê", "ONU": "o ene u", "EUA": "é u a", "URSS": "u erre esse esse",
    "FGTS": "efe gê tê esse", "STF": "esse tê efe", "STJ": "esse tê jota",
    "TSE": "tê esse é", "OAB": "ó a bê", "MEC": "mec", "USP": "u esse pê",
    "UFRJ": "u efe erre jota", "CEO": "cê i ó", "PDF": "pê dê efe", "USB": "u esse bê",
    "SMS": "esse eme esse", "GPS": "gê pê esse", "DVD": "dê vê dê", "LED": "led",
    "ABNT": "a bê ene tê", "BNDES": "bê ene dê é esse", "CLT": "cê ele tê",
    "DDD": "dê dê dê", "CD": "cê dê", "DJ": "dê jota", "FBI": "efe bê i",
    "CIA": "cê i a", "UOL": "u ó ele", "HD": "agá dê", "PC": "pê cê", "BR": "bê erre",
    "OK": "oquei", "ONG": "ó ene gê", "IA": "i a", "TCC": "tê cê cê",
    "UTI": "u tê i", "PM": "pê eme", "SP": "esse pê", "RJ": "erre jota",
    "MG": "eme gê", "RS": "erre esse", "DF": "dê efe",
}

_UP = "A-ZÀ-ÖØ-Ý"
_LOW = "a-zà-öø-ÿ"
_VOWELS = set("AEIOUÀÁÂÃÉÊÍÓÔÕÚ")
_ACRO_RE = re.compile(rf"\b([{_UP}]{{2,8}})(s?)\b")

_PROTECT = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_CNPJ = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")
_PHONE = re.compile(r"\(\d{2}\)\s?9?\d{4}-\d{4}\b|\b9\d{4}-\d{4}\b")
_LONGDIG = re.compile(r"\b\d{9,}\b")
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_HOUR = re.compile(rf"\b(\d{{1,2}})\s?h(?:(\d{{2}}))?(?:\s?min)?(?![{_UP}{_LOW}\d])", re.IGNORECASE)
_CLOCK = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_MONEY = re.compile(
    r"R\$\s*(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?"
    r"(?:\s+(mil|milhão|milhões|bilhão|bilhões))?", re.IGNORECASE)
_PERCENT = re.compile(r"(-?\d{1,3}(?:\.\d{3})*(?:,\d+)?|-?\d+(?:,\d+)?)\s?%")
_ORD = re.compile(r"\b(\d+)\s?([ºª°])")
_ROMAN_BODY = r"(?=[IVXLC])(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_ROMAN_CTX = re.compile(
    rf"\b(século|séc\.|capítulo|cap\.|tomo|volume|parte|ato|fase|classe|"
    rf"Século|Capítulo|Tomo|Volume|Parte|Ato|Fase)\s+({_ROMAN_BODY})\b")
_ROMAN_NAME = re.compile(rf"\b([A-ZÀ-Ý][{_LOW}]{{2,}})\s+({_ROMAN_BODY})\b")
_NUM = re.compile(
    r"(?<![\w-])-(?=\d)|"                       # sinal negativo
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|"            # milhar (+ decimal)
    r"\d+(?:[,.]\d+)?"                          # inteiro / decimal
)


def _digits(s: str) -> str:
    return " ".join(_DIGITS[int(c)] for c in s if c.isdigit())


def _card(n: int) -> str:
    return num2words(int(n), lang="pt_BR").replace(",", "")


def _fem(words: str) -> str:
    """Feminiza o ultimo 'um'/'dois' (uma hora, vinte e duas horas)."""
    parts = words.split(" ")
    if parts and parts[-1] == "um":
        parts[-1] = "uma"
    elif parts and parts[-1] == "dois":
        parts[-1] = "duas"
    return " ".join(parts)


def _ordinal(n: int, fem: bool = False) -> str:
    try:
        w = num2words(int(n), to="ordinal", lang="pt_BR")
    except Exception:  # noqa: BLE001
        return _card(n)
    if fem:
        w = " ".join(p[:-1] + "a" if p.endswith("o") else p for p in w.split(" "))
    return w


def _decimal(int_part: str, frac: str) -> str:
    ip = _card(int(int_part or "0"))
    if frac.startswith("0"):
        z = len(frac) - len(frac.lstrip("0"))
        rest = frac.lstrip("0")
        fw = " ".join(["zero"] * z + ([_card(int(rest))] if rest and len(rest) <= 3 else
                                        [_digits(rest)] if rest else []))
    elif len(frac) <= 3:
        fw = _card(int(frac))
    else:
        fw = _digits(frac)
    return f"{ip} vírgula {fw}"


def _number_token(s: str) -> str:
    """Converte um token numerico ja isolado (sem sinal)."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", s):
        ip, _, fr = s.replace(".", "").partition(",")
        return _decimal(ip, fr) if fr else _card(int(ip))
    if "," in s:
        ip, fr = s.split(",", 1)
        return _decimal(ip, fr)
    if "." in s:                                  # 3.14 (estilo EN) -> decimal
        ip, fr = s.split(".", 1)
        return _decimal(ip, fr)
    if len(s) > 1 and s.startswith("0"):          # 007 -> zero zero sete
        return _digits(s)
    return _card(int(s))


def _num_sub(m: re.Match) -> str:
    s = m.group(0)
    if s == "-":
        return "menos "
    try:
        return _number_token(s)
    except Exception:  # noqa: BLE001
        return s


def _roman_to_int(r: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    tot = 0
    for i, ch in enumerate(r):
        v = vals[ch]
        tot += -v if i + 1 < len(r) and vals[r[i + 1]] > v else v
    return tot


def _acro_sub(m: re.Match) -> str:
    tok, plural = m.group(1), m.group(2)
    if tok in _ACRONYMS:
        return _ACRONYMS[tok] + ("s" if plural and " " in _ACRONYMS[tok] else "")
    if any(ch in "ÀÁÂÃÉÊÍÓÔÕÚÇ" for ch in tok) or (set(tok) & _VOWELS):
        return (tok + plural).lower()             # palavra em caixa alta / enfase
    spelled = " ".join(_LETTERS.get(c.lower(), c) for c in tok)
    return spelled + ("s" if plural else "")


def _money_sub(m: re.Match) -> str:
    ip = int(m.group(1).replace(".", ""))
    cents = m.group(2)
    scale = m.group(3)
    if scale:
        return f"{_card(ip)} {scale.lower()} de reais" if scale.lower() != "mil" else f"{_card(ip)} mil reais"
    out = f"{_card(ip)} {'real' if ip == 1 else 'reais'}"
    if cents:
        c = int(cents.ljust(2, "0"))
        if c:
            out += f" e {_card(c)} {'centavo' if c == 1 else 'centavos'}"
    return out


def _hm(h: int, mi: int | None) -> str:
    out = f"{_fem(_card(h))} {'hora' if h == 1 else 'horas'}"
    if mi:
        out += f" e {_card(mi)} {'minuto' if mi == 1 else 'minutos'}"
    return out


_MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
           "setembro", "outubro", "novembro", "dezembro"]


def _date_sub(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    year = int(y) if len(y) == 4 else (2000 + int(y) if int(y) < 50 else 1900 + int(y))
    day = "primeiro" if d == 1 else _card(d)
    return f"{day} de {_MONTHS[mo - 1]} de {_card(year)}"


def _normalize_plain(t: str) -> str:
    t = _CNPJ.sub(lambda m: _digits(m.group(0)), t)
    t = _CPF.sub(lambda m: _digits(m.group(0)), t)
    t = _PHONE.sub(lambda m: _digits(m.group(0)), t)
    t = _LONGDIG.sub(lambda m: _digits(m.group(0)), t)
    t = _DATE.sub(_date_sub, t)
    t = _HOUR.sub(lambda m: _hm(int(m.group(1)), int(m.group(2)) if m.group(2) else None), t)
    t = _CLOCK.sub(lambda m: _hm(int(m.group(1)), int(m.group(2))), t)
    t = _MONEY.sub(_money_sub, t)
    t = _PERCENT.sub(lambda m: _num_sub_signed(m.group(1)) + " por cento", t)
    t = _ORD.sub(lambda m: _ordinal(int(m.group(1)), fem=m.group(2) == "ª"), t)

    def roman_ctx(m: re.Match) -> str:
        n = _roman_to_int(m.group(2))
        return f"{m.group(1)} {_card(n)}" if 0 < n <= 30 else m.group(0)

    def roman_name(m: re.Match) -> str:
        n = _roman_to_int(m.group(2))
        if n <= 0 or n > 30:
            return m.group(0)
        return f"{m.group(1)} {_ordinal(n) if n <= 10 else _card(n)}"

    t = _ROMAN_CTX.sub(roman_ctx, t)
    t = _ROMAN_NAME.sub(roman_name, t)
    t = _NUM.sub(_num_sub, t)
    t = _ACRO_RE.sub(_acro_sub, t)
    t = re.sub(r"\s*&\s*", " e ", t)
    return t


def _num_sub_signed(s: str) -> str:
    neg = s.startswith("-")
    body = s.lstrip("-")
    try:
        w = _number_token(body)
    except Exception:  # noqa: BLE001
        return s
    return ("menos " if neg else "") + w


def normalize(text: str) -> str:
    """Normaliza `text` para leitura em voz alta (pt-BR). Idempotente para texto ja normalizado."""
    if not text:
        return text or ""
    parts = _PROTECT.split(text)
    # re.split com grupo: indices impares sao os trechos protegidos
    return "".join(p if i % 2 else _normalize_plain(p) for i, p in enumerate(parts))


if __name__ == "__main__":
    import sys

    print(normalize(" ".join(sys.argv[1:]) or "O CPF 123.456.789-00 é da TV às 23h30 por R$ 1,5 mil."))
