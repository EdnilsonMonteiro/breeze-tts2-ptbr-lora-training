"""text_norm.py — normaliza texto para a FALA (pt-BR) antes da geracao.

Dois problemas conhecidos e tratados aqui:
  1) NUMEROS: o modelo fala melhor por extenso ("vinte e três") que com digitos ("23").
  2) SIGLAS/CAIXA-ALTA: o base (EN/ZH) le siglas com fonologia inglesa; expandimos
     para os NOMES DAS LETRAS em portugues ("CPF" -> "cê pê efe").

Uso:
  from text_norm import normalize
  normalize("O CPF 123.456 é da TV às 23h")
"""
from __future__ import annotations

import re

from num2words import num2words

_NUM = re.compile(r"\d+(?:[.,]\d+)?")

_LETTERS = {
    "a": "a", "b": "bê", "c": "cê", "d": "dê", "e": "e", "f": "efe", "g": "gê",
    "h": "agá", "i": "i", "j": "jota", "k": "cá", "l": "ele", "m": "eme",
    "n": "ene", "o": "o", "p": "pê", "q": "quê", "r": "erre", "s": "esse",
    "t": "tê", "u": "u", "v": "vê", "w": "dáblio", "x": "xis", "y": "ípsilon",
    "z": "zê",
}

# siglas frequentes -> leitura por extenso (letra a letra)
_ACRONYMS = {
    "CPF": "cê pê efe", "CNPJ": "cê ene pê jota", "RG": "erre gê",
    "IBGE": "i bê gê e", "INSS": "i ene esse esse", "SUS": "esse u esse",
    "PIX": "pê i xis", "TV": "tê vê", "CEP": "cê e pê", "UF": "u efe",
    "IPVA": "i pê vê a", "IPTU": "i pê tê u", "CNH": "cê ene agá",
    "PIB": "pê i bê", "ONU": "o ene u", "EUA": "e u a", "URSS": "u erre esse esse",
}

_ACRO_RE = re.compile(r"\b[A-Z]{2,6}\b")
_HOUR = re.compile(r"\b(\d+)\s*h\b", re.IGNORECASE)


def _spell(token: str) -> str:
    return " ".join(_LETTERS.get(ch.lower(), ch) for ch in token)


def _num(m: re.Match) -> str:
    s = m.group(0)
    try:
        if "," in s:  # decimal BR (1,5)
            return num2words(float(s.replace(",", ".")), lang="pt-BR")
        if "." in s:
            parts = s.split(".")
            if len(parts) >= 2 and len(parts[-1]) == 3:  # 1.234 -> milhar
                return num2words(int(s.replace(".", "")), lang="pt-BR")
            return num2words(float(s), lang="pt-BR")
        return num2words(int(s), lang="pt-BR")
    except Exception:  # noqa: BLE001
        return s


def normalize(text: str) -> str:
    text = _ACRO_RE.sub(lambda m: _ACRONYMS.get(m.group(0), _spell(m.group(0))), text or "")
    text = _HOUR.sub(lambda m: num2words(int(m.group(1)), lang="pt-BR") + " horas", text)
    return _NUM.sub(_num, text)


if __name__ == "__main__":
    import sys

    print(normalize(" ".join(sys.argv[1:]) or "O CPF 123 é da TV às 23h por 1,5 mil."))
