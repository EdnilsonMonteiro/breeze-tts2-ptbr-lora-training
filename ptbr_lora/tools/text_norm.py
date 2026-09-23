"""text_norm.py — normaliza numeros para PALAVRAS (pt-BR) antes da geracao.

Motivo: o modelo fala melhor com numeros por extenso ("vinte e três") do que com
digitos crus ("23"), e o dataset de treino foi construido assim (num2words).
Aplique em QUALQUER texto que vai para a inferencia.

Uso:
  from text_norm import normalize
  texto = normalize("A temperatura é de 23 graus e 1,5 litro.")
"""
from __future__ import annotations

import re

from num2words import num2words

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _one(m: re.Match) -> str:
    s = m.group(0)
    try:
        if "," in s:  # decimal BR (1,5)
            return num2words(float(s.replace(",", ".")), lang="pt-BR")
        if "." in s:
            parts = s.split(".")
            if len(parts) >= 2 and len(parts[-1]) == 3:  # 1.234 -> milhar
                return num2words(int(s.replace(".", "")), lang="pt-BR")
            return num2words(float(s), lang="pt-BR")  # 1.5 -> decimal
        return num2words(int(s), lang="pt-BR")
    except Exception:  # noqa: BLE001
        return s


def normalize(text: str) -> str:
    """Converte todos os numeros do texto para palavras (pt-BR)."""
    return _NUM.sub(_one, text or "")


if __name__ == "__main__":
    import sys

    print(normalize(" ".join(sys.argv[1:]) or "23 graus e 1,5 litro em 2026."))
