"""lr_schedule.py — fator de LR (warmup + cosseno, com piso e restarts). Modulo puro."""
from __future__ import annotations

import math


def cosine_lr_factor(step: int, total: int, warmup: int, floor: float, cycles: int) -> float:
    """Multiplicador do LR no passo `step`.

    floor == 0: cosseno unico ate 0 ao fim de `total`.
    floor  > 0: `cycles` ciclos cosseno entre 1.0 e `floor` (restarts), cada um com
                (total - warmup) / cycles passos. (A versao anterior dividia por
                `cycle - warmup`, o que zerava a fase do ciclo antes do fim e criava um
                degrau de LR no final de cada ciclo.)
    """
    if step < warmup:
        return step / max(1, warmup)
    if floor > 0.0:
        n_cyc = max(1, int(cycles))
        cycle = max(1, (total - warmup) // n_cyc)
        pos = (step - warmup) % cycle
        prog = pos / cycle
        return floor + (1.0 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
