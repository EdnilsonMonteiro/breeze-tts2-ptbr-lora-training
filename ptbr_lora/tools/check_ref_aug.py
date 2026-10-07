"""check_ref_aug.py — confere que a augmentation troca SO a referencia (nunca o alvo).

Para alguns clipes em tokens_aug/ monta o MESMO item duas vezes (ref original x ref aumentada) e exige:
  * os frames do ALVO e as posicoes supervisionadas dos labels sao identicos;
  * so o numero de frames da REFERENCIA pode mudar (e a diferenca = diferenca de comprimento do prompt);
  * _ref_aug = True so no item aumentado.
Uso (apos augment_refs.py):  python ptbr_lora\\tools\\check_ref_aug.py [--n 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(_CORE.parents[1]))

import common_breeze as CB  # noqa: E402
import prepare_dataset as PD  # noqa: E402
import refs as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    aug_dir = CB.TRAINING / "tokens_aug"
    files = sorted(p for p in aug_dir.glob("*__a0.npz"))[: args.n * 4]
    assert files, f"sem tokens em {aug_dir} (rode augment_refs.py)"
    recs = {r["idx"]: r for r in PD.load_meta()}
    train = PD.load_split_file("train")
    pools = R.build_pools(list(recs.values()), train)
    tok = CB.load_text_tokenizer()
    ok = 0
    for p in files:
        ref_idx = p.name.split("__a")[0]
        spk = recs[ref_idx].get("speaker")
        tgt = next((recs[i] for i in pools.get(spk, []) if i != ref_idx), None)
        if tgt is None:
            continue
        a = PD.build_item(tok, tgt, "ref_edit_tata", ref_rec=recs[ref_idx], ref_codes_path=p)
        b = PD.build_item(tok, tgt, "ref_edit_tata", ref_rec=recs[ref_idx])
        assert a["_n_frames"] == b["_n_frames"], "ALVO mudou!"
        assert a["_ref_aug"] is True and b["_ref_aug"] is False
        la, lb = a["labels"][0], b["labels"][0]
        sa, sb = int((la != CB.IGNORE_IDX).sum()), int((lb != CB.IGNORE_IDX).sum())
        assert sa == sb, f"posicoes supervisionadas diferem: {sa} x {sb}"
        d_len = len(a["input_ids"][0]) - len(b["input_ids"][0])
        assert d_len == a["_n_ref_frames"] - b["_n_ref_frames"], (d_len, a["_n_ref_frames"], b["_n_ref_frames"])
        ia, ib = a["input_values"].reshape(-1, 16), b["input_values"].reshape(-1, 16)
        nt = a["_n_frames"]
        assert ia.shape[0] == nt + a["_n_ref_frames"] and ib.shape[0] == nt + b["_n_ref_frames"]
        assert (ia[-nt:] == ib[-nt:]).all(), "codes do ALVO diferem"
        assert ia.shape != ib.shape or not (ia[: a["_n_ref_frames"]] == ib[: b["_n_ref_frames"]]).all(), \
            "referencia aumentada identica a original (aug nao aplicada?)"
        print(f"OK {tgt['idx']} <- ref {ref_idx}: frames alvo={a['_n_frames']} ref {b['_n_ref_frames']}->"
              f"{a['_n_ref_frames']} supervisionados={sa}")
        ok += 1
        if ok >= args.n:
            break
    assert ok, "nenhum par verificado"
    print(f"[check_ref_aug] {ok} pares OK — augmentation afeta so a referencia")


if __name__ == "__main__":
    main()
