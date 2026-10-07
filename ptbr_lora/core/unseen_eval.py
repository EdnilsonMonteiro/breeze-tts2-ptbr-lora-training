"""unseen_eval.py — SECS + WER em VOZES NUNCA VISTAS, a cada N passos do treino.

Motivo: val_loss nao mede clonagem (diagnostico r73_01: a loss melhorava enquanto o SECS de vozes
novas caia 0,14-0,22). Aqui o plano e fixo e deterministico: N vozes do split `val` (grupos que o
treino nunca viu; um por corpus) + opcionalmente uma voz extra do usuario, mesmas frases e mesma seed
em todos os checkpoints -> curvas comparaveis. SECS (ECAPA, CPU) contra clipes held-out da MESMA voz,
com teto (prompt x held-out real) e piso (voz x outras vozes); WER (faster-whisper, CPU).

A geracao em si e feita por `train_lora.generate_samples(jobs=...)` (mesmo caminho validado das
amostras); este modulo so planeja (`build_plan`, `jobs_for`) e pontua (`score`). Sem torch.
"""
from __future__ import annotations

import csv
import html
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import voices as V

PHRASES = [
    ("hoje-tempo", "Hoje o tempo está bom, então vou sair para caminhar com calma pelo parque."),
    # palavras que o r74 errava (pronuncia): temperatura, previsao, umidade...
    ("temperatura", "A temperatura máxima prevista para amanhã é de vinte e oito graus, com umidade baixa."),
    ("compras", "Preciso comprar pão, leite e algumas frutas antes que o mercado feche."),
    ("reuniao", "A reunião de amanhã foi adiada para quinta-feira, às três horas da tarde."),
    ("cafe", "Quando chegar em casa, vou preparar um café bem forte e ler um livro."),
]

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def build_plan(recs: dict[str, dict], split_idxs: list[str], wavs_dir: Path, n_voices: int = 4,
               extra: dict | list | None = None) -> dict:
    """Plano fixo: {voices:[{name, corpus, group, prompt:{audio,text}, heldout:[wav...]}]}.

    extra = {"name","audio","text","heldout":[wav,...]} OU uma lista deles (as vozes do usuario; fora do corpus).
    `heldout` vazio => usa o proprio audio de referencia (o SECS e contra a referencia, como no benchmark;
    `secs_norm` perde o sentido nessas vozes). group = "val" (vozes nunca vistas do split) | "user".
    """
    if isinstance(extra, dict):
        extra = [extra]
    voices = []
    for e in V.pick_eval_voices(recs, split_idxs, n=n_voices):
        pr = recs[e["prompt_idx"]]
        voices.append({
            "name": e["name"], "corpus": e["corpus"], "speaker": e["speaker"],
            "prompt": {"idx": e["prompt_idx"], "speaker": e["speaker"],
                       "audio": str(Path(wavs_dir) / f"{e['prompt_idx']}.wav"), "text": pr["text"]},
            "heldout": [str(Path(wavs_dir) / f"{i}.wav") for i in e["heldout_idx"]],
            "group": "val",
        })
    for k, ex in enumerate(extra or []):
        nm = ex.get("name", "extra" if len(extra) == 1 else f"extra{k}")
        ho = [x for x in (ex.get("heldout") or []) if str(x).strip()] or [ex["audio"]]
        voices.append({"name": nm, "corpus": "extra", "speaker": nm, "group": "user",
                       "self_heldout": not ex.get("heldout"),
                       "prompt": {"idx": nm, "speaker": nm, "audio": ex["audio"], "text": ex["text"]},
                       "heldout": list(ho)})
    names = [v["name"] for v in voices]
    if len(set(names)) != len(names):
        raise ValueError(f"nomes de voz repetidos no plano: {names}")
    return {"voices": voices}


def jobs_for(plan: dict, n_phrases: int = 2, val_phrase_ids: list[int] | None = None
             ) -> list[tuple[str, str, dict]]:
    """[(nome_do_arquivo, texto, ref)] — voz x frase; usado por generate_samples(jobs=...).

    `val_phrase_ids` (ex.: [1]) limita as frases das vozes do split val (menos geracoes por avaliacao);
    as vozes do usuario sempre geram as `n_phrases` frases.
    """
    jobs = []
    for v in plan["voices"]:
        for k, (slug_, text) in enumerate(PHRASES[:n_phrases]):
            if v.get("group") == "val" and val_phrase_ids is not None and k not in val_phrase_ids:
                continue
            jobs.append((f"{v['name']}__p{k}", text, v["prompt"]))
    return jobs


def _emb_cache(plan: dict) -> dict:
    return plan.setdefault("_emb", {})


def _embed(path: str, plan: dict, metrics):
    import librosa

    cache = _emb_cache(plan)
    if path in cache:
        return cache[path]
    y, _ = librosa.load(path, sr=16000, mono=True)
    y = np.asarray(y, dtype=np.float32)
    r = float(np.sqrt(np.mean(y ** 2)))
    if r > 1e-6:
        y = y * (0.05 / r)                           # iguala o RMS (o ECAPA e sensivel ao nivel)
    e = metrics.embed_array(y, "cpu")
    cache[path] = e
    return e


def score(out_dir: Path, plan: dict, n_phrases: int = 2, asr_size: str = "small") -> dict | None:
    """SECS/WER dos wavs `<voz>__p<k>.wav` em out_dir. None se as dependencias faltarem."""
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    try:
        import metrics
    except Exception as exc:  # noqa: BLE001
        print(f"[unseen] metrics indisponivel: {type(exc).__name__}: {str(exc)[:100]}")
        return None
    out_dir = Path(out_dir)
    # teto/piso por voz (calculados uma vez; embeddings em cache no plano)
    for v in plan["voices"]:
        if "ceiling" in v:
            continue
        pe = _embed(v["prompt"]["audio"], plan, metrics)
        hos = [_embed(p, plan, metrics) for p in v["heldout"]]
        others = [o for o in plan["voices"] if o is not v]
        fl = [_embed(p, plan, metrics) for o in others for p in o["heldout"][:2]]
        v["ceiling"] = float(np.mean([metrics.cos(pe, h) for h in hos])) if hos else float("nan")
        v["floor"] = float(np.mean([metrics.cos(h, f) for h in hos for f in fl])) if hos and fl \
            else float("nan")
    items = []
    for v in plan["voices"]:
        hos = [_embed(p, plan, metrics) for p in v["heldout"]]
        pe = _embed(v["prompt"]["audio"], plan, metrics)
        for k in range(n_phrases):
            f = out_dir / f"{v['name']}__p{k}.wav"
            if not f.is_file():
                continue
            ge = _embed(str(f), plan, metrics)
            sh = float(np.mean([metrics.cos(ge, h) for h in hos]))
            try:
                hyp = metrics.transcribe(f, asr_size, "cpu", "int8")
                wer, cer = metrics.wer_cer(PHRASES[k][1], hyp)
            except Exception as exc:  # noqa: BLE001
                hyp, wer, cer = "", float("nan"), float("nan")
                print(f"[unseen] ASR falhou: {type(exc).__name__}: {str(exc)[:100]}")
            items.append({"voice": v["name"], "corpus": v["corpus"], "phrase": k,
                          "secs_heldout": sh, "secs_prompt": metrics.cos(ge, pe),
                          "secs_norm": V.normalized_secs(sh, v["ceiling"], v["floor"]),
                          "wer": wer, "cer": cer, "hyp": hyp})
    if not items:
        return None

    def m(key, rows):
        x = [r[key] for r in rows if r[key] == r[key]]
        return float(np.mean(x)) if x else float("nan")

    per_voice = {}
    for v in plan["voices"]:
        rows = [r for r in items if r["voice"] == v["name"]]
        if rows:
            per_voice[v["name"]] = {"corpus": v["corpus"], "ceiling": v["ceiling"], "floor": v["floor"],
                                    "secs_heldout": m("secs_heldout", rows), "secs_norm": m("secs_norm", rows),
                                    "wer": m("wer", rows)}
    grp = {v["name"]: v.get("group", "val") for v in plan["voices"]}
    for it in items:
        it["group"] = grp.get(it["voice"], "val")
    rv = [r for r in items if r["group"] == "val"]
    ru = [r for r in items if r["group"] == "user"]
    res = {"secs_heldout": m("secs_heldout", items), "secs_prompt": m("secs_prompt", items),
           "secs_norm": m("secs_norm", items), "wer": m("wer", items), "cer": m("cer", items),
           "secs_val": m("secs_heldout", rv), "wer_val": m("wer", rv),
           "secs_user": m("secs_prompt", ru), "wer_user": m("wer", ru),
           "n": len(items), "per_voice": per_voice, "items": items}
    (out_dir / "unseen_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    return res


# secs_val/wer_val = vozes nunca vistas do split (SECS contra clipes held-out reais); secs_user/wer_user = as
# vozes de referencia do usuario (SECS contra a propria referencia, comparavel ao benchmark de adapters).
CSV_COLS = ["step", "secs_heldout", "secs_prompt", "secs_norm", "wer", "cer", "n",
            "secs_val", "wer_val", "secs_user", "wer_user"]


def append_csv(path: Path, step: int, res: dict) -> None:
    new = not Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_COLS)
        w.writerow([step] + [f"{res[c]:.4f}" if isinstance(res[c], float) else res[c]
                             for c in CSV_COLS[1:]])


def _step_of(name: str) -> int:
    m = re.fullmatch(r"step(\d+)", name)
    return int(m.group(1)) if m else -1


def write_listen_page(run_dir: Path, plan: dict, n_phrases: int = 2) -> Path | None:
    """`<run>/ouvir/index.html`: para cada voz x frase, a REFERENCIA e o audio gerado em cada
    checkpoint lado a lado (com SECS/WER), para ouvir se a clonagem melhora com o treino.

    Os wavs gerados ficam em `eval_unseen/step<N>/` (caminho relativo, sem copiar); so as
    referencias sao copiadas para `ouvir/refs/`. Abra o index.html direto do disco.
    """
    run_dir = Path(run_dir)
    base = run_dir / "eval_unseen"
    steps = sorted((d for d in base.glob("step*") if d.is_dir() and _step_of(d.name) >= 0),
                   key=lambda d: _step_of(d.name))
    if not steps:
        return None
    out = run_dir / "ouvir"
    (out / "refs").mkdir(parents=True, exist_ok=True)
    scores: dict[int, dict] = {}
    for d in steps:
        f = d / "unseen_scores.json"
        if f.is_file():
            try:
                scores[_step_of(d.name)] = {(r["voice"], r["phrase"]): r
                                            for r in json.loads(f.read_text(encoding="utf-8"))["items"]}
            except Exception:  # noqa: BLE001
                pass
    ref_rel: dict[str, str] = {}
    for v in plan["voices"]:
        src = Path(v["prompt"]["audio"])
        if src.is_file():
            dst = out / "refs" / f"{v['name']}.wav"
            if not dst.exists():
                shutil.copy2(src, dst)
            ref_rel[v["name"]] = f"refs/{v['name']}.wav"
    head = "".join(f"<th>passo {_step_of(d.name)}{' (base, sem treino)' if _step_of(d.name) == 0 else ''}</th>"
                   for d in steps)
    rows = []
    for v in plan["voices"]:
        for k in range(n_phrases):
            text = html.escape(PHRASES[k][1])
            r_audio = (f'<audio controls preload="none" src="{ref_rel[v["name"]]}"></audio>'
                       if v["name"] in ref_rel else "-")
            cells = [f"<td><b>{html.escape(v['name'])}</b> <small>({html.escape(v['corpus'])})</small>"
                     f"<br><small>{text}</small></td><td>{r_audio}</td>"]
            for d in steps:
                n = _step_of(d.name)
                f = d / f"{v['name']}__p{k}.wav"
                if not f.is_file():
                    cells.append("<td>-</td>")
                    continue
                sc = scores.get(n, {}).get((v["name"], k))
                info = (f"<br><small>SECS {sc['secs_heldout']:.2f} | WER {sc['wer']:.2f}</small>"
                        if sc and sc["secs_heldout"] == sc["secs_heldout"] else "")
                cells.append(f'<td><audio controls preload="none" '
                             f'src="../eval_unseen/{d.name}/{f.name}"></audio>{info}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
    page = ("<!doctype html><meta charset=utf-8><title>Ouvir clonagem</title>"
            "<style>body{font-family:sans-serif;margin:16px}td,th{border:1px solid #ccc;padding:6px;"
            "vertical-align:top}audio{width:230px}</style>"
            "<h2>Referencia x gerado por checkpoint (vozes nao vistas)</h2>"
            "<p>SECS = similaridade com a voz real (maior = mais parecida). Recarregue a pagina apos cada checkpoint.</p>"
            f"<table><tr><th>voz / frase</th><th>REFERENCIA</th>{head}</tr>" + "".join(rows) + "</table>")
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
