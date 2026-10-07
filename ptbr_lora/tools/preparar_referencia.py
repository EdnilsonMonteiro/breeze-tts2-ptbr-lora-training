"""preparar_referencia.py — prepara a REFERÊNCIA DE VOZ que o módulo de inferência consome.

O que faz (e por que importa): a inferência condiciona a voz por *in-context prompt*, isto é,
o áudio de referência + sua transcrição. O treino viu referências de ~6 s, mono, 24 kHz, sem
silêncio e normalizadas — então mandar para a inferência o arquivo cru (18 s, 48 kHz, estéreo,
com silêncios) é extrapolação de formato e janela. Este script normaliza e corta exatamente
como a Fase B do treino e escolhe a referência mais "típica" do locutor.

  1. Normaliza a(s) gravação(ões) como no TREINO (mono -> 24 kHz -> trim top_db=40 -> peak-norm p/ 0.95 -> clip).
     (--sr 48000 reproduz o formato antigo da UI; o default agora é o que o adapter viu.)
  2. Fatia em candidatos de 4-10 s (cortes em silêncio), juntando trechos curtos.
  3. Escolhe o **medoid** (candidato com maior cos médio ECAPA contra os demais):
     a referência mais "típica" do locutor, que maximiza a similaridade esperada
     com qualquer outra gravação — em vez do candidato mais "bonito".
  4. Valida: duração, pico, RMS, coesão interna (cos entre candidatos) e, opcional,
     WER entre o texto declarado (--ref-text / --ref-text-file) e o ASR do corte.
  5. Grava a referência PRONTA PARA INFERÊNCIA em <workspace>/<Nome>/
     {ref.wav (normalizado), ref.txt, voice.json} + enroll_report.json.
     É esse par `ref.wav`+`ref.txt` que a UI e o `clone_voice.py` usam — não o arquivo cru.

Uso:
  # uma gravação longa (corta automaticamente)
  python preparar_referencia.py --src voz_autor.wav --name voz_autorA \
      --workspace "<PTBR_ARTIFACTS>/voices" --transcribe

  # várias gravações do mesmo locutor (escolhe o medoid entre elas)
  python preparar_referencia.py --src "C:/voz/*.wav" --name voz_autorA \
      --workspace "<PTBR_ARTIFACTS>/voices" --no-slice

  # valida o texto da referência
  python preparar_referencia.py --src voz_autor.wav --name Check \
      --workspace "C:/tmp/vozes" --ref-text-file ref_voz_autor.txt --transcribe
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import metrics  # noqa: E402

SR = 24_000  # = formato do treino (corpus a 24 kHz). --sr 48000 = legado (UI antiga).
MIN_CAND_S = 4.0
MAX_CAND_S = 10.0
PEAK_NORM = 0.95


# ------------------------------------------------------------------ audio
def normalize(path: Path) -> np.ndarray:
    """mono -> SR (24 kHz) -> trim -> peak-norm (formato do treino)."""
    import librosa

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    wav, _ = librosa.effects.trim(wav, top_db=40)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak < 0.30 or peak > 0.99:
        wav = wav * (PEAK_NORM / max(peak, 1e-9))
    return np.clip(wav, -1.0, 1.0).astype("float32")


def slice_candidates(wav: np.ndarray, min_s=MIN_CAND_S, max_s=MAX_CAND_S) -> list[np.ndarray]:
    """Fatia em trechos 4-10 s cortando em silencios; junta trechos curtos."""
    import librosa

    iv = librosa.effects.split(wav, top_db=35)
    if len(iv) == 0:
        return [wav] if len(wav) / SR >= 1.0 else []
    out: list[np.ndarray] = []
    cur_s: int | None = None
    cur_e: int | None = None
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = int(s), int(e)
            continue
        dur_if_join = (e - cur_s) / SR
        gap = (s - cur_e) / SR
        if dur_if_join <= max_s and gap <= 0.60:
            cur_e = int(e)
        else:
            seg = wav[cur_s:cur_e]
            if len(seg) / SR >= min_s:
                out.append(seg)
            cur_s, cur_e = int(s), int(e)
    if cur_s is not None and (cur_e - cur_s) / SR >= min_s:
        out.append(wav[cur_s:cur_e])
    return [s[: int(max_s * SR)] for s in out]


def medoid_index(paths: list[Path], device="cpu") -> tuple[int, np.ndarray]:
    E = np.stack([metrics.embed(str(p), device) for p in paths])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    return int(np.nanargmax(np.nanmean(S, axis=1))), S


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Enrollment/validacao de referencia de voz")
    ap.add_argument("--src", required=True, help="arquivo, pasta ou glob")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workspace", default=None, help="pasta de vozes (default: voices/)")
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--ref-text-file", default=None)
    ap.add_argument("--transcribe", action="store_true", help="ASR do corte escolhido")
    ap.add_argument("--no-slice", action="store_true", help="nao fatiar (trata cada --src como 1 candidato)")
    ap.add_argument("--sr", type=int, default=24_000, choices=[24_000, 48_000],
                    help="taxa da ref.wav (24000 = como no treino; 48000 = legado)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    srcs: list[Path] = []
    for pat in a.src.split(","):
        pat = pat.strip()
        p = Path(pat)
        if p.is_dir():
            srcs += sorted(p.glob("*.wav"))
        elif any(c in pat for c in "*?"):
            srcs += [Path(x) for x in sorted(glob.glob(pat))]
        elif p.is_file():
            srcs.append(p)
    srcs = [p for p in srcs if p.suffix.lower() in (".wav", ".flac", ".m4a", ".mp3", ".ogg")]
    if not srcs:
        raise SystemExit(f"[enroll] nenhum audio em --src {a.src!r}")

    global SR
    SR = int(a.sr)
    ws = Path(a.workspace) if a.workspace else HERE.parent.parent / "voices"
    outdir = ws / a.name
    if outdir.exists() and not a.force:
        print(f"[enroll] (aviso) {outdir} ja existe; use --force para sobrescrever")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[enroll] {len(srcs)} arquivo(s) de origem")
    print("=" * 78)

    # 1) candidatos
    cand_dir = outdir / "_candidates"
    cand_dir.mkdir(exist_ok=True)
    cands: list[Path] = []
    for i, src in enumerate(srcs):
        wav = normalize(src)
        dur = len(wav) / SR if len(wav) else 0.0
        segs = [wav] if a.no_slice else slice_candidates(wav)
        print(f"  {src.name}: {dur:5.2f}s normalizado -> {len(segs)} candidato(s)")
        for j, seg in enumerate(segs):
            p = cand_dir / f"cand{i:02d}_{j:02d}.wav"
            sf.write(str(p), seg, SR, subtype="PCM_16")
            cands.append(p)
    if not cands:
        raise SystemExit("[enroll] nenhum candidato >= 4 s; grave material com mais fala")

    # 2) medoid
    idx, S = medoid_index(cands, a.device)
    best = cands[idx]
    off = S[~np.isnan(S)]
    coesao = float(np.mean(off)) if off.size else float("nan")
    wav_best, _ = sf.read(str(best), dtype="float32")
    dur_best = len(wav_best) / SR
    peak = float(np.max(np.abs(wav_best)))
    rms = float(np.sqrt(np.mean(wav_best ** 2)))
    print(f"\n[medoid] escolhido: {best.name} ({dur_best:.2f}s, pico {peak:.3f}, rms {rms:.4f})")
    print(f"[medoid] coesao media entre candidatos (cos ECAPA) = {coesao:.4f}")
    for k, p in enumerate(cands):
        print(f"    cos medio de {p.name} = {np.nanmean(S[k]):.4f}")

    # 3) transcricao
    declared = (a.ref_text or "").strip()
    if not declared and a.ref_text_file:
        declared = Path(a.ref_text_file).read_text(encoding="utf-8").strip()
    asr = None
    wer_prompt = None
    if a.transcribe:
        asr = metrics.transcribe(best, device=a.device)
        print(f"\n[asr] corte escolhido: {asr}")
        if declared:
            wer_prompt, cer = metrics.wer_cer(declared, asr)
            print(f"[asr] WER(texto declarado vs ASR do corte) = {wer_prompt:.3f}  CER = {cer:.3f}")

    # O texto do prompt PRECISA descrever exatamente o audio do prompt. Se o codigo
    # fatiou/cortou, o texto declarado do arquivo original deixa de valer -> usamos
    # o ASR do corte (revise antes de publicar) e avisamos.
    fatiado = not a.no_slice
    if declared and asr and wer_prompt is not None and wer_prompt > 0.05:
        if fatiado:
            print("[asr] o corte e um TRECHO do audio original -> ref.txt passa a ser o "
                  "ASR do corte (revise o texto antes de usar)")
        else:
            print("      ATENCAO: a transcricao declarada NAO confere com o audio. "
                  "Corrija o ref.txt (o modelo usa texto+audio como prompt).")
    final_text = declared if (declared and not (fatiado and asr)) else (asr or declared)
    if not final_text:
        raise SystemExit("[enroll] sem texto de referencia: rode com --transcribe ou informe "
                         "--ref-text/--ref-text-file")

    # 4) salva no formato da UI
    import shutil

    shutil.copyfile(best, outdir / "ref.wav")
    (outdir / "ref.txt").write_text(final_text.strip() + "\n", encoding="utf-8")
    meta = {
        "name": a.name,
        "ref_audio": "ref.wav",
        "ref_path": str(best),
        "ref_text": final_text.strip(),
        "ref_text_from_asr": bool(asr and final_text == asr),
        "ref_text_declared": declared,
        "ref_normalizada": {"sr": SR, "canais": 1, "trim_db": 40, "peak": PEAK_NORM,
                            "subtype": "PCM_16",
                            "nota": "ja normalizada no formato que o prompt de inferencia espera"},
        "adapter": "",
        "emotion": "Neutro / natural",
        "instruction": "Fale com clareza e naturalidade.",
        "speaker": "S0",
        "params": {"temperature": 0.7, "top_k": 50, "top_p": 1.0, "cfg_scale": 1.0,
                   "use_dual_cfg": False, "cfg_ref": 1.2, "cfg_ins": 1.0,
                   "max_new_tokens": 300, "seed": 1},
        "enrollment": {"n_candidates": len(cands), "chosen": best.name,
                       "dur_s": round(dur_best, 3), "peak": round(peak, 4),
                       "rms": round(rms, 5), "cohesion_cos": round(coesao, 4),
                       "wer_declared_vs_asr": wer_prompt},
    }
    (outdir / "voice.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    report = {"voice": a.name, "workspace": str(outdir), "chosen": str(best),
              "dur_s": dur_best, "cohesion_cos": coesao, "wer_declared_vs_asr": wer_prompt,
              "ref_text_usado": final_text, "ref_text_veio_do_asr": meta["ref_text_from_asr"],
              "candidates": [str(c) for c in cands],
              "cos_matrix_mean": [float(np.nanmean(S[k])) for k in range(len(cands))]}
    (outdir / "enroll_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    print(f"\n[ref] pronto para inferência -> {outdir}")
    print(f"[ref]   ref.wav : mono {SR} Hz, trim top_db=40, pico {PEAK_NORM} "
          f"({dur_best:.2f}s, rms {rms:.4f})")
    print(f"[ref]   ref.txt : {final_text[:80]!r}{'...' if len(final_text) > 80 else ''}")
    ok = (MIN_CAND_S <= dur_best <= MAX_CAND_S) and (coesao >= 0.80 or len(cands) == 1) \
        and (wer_prompt is None or wer_prompt <= 0.05 or meta["ref_text_from_asr"])
    print(f"[ref] veredito: {'OK' if ok else 'REVISAR'} "
          f"(duracao {MIN_CAND_S}-{MAX_CAND_S}s | coesao >= 0.80 | WER <= 0.05)")
    print("\n[ref] como consumir (a referência acima é a que deve ir para a inferência):")
    print("  # geração em blocos <=10 s com medoid (este repo):")
    print(f"  python ptbr_lora/tools/gerar_em_blocos.py --voice {a.name} "
          f"--workspace {outdir.parent} \\\n"
          f"      --adapter <checkpoint> --text-file seu_texto.txt --candidates 8 --out <saida>")
    print("  # clonagem simples (repo de inferência, template ref_edit):")
    print(f"  python lora/infer/clone_voice.py --ref-audio \"{outdir / 'ref.wav'}\" "
          f"--ref-text-file \"{outdir / 'ref.txt'}\" --text \"...\" --out saida.wav")
    print(f"  # a UI usa <workspace>/<Nome>/ref.wav+ref.txt automaticamente "
          f"(basta selecionar a voz '{a.name}')")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
