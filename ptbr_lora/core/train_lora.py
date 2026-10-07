"""train_lora.py — FASE C: treino LoRA do Breeze TTS 2 p/ PT-BR na RTX 4060 Ti 16GB.

Gates implementados:
  G1: LoraConfig com EXCLUSAO EXPLICITA de todo o codec_model (exclude_modules).
  G2 (modo --smoke): N steps curtos imprimindo
      - print_trainable_parameters() + dump da lista de modulos treinaveis
        (assert automatico: nenhum treinavel fora backbone/depth/text_encoder);
      - estabilidade da loss (media por janela, sem NaN);
      - pico de VRAM impresso (< 13-14 GB exigido).

Uso:
  SMOKE : python train_lora.py --run smoke --smoke --steps 30
  FULL  : python train_lora.py --run myrun --epochs 3
  r74   : python train_lora.py --run r74_01 --sampling voice --total-steps 2500 --ref-aug-p 0.5 \\
              --targets attn --rank 64 --alpha 128 --eval-unseen-every 500

Amostras audiveis:
  training/runs/<run>/samples/checkpoint-<tag>/<NN>_<slug>.wav
  (a cada epoca e no fim; e a cada --sample-every-steps quando houver)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB
import prepare_dataset as PD
import ref_aug as RA
import refs as R
import text_norm
import unseen_eval as UE
import adapter_init as AI
import voices as V
from lr_schedule import cosine_lr_factor

CB.GOLD_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CB.REPO))
# eval/ fora do pacote (eval_wer) precisa estar no path para o WER por checkpoint
_EVAL_DIR = _CORE.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import numpy as np
import torch

DEV = "cuda"

from sample_texts import SAMPLE_TEXTS  # noqa: E402  (fonte unica)
SAMPLES_PER_EVENT = len(SAMPLE_TEXTS)

TARGET_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
}
DEFAULT_CORPUS_WEIGHTS = {"tagarela": 0.7, "cetuc": 2.0, "cml_pt": 2.0,
                          "podcast": 1.5, "tata": 1.0}


class _Tee:
    """Espelha stdout/stderr no terminal E em arquivo (progresso ao vivo + log), sem redirecionar no .bat."""

    def __init__(self, stream, f):
        self._s, self._f = stream, f

    def write(self, x):
        try:
            self._s.write(x)
        except UnicodeEncodeError:                      # console cp850/cp1252 sem o caractere
            self._s.write(x.encode("ascii", "replace").decode("ascii"))
        self._f.write(x)
        return len(x)

    def flush(self):
        try:
            self._s.flush()
        finally:
            self._f.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


def tee_console(path: Path) -> None:
    f = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:40]


# ------------------------------------------------------------------ data


class TrainDataset(torch.utils.data.Dataset):
    """Itens montados on-the-fly via cache de codes.

    Condicao de cada item = `refs.resolve_condition(rec, pools, mix, epoch)`:
      * pools sao construidos SO com os idxs deste dataset (treino usa treino, val usa val);
      * a referencia e OUTRO clipe do mesmo locutor, re-sorteado a cada epoca (`set_epoch`);
      * `cross_ref_only=True` (validacao): so variantes com referencia, itens sem referencia
        possivel (locutor de 1 clipe, bucket :99) sao descartados; condicao fixa (epoca 0).
    """

    def __init__(self, idxs: list[str], tokenizer, mix: dict | None = None,
                 cross_ref_only: bool = False, seed: int = 0, ref_aug_p: float = 0.0):
        self.tokenizer = tokenizer
        self.records = {r["idx"]: r for r in PD.load_meta()}
        self.idxs = [i for i in idxs if i in self.records]
        self.mix = R.restrict_to_ref(mix or PD.CONDITION_MIX) if cross_ref_only else \
            dict(mix or PD.CONDITION_MIX)
        self.cross_ref_only = cross_ref_only
        self.seed = seed
        self.epoch = 0
        self.pools = R.build_pools(list(self.records.values()), self.idxs)
        if cross_ref_only:
            keep = []
            for i in self.idxs:
                if R.resolve_condition(self.records[i], self.pools, self.mix, 0, seed,
                                       require_ref=True) is not None:
                    keep.append(i)
            self.n_dropped_no_ref = len(self.idxs) - len(keep)
            self.idxs = keep
        else:
            self.n_dropped_no_ref = 0
        # augmentation SO da referencia: tokens de versoes degradadas (tools/augment_refs.py)
        self.ref_aug_p = 0.0 if cross_ref_only else float(ref_aug_p)
        self.aug_dir = CB.TRAINING / "tokens_aug"
        self.aug_avail: dict[str, list[int]] = {}
        if self.ref_aug_p > 0:
            import os

            if self.aug_dir.is_dir():
                for fn in os.listdir(self.aug_dir):
                    if fn.endswith(".npz") and "__a" in fn:
                        base, k = fn[:-4].rsplit("__a", 1)
                        if k.isdigit():
                            self.aug_avail.setdefault(base, []).append(int(k))
                for v in self.aug_avail.values():
                    v.sort()
            if not self.aug_avail:
                print(f"[aug] AVISO: --ref-aug-p={self.ref_aug_p} mas sem tokens em {self.aug_dir} "
                      f"(rode tools/augment_refs.py); treino segue SEM augmentation")
                self.ref_aug_p = 0.0

    def __len__(self) -> int:
        return len(self.idxs)

    def set_epoch(self, epoch: int) -> None:
        """`epoch` e so o SAL do sorteio de condicao/referencia/augmentation (no modo por massa de
        locutor e a posicao do sorteio -> cada amostra recebe uma condicao independente)."""
        self.epoch = 0 if self.cross_ref_only else int(epoch)

    def ref_aug_choice(self, rec: dict, ref_idx: str) -> Path | None:
        """Caminho dos tokens aumentados da referencia (ou None = referencia original)."""
        k = RA.choose_aug_variant(self.seed, self.epoch, rec["idx"], self.aug_avail.get(ref_idx),
                                  self.ref_aug_p)
        return None if k is None else self.aug_dir / f"{ref_idx}__a{k}.npz"

    def condition(self, i: int) -> tuple[str, str | None]:
        rec = self.records[self.idxs[i]]
        v, ref = R.resolve_condition(rec, self.pools, self.mix, self.epoch, self.seed,
                                     require_ref=self.cross_ref_only)
        return v, ref

    def __getitem__(self, i: int) -> dict:
        rec = self.records[self.idxs[i]]
        variant, ref = self.condition(i)
        ref_codes = self.ref_aug_choice(rec, ref) if ref else None
        ex = PD.build_item(self.tokenizer, rec, variant,
                           ref_rec=self.records[ref] if ref else None, ref_codes_path=ref_codes)
        return {
            "input_ids": ex["input_ids"][0],
            "attention_mask": torch.ones_like(ex["input_ids"][0]),
            "text_ids_mask": ex["text_ids_mask"][0],
            "text_ids_len": ex["text_ids_len"],
            "input_values": ex["input_values"].squeeze(0),
            "labels": ex["labels"][0],
        }

    def variant_histogram(self, max_items: int = 4000) -> dict[str, int]:
        from collections import Counter

        step = max(1, len(self.idxs) // max_items)
        return dict(Counter(self.condition(i)[0] for i in range(0, len(self.idxs), step)))

    def aug_fraction(self, max_items: int = 4000) -> float:
        """Fracao (amostra) dos itens COM referencia que recebem referencia aumentada."""
        step = max(1, len(self.idxs) // max_items)
        n = a = 0
        for i in range(0, len(self.idxs), step):
            v, ref = self.condition(i)
            if ref:
                n += 1
                a += int(self.ref_aug_choice(self.records[self.idxs[i]], ref) is not None)
        return a / n if n else 0.0


def load_split(name: str) -> list[str]:
    return PD.load_split_file(name)


def stratified_val_picks(ds_val: TrainDataset, n_items: int, min_per_corpus: int = 16
                         ) -> list[tuple[int, str]]:
    """Itens de val espalhados por corpus E por locutor, deterministico.

    Cota por corpus proporcional a raiz do tamanho (o tagarela nao afoga os corpora
    pequenos); dentro do corpus ordena por locutor e toma passos regulares (cobre locutores).
    """
    by: dict[str, list[int]] = {}
    for i, idx in enumerate(ds_val.idxs):
        by.setdefault(ds_val.records[idx].get("corpus", "tata"), []).append(i)
    if not by:
        return []
    root = {c: math.sqrt(len(v)) for c, v in by.items()}
    tot = sum(root.values())
    picks: list[tuple[int, str]] = []
    for corp in sorted(by):
        lst = sorted(by[corp], key=lambda i: (ds_val.records[ds_val.idxs[i]].get("speaker", ""),
                                              ds_val.idxs[i]))
        k = min(len(lst), max(min_per_corpus, round(n_items * root[corp] / tot)))
        step = len(lst) / k
        for j in range(k):
            picks.append((lst[int(j * step)], corp))
    return picks


def quick_val_loss(raw, ds_val: TrainDataset, n_items: int = 320) -> dict:
    """Val CROSS-REF (locutor nao visto, referencia != alvo), 1 item por forward.

    Pondera cada item pelo numero de frames supervisionados (senao clipes curtos pesam
    igual aos longos) e separa backbone (codebook 0) de depth decoder (1..15).
    """
    picks = stratified_val_picks(ds_val, n_items)
    acc = {"w": 0.0, "loss": 0.0, "b": 0.0, "d": 0.0}
    per: dict[str, dict] = {}
    was_training = raw.training
    raw.eval()
    with torch.no_grad():
        for i, corp in picks:
            item = ds_val[i]
            batch = CB.collate([item])
            n_sup = float((batch["labels"] == CB.AUDIO_TOKEN_ID).sum().item()) or 1.0
            batch = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = raw(**batch)
            vals = (out.loss.item(), out.backbone_loss.item(), out.depth_decoder_loss.item())
            if not all(math.isfinite(v) for v in vals):
                continue
            for d in (acc, per.setdefault(corp, {"w": 0.0, "loss": 0.0, "b": 0.0, "d": 0.0})):
                d["w"] += n_sup
                d["loss"] += vals[0] * n_sup
                d["b"] += vals[1] * n_sup
                d["d"] += vals[2] * n_sup
    if was_training:
        raw.train()

    def fin(d):
        w = d["w"]
        return {"loss": d["loss"] / w, "backbone": d["b"] / w, "depth": d["d"] / w} if w else \
            {"loss": float("nan"), "backbone": float("nan"), "depth": float("nan")}

    res = fin(acc)
    res["per_corpus"] = {c: fin(d)["loss"] for c, d in per.items()}
    res["n_items"] = len(picks)
    return res


# -------------------------------------------------------------- amostras WAV


def sanitize_inference_tensors(model) -> int:
    """Neutraliza tensores criados em inference_mode que possam ter ficado
    presos em BUFFERS persistentes de modulos (p.ex. rope inv_freq rebinds)
    e derruba caches de cudagraph/fast-path ligados ao objeto do modelo.
    Evita 'Inference tensors cannot be saved for backward' no treino."""
    n_fixed = 0
    for mod in model.modules():
        for name, buf in list(mod.named_buffers(recurse=False)):
            if isinstance(buf, torch.Tensor) and buf.is_inference():
                setattr(mod, name, buf.clone())
                n_fixed += 1
    for attr in (
        "_fast_text_encoder_cudagraph",
        "_fast_text_encoder_graph_cache",
        "_backbone_graph",
        "_depth_decoder_graph",
        "_stream_runtime",
    ):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass
    return n_fixed


def pick_sample_refs(ds_val: TrainDataset, n: int = 2) -> list[dict]:
    """Referencias das amostras: clipes do VAL (locutores nao vistos), corpora distintos,
    4-9 s, deterministico. E a MESMA condicao do uso real (ref_edit_tata com referencia)."""
    by: dict[str, list[dict]] = {}
    for i in ds_val.idxs:
        r = ds_val.records[i]
        if R.eligible_ref_speaker(r.get("speaker")) and 4.0 <= float(r.get("dur_proc_s", 0)) <= 9.0:
            by.setdefault(r.get("corpus", "tata"), []).append(r)
    out: list[dict] = []
    for corp in ("cetuc", "cml_pt", "tagarela", "podcast", "tata"):
        lst = sorted(by.get(corp, []), key=lambda r: r["idx"])
        if lst:
            r = lst[len(lst) // 2]
            out.append({"idx": r["idx"], "corpus": corp, "speaker": r.get("speaker"),
                        "audio": str(CB.WAVS24_DIR / f"{r['idx']}.wav"), "text": r["text"]})
        if len(out) >= n:
            break
    return out


def generate_samples(raw, tokenizer, out_dir, tag: str, seed0: int = 1000,
                     sample_refs: list[dict] | None = None,
                     jobs: list[tuple[str, str, dict]] | None = None,
                     temperature: float = 0.9):
    """Amostragem EAGER via BreezeForConditionalGeneration.generate(output_audio=True).

    Condicao = a do uso real: `ref_edit_tata` com uma referencia de locutor do VAL
    (nao visto no treino). Sem referencias disponiveis, cai em `tts_instruction` (sem ref).
    `jobs=[(nome, texto, ref)]` substitui as SAMPLE_TEXTS: gera `<nome>.wav` com a referencia
    dada (usado pela avaliacao SECS/WER em vozes nao vistas, `unseen_eval`).
    NAO usa FastBreezeStreamingRuntime/iter_audio_chunks: aquele caminho roda sob
    @torch.inference_mode() e monta cudagraphs NO OBJETO DO TREINO, deixando
    inference tensors presos nos modulos -> crash de backward na proxima epoca.
    """
    import soundfile as sf
    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    audio_tok = CB.load_audio_tokenizer(DEV)
    sr_out = audio_tok.get_output_sample_rate()
    update_generation_config_for_breeze(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = raw.training
    raw.eval()
    if jobs is not None:
        work = [(f"{nm}.wav", txt, ref, nm, False) for nm, txt, ref in jobs]
        use_ref = True
    else:
        use_ref = bool(sample_refs)
        work = []
        for k, (name, text) in enumerate(SAMPLE_TEXTS[:SAMPLES_PER_EVENT]):
            ref = sample_refs[k % len(sample_refs)] if use_ref else None
            work.append((f"{k:02d}_{slug(name)}.wav", text, ref, name, name.endswith("-en")))
    template = get_template("ref_edit_tata" if use_ref else "tts_instruction")
    print(f"[samples] gerando {len(work)} amostras ({tag}) ...", flush=True)
    meta: list[dict] = []
    try:
        with torch.no_grad():  # somente no_grad: nao contamina o modelo
            for k, (fname, text, ref, name, is_en) in enumerate(work):
                request = {
                    "id": f"eval-{tag}-{k}",
                    "text": text if is_en else text_norm.normalize(text),
                    "instruction": CB.DEFAULT_INSTRUCTION,
                    "speaker": "S0",
                }
                if use_ref:
                    request["ref_audio_path"] = ref["audio"]
                    request["ref_text"] = text_norm.normalize(ref["text"])
                set_all_seeds(seed0 + k)
                inputs = prepare_inputs(
                    tokenizer, audio_tok, raw, [request], template,
                    guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None,
                )
                if not use_ref:
                    inputs.pop("input_values", None)  # prompt puramente textual
                    inputs.pop("cfg_scale", None)
                path = out_dir / fname
                t_gen = time.time()
                try:
                    res = raw.generate(
                        **inputs,
                        output_audio=True,
                        audio_tokenizer=audio_tok,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=temperature,
                        top_k=50,
                        top_p=1.0,
                    )
                    if isinstance(res, (list, tuple)):
                        wav_t = res[0]
                    elif hasattr(res, "audio"):
                        wav_t = res.audio[0] if res.audio else None
                    else:
                        wav_t = res
                    if wav_t is None:
                        print(f"    [samples] '{name}': sem audio retornado")
                        continue
                    while wav_t.dim() > 1:
                        wav_t = wav_t[0]
                    wav = wav_t.detach().float().cpu().numpy()
                    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr_out, subtype="PCM_16")
                    meta.append({"file": path.name, "name": name,
                                 "ref_idx": ref["idx"] if ref else None,
                                 "ref_speaker": ref["speaker"] if ref else None})
                    print(f"    [samples] {k + 1}/{len(work)} {fname} ({time.time() - t_gen:.1f}s)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    [samples] ERRO '{name}': {type(exc).__name__}: {str(exc)[:140]}"
                    )
        (out_dir / "samples_meta.json").write_text(json.dumps(
            {"mode": "ref_edit_tata" if use_ref else "tts_instruction", "items": meta},
            ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        n_fix = sanitize_inference_tensors(raw)
        del audio_tok
        torch.cuda.empty_cache()
        if was_training:
            raw.train()
        if n_fix:
            print(f"    [sanitize] {n_fix} buffers inference->normal restaurados")


def git_info() -> dict:
    try:
        rev = subprocess.run(["git", "--no-optional-locks", "rev-parse", "--short", "HEAD"],
                             cwd=str(CB.REPO), capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "--no-optional-locks", "status", "--porcelain",
                                     "--untracked-files=no"], cwd=str(CB.REPO),
                                    capture_output=True, text=True, timeout=20).stdout.strip())
        return {"commit": rev, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-acc", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=5.0,
                    help="clip global; a fracao de passos clipados vai para o log (clip_frac)")
    ap.add_argument("--lr-floor", type=float, default=0.0,
                    help="piso do LR (fracao do lr); >0 ativa cosseno com restarts")
    ap.add_argument("--lr-cycles", type=int, default=3,
                    help="numero de ciclos cosseno quando --lr-floor > 0")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=sorted(TARGET_PRESETS), default="attn",
                    help="attn = q/k/v/o | all = + gate/up/down (MLPs)")
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--exclude-text-encoder", action="store_true",
                    help="nao adapta o text_encoder (T5Gemma2); so backbone + depth decoder")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="eager",
                    help="implementacao de atencao (eager = validada; sdpa = mais rapida, NAO validada)")
    ap.add_argument("--mix", type=str, default=None,
                    help="mistura de condicoes, ex.: 'ref_edit=0.75,ref_clone=0.1,instruction=0.1,plain=0.05' "
                         "(default = refs.DEFAULT_MIX)")
    ap.add_argument("--ref-edit-frac", type=float, default=None,
                    help="LEGADO: F ref_edit + (1-F) tts_instruction. Ignorado se --mix for dado")
    ap.add_argument("--instruction-mode", choices=["fixed", "pool"], default="fixed",
                    help="fixed = a instrucao de producao (default); pool = 9 frases variadas")
    ap.add_argument("--no-text-norm", action="store_true",
                    help="NAO normaliza o texto de treino (default: text_norm, igual a inferencia)")
    ap.add_argument("--val-items", type=int, default=320)
    ap.add_argument("--val-every-steps", type=int, default=1000, help="0 = so no fim da epoca")
    ap.add_argument("--corpus-weights", type=str, default=None,
                    help="JSON {corpus: peso}; default = oversampling 24kHz+ 2x")
    ap.add_argument("--unassigned-weight", type=float, default=0.5,
                    help="peso extra dos itens de locutor nao atribuido (:99)")
    ap.add_argument("--sampling", choices=["corpus", "voice"], default="corpus",
                    help="corpus = pesos por corpus (legado r70-r73) | voice = massa por LOCUTOR com "
                         "teto (--cap-spk/--cap-grp): nenhuma voz domina o treino")
    ap.add_argument("--cap-spk", type=float, default=0.01, help="teto de massa por voz (--sampling voice)")
    ap.add_argument("--cap-grp", type=float, default=0.02, help="teto de massa por grupo (show/episodio)")
    ap.add_argument("--mass-beta", type=float, default=0.5,
                    help="massa da voz dentro do corpus ∝ n_clipes**beta (0 = uniforme por voz)")
    ap.add_argument("--corpus-share", type=str, default=None,
                    help="JSON {corpus: fracao} da massa por corpus (default voices.DEFAULT_CORPUS_SHARE)")
    ap.add_argument("--max-repeat", type=float, default=6.0,
                    help="teto: um clipe nao e visto mais que N vezes no treino todo (0 = sem teto)")
    ap.add_argument("--total-steps", type=int, default=0,
                    help=">0: treino de N passos de otimizador em UMA passada amostrada (substitui "
                         "--epochs; condicao/referencia/augmentation re-sorteadas a cada amostra)")
    ap.add_argument("--ref-aug-p", type=float, default=0.0,
                    help="prob. de trocar a REFERENCIA por uma versao degradada (tokens_aug/; o alvo "
                         "nunca e aumentado). Requer tools/augment_refs.py")
    ap.add_argument("--eval-unseen-every", type=int, default=0,
                    help="a cada N passos (e no fim) gera 2 frases x vozes NAO vistas e mede SECS/WER "
                         "-> eval_unseen.csv (0 = desliga)")
    ap.add_argument("--eval-unseen-start", action="store_true",
                    help="avalia tambem no passo 0 (modelo base, sem treino) p/ ter a linha de base")
    ap.add_argument("--eval-unseen-voices", type=int, default=4)
    ap.add_argument("--eval-unseen-phrases", type=int, default=2)
    ap.add_argument("--eval-extra-audio", type=str, default=None,
                    help="wav de referencia de uma voz EXTRA (ex.: a sua) p/ a avaliacao periodica")
    ap.add_argument("--eval-extra-text", type=str, default=None, help="transcricao do wav acima (texto ou caminho de um .txt)")
    ap.add_argument("--eval-extra-heldout", type=str, default=None,
                    help="wavs da MESMA voz, fora do prompt, separados por ';' (alvo do SECS)")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--no-wer", action="store_true",
                    help="desativa WER/CER automatico por checkpoint (faster-whisper CPU)")
    ap.add_argument("--sample-every-steps", type=int, default=500)
    ap.add_argument("--state-every-steps", type=int, default=500,
                    help="salva checkpoints/resume (adapter + otimizador) a cada N passos")
    ap.add_argument("--sample-ref-audio", type=str, default=None,
                    help="wav de referencia p/ as amostras (default: 2 clipes do val)")
    ap.add_argument("--sample-ref-text", type=str, default=None)
    ap.add_argument(
        "--resume-adapter", type=str, default=None,
        help="pasta do adapter para continuar (otimizador NOVO; ver --resume-state)")
    ap.add_argument("--resume-state", type=str, default=None,
                    help="pasta checkpoints/resume: retoma adapter + otimizador + scheduler + posicao")
    ap.add_argument(
        "--init-from-adapter", type=str, default=None,
        help="WARM START com MAIS alvos: cria o LoRA com --targets (ex.: all) e copia os pesos de um adapter "
             "de menos alvos (ex.: r74 so atencao); os modulos novos (MLP) comecam neutros (B=0), entao o "
             "passo 0 e identico ao adapter de origem. rank/alpha/rsLoRA precisam ser os mesmos. Otimizador NOVO.")
    ap.add_argument(
        "--eval-voices-json", type=str, default=None,
        help="JSON [{name, audio, text|caminho.txt, heldout?}] com as vozes de referencia do usuario na "
             "avaliacao periodica (todas as frases de --eval-unseen-phrases); soma-se a --eval-extra-*")
    ap.add_argument(
        "--eval-val-phrase-ids", type=str, default=None,
        help="indices das frases geradas para as vozes NAO vistas do val (ex.: 1 ou 0,1); default = todas. "
             "Menos geracoes por avaliacao; as vozes do usuario sempre geram todas")
    args = ap.parse_args()
    if args.init_from_adapter and (args.resume_adapter or args.resume_state):
        ap.error("--init-from-adapter nao combina com --resume-adapter/--resume-state (use um OU outro)")
    if args.init_from_adapter:
        AI.check_compat(args.init_from_adapter, rank=args.rank, alpha=args.alpha, use_rslora=args.use_rslora)
    val_phrase_ids = ([int(x) for x in args.eval_val_phrase_ids.split(",") if x.strip() != ""]
                      if args.eval_val_phrase_ids else None)

    batch_size = args.batch or (2 if args.smoke else 4)
    grad_acc = args.grad_acc or (2 if args.smoke else 8)
    if args.resume_state and not args.resume_adapter:
        args.resume_adapter = args.resume_state

    run_dir = CB.TRAINING / "runs" / args.run
    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for d in (run_dir, ckpt_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)
    tee_console(run_dir / "console.log")                       # terminal ao vivo + log da corrida

    LOG_COLS = ["step", "epoch", "opt_step", "total_loss", "backbone_loss", "depth_loss",
                "val_loss", "val_backbone", "val_depth", "lr", "grad_norm", "clip_frac",
                "vram_win_gb", "vram_peak_gb", "s_per_step", "skipped"]
    log_path = run_dir / "log.csv"
    log_f = log_path.open("a", encoding="utf-8")
    if log_f.tell() == 0:
        log_f.write(",".join(LOG_COLS) + "\n")

    def log_row(**kw) -> None:
        log_f.write(",".join(str(kw.get(c, "nan")) for c in LOG_COLS) + "\n")
        log_f.flush()

    print(
        f"[cfg] run={args.run} smoke={args.smoke} batch={batch_size} acc={grad_acc} "
        f"lr={args.lr} r={args.rank} a={args.alpha} attn={args.attn}"
    )

    # ---------------------------------------------------------- modelo base
    t0 = time.time()
    raw = CB.load_breeze_model(DEV, attn=args.attn)
    raw.gradient_checkpointing_enable()
    raw.enable_input_require_grads()
    print(f"[load] pronto em {time.time() - t0:.0f}s | gradient checkpointing ON")

    from peft import LoraConfig, get_peft_model

    init_report = None
    if args.resume_adapter:
        from peft import PeftModel

        print(f"[resume] Carregando adapter existente de: {args.resume_adapter}")
        pm = PeftModel.from_pretrained(raw, args.resume_adapter, is_trainable=True)
    else:
        excl = r".*(codec_model|text_encoder).*" if args.exclude_text_encoder else r".*codec_model.*"
        lconf = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=TARGET_PRESETS[args.targets],
            use_rslora=args.use_rslora,
            exclude_modules=excl,
        )
        pm = get_peft_model(raw, lconf)
        if args.init_from_adapter:
            init_report = AI.load_into(pm, args.init_from_adapter)
            print(f"[init] warm start de {args.init_from_adapter}: {init_report}")
    pm.print_trainable_parameters()

    # ------------------------- GATE G1: auditável + asserts de escopo
    PREFIX = "base_model.model."  # PeftModel -> LoraModel -> raw
    trainable_lines, stats, bad = (
        [],
        {"backbone_model": 0, "depth_decoder": 0, "text_encoder": 0},
        [],
    )
    for name, p in pm.named_parameters():
        if not p.requires_grad:
            continue
        trainable_lines.append(name)
        core = name.removeprefix(PREFIX)
        hit = next((s for s in stats if core.startswith(s + ".")), None)
        if hit is None:
            bad.append((name, tuple(p.shape)))
        else:
            stats[hit] += p.numel()
    assert not bad, f"GATE G1 VIOLADO - treinavel fora dos stacks: {bad[:10]}"
    assert not any(".codec_model." in n for n in trainable_lines), "codec nas targets!"
    if args.exclude_text_encoder:
        assert stats["text_encoder"] == 0, "text_encoder deveria estar excluido"
    n_train = sum(stats.values())
    print(f"[G1] treinaveis por stack: {stats} | total={n_train:,}")
    (run_dir / "trainable_modules.txt").write_text(
        "\n".join(sorted(trainable_lines)), encoding="utf-8"
    )

    # ---------------------------------------------------------- dados
    tokenizer = CB.load_text_tokenizer()
    if args.mix:
        PD.set_condition_mix(args.mix)
    elif args.ref_edit_frac is not None:
        PD.set_ref_edit_frac(args.ref_edit_frac)
    PD.set_instruction_mode(args.instruction_mode)
    PD.set_text_normalize(not args.no_text_norm)
    ds = TrainDataset(load_split("train"), tokenizer, seed=1, ref_aug_p=args.ref_aug_p)
    ds_val = TrainDataset(load_split("val"), tokenizer, cross_ref_only=True, seed=2)
    print(f"[data] mix={ {k: round(v, 3) for k, v in ds.mix.items()} } | "
          f"instrucao={args.instruction_mode} | text_norm={not args.no_text_norm}")
    print(f"[data] variantes efetivas (amostra, epoca 0)={ds.variant_histogram()}")
    if ds.ref_aug_p > 0:
        print(f"[aug] referencia aumentada: p={ds.ref_aug_p} | clipes com variantes={len(ds.aug_avail)} | "
              f"fracao efetiva (itens com ref)={ds.aug_fraction():.2f}")
    print(f"[data] val cross-ref: {len(ds_val)} itens ({ds_val.n_dropped_no_ref} sem ref possivel "
          f"descartados) | val itens/avaliacao={args.val_items}")
    cw = dict(DEFAULT_CORPUS_WEIGHTS)
    if args.corpus_weights:
        cw.update(json.loads(args.corpus_weights))
    micro_per_step = batch_size * grad_acc
    steps_per_epoch = max(1, len(ds) // micro_per_step)          # floor: sem passo parcial
    if args.total_steps > 0 and not args.smoke:
        steps_per_epoch = int(args.total_steps)                  # UMA passada amostrada
        args.epochs = 1
    opt_steps_total = args.steps if args.smoke else steps_per_epoch * args.epochs
    from collections import Counter as _Counter
    voice_info = None
    if args.sampling == "voice":
        n_draw = steps_per_epoch * micro_per_step * (1 if args.smoke else args.epochs)
        share = None
        if args.corpus_share:   # JSON inline ou caminho de um arquivo .json (evita aspas no cmd do Windows)
            share = json.loads(Path(args.corpus_share).read_text(encoding="utf-8")
                               if Path(args.corpus_share).is_file() else args.corpus_share)
        wmap, voice_info = V.speaker_mass_weights(
            [ds.records[i] for i in ds.idxs], corpus_share=share, beta=args.mass_beta,
            cap_spk=args.cap_spk, cap_grp=args.cap_grp, total_samples=n_draw,
            max_repeat=args.max_repeat)
        weights = [wmap[i] for i in ds.idxs]
        print(f"[data] amostragem por MASSA DE LOCUTOR: vozes={voice_info['n_voices']} "
              f"grupos={voice_info['n_groups']} voz_max={voice_info['max_voice_mass']:.4f} "
              f"grupo_max={voice_info['max_group_mass']:.4f} vozes_efetivas={voice_info['eff_voices']:.0f} "
              f"repeticao_max={voice_info['max_repeat_obs']:.1f}x relaxado={voice_info['relaxed']}")
        print(f"[data] massa por corpus={ {c: round(x, 3) for c, x in voice_info['corpus_mass'].items()} }")
        (run_dir / "voice_mass.json").write_text(json.dumps(voice_info, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
    else:
        def _w(i: str) -> float:
            rec = ds.records.get(i, {})
            w = float(cw.get(rec.get("corpus", "tata"), 1.0))
            if not R.eligible_ref_speaker(rec.get("speaker")):
                w *= float(args.unassigned_weight)
            return w

        weights = [_w(i) for i in ds.idxs]
        print(f"[data] corpus weights={cw} | "
              f"dist={dict(_Counter(ds.records.get(i, {}).get('corpus', 'tata') for i in ds.idxs))}")
    print(
        f"[data] itens={len(ds)} val={len(ds_val)} "
        f"steps/epoca={steps_per_epoch} total_alvo={opt_steps_total}"
    )
    if args.sample_ref_audio:
        sample_refs = [{"idx": "cli", "corpus": "cli", "speaker": "cli",
                        "audio": args.sample_ref_audio, "text": args.sample_ref_text or ""}]
        assert args.sample_ref_text, "--sample-ref-audio exige --sample-ref-text"
    else:
        sample_refs = pick_sample_refs(ds_val)
    print(f"[samples] modo={'ref_edit_tata' if sample_refs else 'tts_instruction (sem refs no val)'} "
          f"refs={[(r['corpus'], r['idx']) for r in sample_refs]}")
    unseen_plan = None
    if args.eval_unseen_every > 0:
        extra = []
        if args.eval_extra_audio:
            assert args.eval_extra_text and args.eval_extra_heldout, \
                "--eval-extra-audio exige --eval-extra-text e --eval-extra-heldout"
            _t = args.eval_extra_text
            if Path(_t).is_file():                              # aceita caminho de .txt (evita aspas no .bat)
                _t = Path(_t).read_text(encoding="utf-8").strip()
            extra.append({"name": "extra", "audio": args.eval_extra_audio, "text": _t,
                          "heldout": [x for x in args.eval_extra_heldout.split(";") if x.strip()]})
        if args.eval_voices_json:
            for ev in json.loads(Path(args.eval_voices_json).read_text(encoding="utf-8")):
                _t = ev["text"]
                if Path(_t).is_file():
                    _t = Path(_t).read_text(encoding="utf-8").strip()
                assert Path(ev["audio"]).is_file(), f"--eval-voices-json: audio inexistente: {ev['audio']}"
                extra.append({"name": ev["name"], "audio": ev["audio"], "text": _t,
                              "heldout": ev.get("heldout") or []})
        extra = extra or None
        unseen_plan = UE.build_plan(ds_val.records, load_split("val"), CB.WAVS24_DIR,
                                    args.eval_unseen_voices, extra)
        print(f"[unseen] avaliacao a cada {args.eval_unseen_every} passos | vozes="
              f"{[(v['name'], v['speaker']) for v in unseen_plan['voices']]} | "
              f"frases={args.eval_unseen_phrases}")
    splits_meta = CB.TRAINING / "splits_meta.json"
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "batch_size": batch_size,
                "grad_acc": grad_acc,
                "n_train_items": len(ds),
                "n_val_items_pool": len(ds_val),
                "opt_steps_total": opt_steps_total,
                "trainable_params": n_train,
                "per_stack": stats,
                "lora_scale": (args.alpha / (args.rank ** 0.5)) if args.use_rslora
                else (args.alpha / max(1, args.rank)),
                "condition_mix": ds.mix,
                "corpus_weights": cw,
                "voice_mass": voice_info,
                "ref_aug": {"p": ds.ref_aug_p, "clips_with_variants": len(ds.aug_avail)},
                "init_report": init_report,
                "git": git_info(),
                "torch": torch.__version__,
                "splits_meta": json.loads(splits_meta.read_text(encoding="utf-8"))
                if splits_meta.exists() else "splits legados (sem splits_meta.json)",
                "protocol": ("v4: v3 + massa por locutor c/ teto, aug so da referencia, "
                             "avaliacao SECS/WER em vozes nao vistas" if args.sampling == "voice"
                             else "v3: cross-ref aleatorio por epoca, sem self-ref, val cross-ref ponderado"),
            },
            indent=2, default=str, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    opt = torch.optim.AdamW(
        (p for p in pm.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    def lr_lambda(step):
        return cosine_lr_factor(step, opt_steps_total, args.warmup, args.lr_floor, args.lr_cycles)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    trainable_params = [p for p in pm.parameters() if p.requires_grad]

    # ----------------------------------------------------------- estado / resume
    global_step = 0
    start_epoch = 0
    skip_steps = 0            # passos ja feitos DENTRO da epoca retomada
    best_val = float("inf")
    n_skipped = 0
    if args.resume_state:
        st = torch.load(Path(args.resume_state) / "trainer_state.pt", map_location="cpu",
                        weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        global_step, start_epoch = int(st["global_step"]), int(st["epoch"])
        skip_steps, best_val = int(st["step_in_epoch"]), float(st["best_val"])
        n_skipped = int(st.get("n_skipped", 0))
        print(f"[resume] estado restaurado: passo={global_step} epoca={start_epoch} "
              f"+{skip_steps} passos | melhor val={best_val:.4f}")

    def save_checkpoint(tag_s: str):
        d = ckpt_dir / tag_s
        pm.save_pretrained(str(d), safe_serialization=True)
        print(f"[ckpt] salvo: {d}")
        return d

    def save_resume_state(epoch: int, step_in_epoch: int):
        d = ckpt_dir / "resume"
        pm.save_pretrained(str(d), safe_serialization=True)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                    "global_step": global_step, "epoch": epoch, "step_in_epoch": step_in_epoch,
                    "best_val": best_val, "n_skipped": n_skipped, "args": vars(args)},
                   d / "trainer_state.pt")
        print(f"[ckpt] estado de retomada salvo: {d} (passo {global_step})")

    tb = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb = SummaryWriter(log_dir=str(run_dir / "tb"))
            print(f"[tb] TensorBoard -> {run_dir / 'tb'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[tb] indisponivel: {type(exc).__name__}: {exc}")

    def run_wer(sample_dir: Path):
        if args.no_wer:
            return
        try:
            import eval_wer

            res = eval_wer.evaluate_dir(sample_dir, SAMPLE_TEXTS)
            if not res:
                return
            summ = eval_wer.summarize_results(res)
            pt = summ.get("pt", {})
            print(f"[wer] {sample_dir.name}: pt WER={pt.get('wer_mean', float('nan')):.3f} "
                  f"CER={pt.get('cer_mean', float('nan')):.3f} (n={pt.get('n', 0)}, "
                  f"falhas={pt.get('n_fail', 0)}) | en WER={summ.get('en', {}).get('wer_mean', float('nan')):.3f}")
            if tb is not None and pt:
                tb.add_scalar("wer_pt", pt["wer_mean"], global_step)
                tb.add_scalar("cer_pt", pt["cer_mean"], global_step)
                tb.add_scalar("wer_pt_fail", pt["n_fail"], global_step)
                tb.flush()
            (sample_dir / "wer.json").write_text(
                json.dumps({"summary": summ, "per_sample": res},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[wer] falhou: {type(exc).__name__}: {str(exc)[:120]}")

    def do_val(epoch: int, tag_s: str) -> dict:
        nonlocal best_val
        v = quick_val_loss(raw, ds_val, n_items=args.val_items)
        log_row(step=global_step, epoch=epoch, opt_step=global_step, val_loss=f"{v['loss']:.4f}",
                val_backbone=f"{v['backbone']:.4f}", val_depth=f"{v['depth']:.4f}",
                vram_peak_gb=f"{torch.cuda.max_memory_allocated() / 2**30:.2f}")
        msg = " ".join(f"{c}={x:.3f}" for c, x in sorted(v["per_corpus"].items()))
        print(f"[val {tag_s}] loss={v['loss']:.4f} (backbone={v['backbone']:.4f} "
              f"depth={v['depth']:.4f}; n={v['n_items']}) | {msg}")
        if tb is not None:
            tb.add_scalar("val/loss", v["loss"], global_step)
            tb.add_scalar("val/backbone", v["backbone"], global_step)
            tb.add_scalar("val/depth", v["depth"], global_step)
            for c, x in v["per_corpus"].items():
                tb.add_scalar(f"val/{c}", x, global_step)
            tb.flush()
        if v["loss"] == v["loss"] and v["loss"] < best_val:
            best_val = v["loss"]
            save_checkpoint("best")
            (ckpt_dir / "best" / "best_info.json").write_text(json.dumps(
                {"step": global_step, "val_loss": best_val, "tag": tag_s}), encoding="utf-8")
        return v

    def run_unseen() -> None:
        """SECS/WER em vozes NUNCA vistas (mesmas frases/seed em todo checkpoint -> curva comparavel)."""
        if unseen_plan is None:
            return
        try:
            if global_step > 0 and not (ckpt_dir / f"step{global_step}").exists():
                save_checkpoint(f"step{global_step}")           # adapter p/ varrer escala depois
            d = run_dir / "eval_unseen" / f"step{global_step}"
            jobs = UE.jobs_for(unseen_plan, args.eval_unseen_phrases, val_phrase_ids)
            t_u = time.time()
            print(f"[unseen step{global_step}] gerando {len(jobs)} amostras em vozes nao vistas ...", flush=True)
            generate_samples(raw, tokenizer, d, f"u{global_step}", seed0=7000, jobs=jobs)
            print(f"[unseen step{global_step}] pontuando SECS/WER (CPU) ...", flush=True)
            res = UE.score(d, unseen_plan, args.eval_unseen_phrases)
            if res is None:
                print("[unseen] sem resultado (dependencias/arquivos)")
                return
            UE.append_csv(run_dir / "eval_unseen.csv", global_step, res)
            lp = UE.write_listen_page(run_dir, unseen_plan, args.eval_unseen_phrases)
            if lp:
                print(f"[unseen] para OUVIR (referencia x gerado por checkpoint): {lp}", flush=True)
            pv = " | ".join(f"{n}={x['secs_heldout']:.3f}(teto {x['ceiling']:.2f})"
                            for n, x in res["per_voice"].items())
            print(f"[unseen step{global_step}] SECS={res['secs_heldout']:.3f} norm={res['secs_norm']:.2f} "
                  f"WER={res['wer']:.3f} CER={res['cer']:.3f} n={res['n']} ({time.time() - t_u:.0f}s) | {pv}",
                  flush=True)
            print(f"[unseen step{global_step}] VOZES NAO VISTAS: SECS={res['secs_val']:.3f} WER={res['wer_val']:.3f} | "
                  f"SUAS VOZES (vs referencia): SECS={res['secs_user']:.3f} WER={res['wer_user']:.3f}", flush=True)
            if tb is not None:
                for k in ("secs_heldout", "secs_prompt", "secs_norm", "wer", "cer",
                          "secs_val", "wer_val", "secs_user", "wer_user"):
                    if res[k] == res[k]:
                        tb.add_scalar(f"unseen/{k}", res[k], global_step)
                for n, x in res["per_voice"].items():
                    tb.add_scalar(f"unseen_secs/{n}", x["secs_heldout"], global_step)
                tb.flush()
        except Exception as exc:  # noqa: BLE001  (a avaliacao nunca derruba o treino)
            print(f"[unseen] falhou: {type(exc).__name__}: {str(exc)[:160]}")

    if args.eval_unseen_start and args.eval_unseen_every and not args.smoke and global_step == 0:
        run_unseen()                                           # linha de base (modelo sem treino)

    t_run = time.time()
    stop = False
    torch.cuda.reset_peak_memory_stats()
    tot_clip = tot_steps = 0

    step_in_epoch = 0
    epochs_to_run = 1 if args.smoke else args.epochs
    for epoch in range(start_epoch, epochs_to_run):
        if stop:
            break
        ds.set_epoch(epoch)                                    # re-sorteia refs/condicoes
        g = torch.Generator()
        g.manual_seed(42 + epoch)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), steps_per_epoch * micro_per_step,
            replacement=True, generator=g)
        order = list(sampler)
        first = skip_steps * micro_per_step if epoch == start_epoch else 0
        step_in_epoch = first // micro_per_step
        micro: list[dict] = []
        micro_done = 0                                        # micro-batches acumulados no passo
        win = {"n_micro": 0, "steps": 0, "loss": 0.0, "b": 0.0, "d": 0.0, "g": 0.0, "clip": 0}
        win_peak0 = torch.cuda.max_memory_allocated()
        t_win = time.time()
        raw.train()
        for j, i in enumerate(order[first:], start=first):
            if args.sampling == "voice":
                ds.set_epoch(epoch * 10_000_000 + j)           # condicao/ref/aug independentes por sorteio
            try:
                micro.append(ds[i])
            except Exception as exc:  # noqa: BLE001  (item quebrado nao derruba o run)
                n_skipped += 1
                print(f"[data] item {ds.idxs[i]} ignorado: {type(exc).__name__}: {str(exc)[:100]}")
                continue
            if len(micro) < batch_size:
                continue
            batch = CB.collate(micro)
            batch = {
                k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            micro.clear()
            out = raw(**batch)
            if not torch.isfinite(out.loss):
                n_skipped += 1
                print(f"[warn] loss nao-finita no passo {global_step} -> micro-batch ignorado")
                continue
            (out.loss / grad_acc).backward()
            micro_done += 1
            win["n_micro"] += 1
            win["loss"] += out.loss.item()
            win["b"] += out.backbone_loss.item()
            win["d"] += out.depth_decoder_loss.item()
            if micro_done < grad_acc:
                continue
            micro_done = 0

            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                n_skipped += 1
                print(f"[warn] grad nao-finito no passo {global_step} -> passo ignorado")
                continue
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            step_in_epoch += 1
            win["steps"] += 1
            win["g"] += float(gnorm)
            clipped = float(gnorm) > args.max_grad_norm
            win["clip"] += int(clipped)
            tot_clip += int(clipped)
            tot_steps += 1

            if global_step <= 3 or global_step % 5 == 0:
                n_m = max(1, win["n_micro"])
                n_s = max(1, win["steps"])
                mean_l, mean_b, mean_d = win["loss"] / n_m, win["b"] / n_m, win["d"] / n_m
                sp = (time.time() - t_win) / n_s                  # s por PASSO de otimizador
                win_peak = torch.cuda.max_memory_allocated()
                log_row(step=global_step, epoch=epoch, opt_step=global_step,
                        total_loss=f"{mean_l:.4f}", backbone_loss=f"{mean_b:.4f}",
                        depth_loss=f"{mean_d:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}",
                        grad_norm=f"{win['g'] / n_s:.3f}", clip_frac=f"{win['clip'] / n_s:.2f}",
                        vram_win_gb=f"{(win_peak - win_peak0) / 2**30:.2f}",
                        vram_peak_gb=f"{win_peak / 2**30:.2f}", s_per_step=f"{sp:.3f}",
                        skipped=n_skipped)
                eta_min = max(0, opt_steps_total - global_step) * sp / 60
                print(
                    f"[{global_step}/{opt_steps_total}] {100 * global_step / max(1, opt_steps_total):.0f}% "
                    f"ep{epoch} loss={mean_l:.3f} "
                    f"(b{mean_b:.3f}/d{mean_d:.3f}) g={win['g'] / n_s:.2f} "
                    f"clip={win['clip']}/{n_s} pk={win_peak / 2**30:.2f}GB "
                    f"{sp:.1f}s/passo | restam~{eta_min:.0f}min (sem contar val/amostras)",
                    flush=True,
                )
                if tb is not None:
                    tb.add_scalar("loss/total", mean_l, global_step)
                    tb.add_scalar("loss/backbone", mean_b, global_step)
                    tb.add_scalar("loss/depth", mean_d, global_step)
                    tb.add_scalar("lr", sched.get_last_lr()[0], global_step)
                    tb.add_scalar("grad_norm", win["g"] / n_s, global_step)
                    tb.add_scalar("clip_frac", win["clip"] / n_s, global_step)
                    tb.add_scalar("vram_peak_gb", win_peak / 2**30, global_step)
                    tb.flush()
                win = {"n_micro": 0, "steps": 0, "loss": 0.0, "b": 0.0, "d": 0.0, "g": 0.0, "clip": 0}
                win_peak0 = torch.cuda.max_memory_allocated()
                t_win = time.time()

            if args.smoke and global_step >= args.steps:
                stop = True
                break
            if not args.smoke:
                if args.val_every_steps and global_step % args.val_every_steps == 0:
                    do_val(epoch, f"step{global_step}")
                if args.sample_every_steps and global_step % args.sample_every_steps == 0:
                    save_checkpoint(f"step{global_step}")
                    generate_samples(raw, tokenizer, samples_dir / f"checkpoint-{global_step}",
                                     f"s{global_step}", seed0=1000 + global_step,
                                     sample_refs=sample_refs)
                if args.state_every_steps and global_step % args.state_every_steps == 0:
                    save_resume_state(epoch, step_in_epoch)
                if args.eval_unseen_every and global_step % args.eval_unseen_every == 0:
                    run_unseen()

        micro.clear()
        opt.zero_grad(set_to_none=True)                       # nao vaza grad parcial p/ a proxima epoca
        if not args.smoke:
            v = do_val(epoch, f"epoca{epoch}")
            save_checkpoint(f"epoch{epoch}_val{v['loss']:.3f}".replace(".", "_"))
            ep_samples = samples_dir / f"checkpoint-epoch{epoch}"
            generate_samples(raw, tokenizer, ep_samples, f"ep{epoch}", seed0=2000 + epoch,
                             sample_refs=sample_refs)
            run_wer(ep_samples)
            save_resume_state(epoch + 1, 0)

    if args.smoke and global_step > 0:
        # o smoke tambem exercita val cross-ref, best/ e o estado de retomada
        do_val(0, "smoke")
        save_resume_state(0, step_in_epoch)
        run_unseen()                                           # exercita o caminho da avaliacao

    final_d = save_checkpoint("final")
    final_samples = samples_dir / "final"
    generate_samples(raw, tokenizer, final_samples, "final", seed0=9000, sample_refs=sample_refs)
    run_wer(final_samples)
    if not args.smoke and args.eval_unseen_every and global_step % args.eval_unseen_every != 0:
        run_unseen()
    if tb is not None:
        tb.close()
    dt_min = (time.time() - t_run) / 60
    peak_all = torch.cuda.max_memory_allocated() / 2**30
    (run_dir / "run_summary.json").write_text(json.dumps(
        {"steps": global_step, "minutes": round(dt_min, 1), "vram_peak_gb": round(peak_all, 2),
         "best_val": best_val if best_val < float("inf") else None,
         "clip_fraction": tot_clip / max(1, tot_steps), "skipped_events": n_skipped}, indent=1),
        encoding="utf-8")
    print(f"[FIM] steps={global_step} tempo={dt_min:.1f}min vrampico={peak_all:.2f}GB "
          f"clip={tot_clip}/{tot_steps} skips={n_skipped}")
    print(f"[ENTREGA] adapter final: {final_d} | melhor por val: {ckpt_dir / 'best'}")
    print(f"[AUDICAO] wavs em: {samples_dir}")


if __name__ == "__main__":
    main()
