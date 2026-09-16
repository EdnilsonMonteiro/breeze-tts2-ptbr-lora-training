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

Amostras audiveis:
  training/runs/<run>/samples/checkpoint-<tag>/<NN>_<slug>.wav
  (a cada epoca e no fim; e a cada --sample-every-steps quando houver)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB
import prepare_dataset as PD

CB.GOLD_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CB.REPO))

import numpy as np
import torch

DEV = "cuda"

SAMPLE_TEXTS = [
    ("ola-pt", "Olá! Este é um teste de voz em português brasileiro."),
    ("numeros-pt", "O número da minha casa é quinze oh dois, no bairro Jardim Europa."),
    (
        "clima-pt",
        "A previsão do tempo indica pancadas de chuva à tarde, com temperaturas "
        "entre dezesseis e vinte e três graus.",
    ),
    (
        "siglas-pt",
        "Atenção: CPF um dois três ponto quatro cinco seis ponto sete oito nove, traço zero um.",
    ),
    ("afetivo-pt", "Que saudade daquele café quentinho da vovó no fim da tarde!"),
    ("regressao-en", "The weather today is sunny with a gentle breeze from the east."),
    ("placa-pt", "O carro de placa ABC um D vinte e três foi apreendido ontem à noite."),
    ("letras-pt", "As vogais são A, E, I, O, U; e as consoantes seguem o alfabeto."),
    ("siglas2-pt", "O IBGE e o INSS divulgaram os números na quinta-feira passada."),
    ("oov-pt", "O buzinaço assustou o gatíneo enquanto ele papeava na varanda."),
    ("trabalenguas-pt", "O rato roeu a roupa do rei de Roma e o mundo se admirou."),
]
SAMPLES_PER_EVENT = len(SAMPLE_TEXTS)

TARGET_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
}
DEFAULT_CORPUS_WEIGHTS = {"tagarela": 1.0, "cetuc": 1.0, "cml_pt": 2.0,
                          "podcast": 2.0, "tata": 2.0}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:40]


# ------------------------------------------------------------------ data


class TrainDataset(torch.utils.data.Dataset):
    """Itens do split de treino montados on-the-fly via cache de codes."""

    def __init__(self, idxs: list[str], tokenizer):
        self.tokenizer = tokenizer
        self.records = {r["idx"]: r for r in PD.load_meta()}
        self.idxs = [i for i in idxs if i in self.records]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[self.idxs[i]]
        variant = PD.deterministic_variant(rec["idx"], rec)
        ex = PD.build_item(self.tokenizer, rec, variant)
        return {
            "input_ids": ex["input_ids"][0],
            "attention_mask": torch.ones_like(ex["input_ids"][0]),
            "text_ids_mask": ex["text_ids_mask"][0],
            "text_ids_len": ex["text_ids_len"],
            "input_values": ex["input_values"].squeeze(0),
            "labels": ex["labels"][0],
        }


def load_split(name: str) -> list[str]:
    return [
        l.strip()
        for l in open(CB.TRAINING / f"splits_{name}.txt", encoding="utf-8")
        if l.strip()
    ]


def stratified_val_picks(ds_val: TrainDataset, n_items: int) -> list[tuple[int, str]]:
    """Amostra estratificada por corpus (proporcional ao tamanho), determinista."""
    by: dict[str, list[int]] = {}
    for i, idx in enumerate(ds_val.idxs):
        rec = ds_val.records.get(idx, {})
        by.setdefault(rec.get("corpus", "tata"), []).append(i)
    total = max(1, len(ds_val))
    picks: list[tuple[int, str]] = []
    for corp in sorted(by):
        lst = by[corp]
        k = max(1, round(n_items * len(lst) / total))
        step = max(1, len(lst) // k)
        for i in lst[::step][:k]:
            picks.append((i, corp))
    return picks


def quick_val_loss(raw, ds_val: TrainDataset, n_items: int = 96):
    """Val ESTRATIFICADO por corpus. Retorna (micro_global, {corpus: loss})."""
    picks = stratified_val_picks(ds_val, n_items)
    by_corp: dict[str, list[int]] = {}
    for i, c in picks:
        by_corp.setdefault(c, []).append(i)
    per: dict[str, float] = {}
    all_l: list[float] = []
    was_training = raw.training
    raw.eval()
    with torch.no_grad():
        for corp, lst in by_corp.items():
            ls = []
            for j in range(0, len(lst), 2):
                items = [ds_val[k] for k in lst[j:j + 2]]
                batch = CB.collate(items)
                batch = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                ls.append(raw(**batch).loss.item())
            per[corp] = float(np.mean(ls)) if ls else float("nan")
            all_l.extend(ls)
    if was_training:
        raw.train()
    overall = float(np.mean(all_l)) if all_l else float("nan")
    return overall, per


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


def generate_samples(raw, tokenizer, out_dir, tag: str, seed0: int = 1000):
    """Amostragem EAGER via BreezeForConditionalGeneration.generate(output_audio=True).

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
    template = get_template("tts_instruction")
    try:
        with torch.no_grad():  # somente no_grad: nao contamina o modelo
            for k, (name, text) in enumerate(SAMPLE_TEXTS[:SAMPLES_PER_EVENT]):
                request = {
                    "id": f"eval-{tag}-{k}",
                    "text": text,
                    "instruction": "Fale com clareza e naturalidade.",
                    "speaker": "S0",
                }
                set_all_seeds(seed0 + k)
                inputs = prepare_inputs(
                    tokenizer,
                    audio_tok,
                    raw,
                    [request],
                    template,
                    guidance_scale=1.0,
                    guidance_scale_ref=None,
                    guidance_scale_ins=None,
                )
                inputs.pop("input_values", None)  # prompt puramente textual
                inputs.pop("cfg_scale", None)
                path = out_dir / f"{k:02d}_{slug(name)}.wav"
                try:
                    res = raw.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        text_ids_mask=inputs["text_ids_mask"],
                        text_ids_len=inputs["text_ids_len"],
                        output_audio=True,
                        audio_tokenizer=audio_tok,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=0.9,
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
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    [samples] ERRO '{name}': {type(exc).__name__}: {str(exc)[:140]}"
                    )
    finally:
        n_fix = sanitize_inference_tensors(raw)
        del audio_tok
        torch.cuda.empty_cache()
        if was_training:
            raw.train()
        if n_fix:
            print(f"    [sanitize] {n_fix} buffers inference->normal restaurados")


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
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", choices=sorted(TARGET_PRESETS), default="attn",
                    help="attn = q/k/v/o | all = + gate/up/down (MLPs)")
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--ref-edit-frac", type=float, default=0.9,
                    help="fracao de exemplos em modo ref_edit (com referencia)")
    ap.add_argument("--val-items", type=int, default=96)
    ap.add_argument("--corpus-weights", type=str, default=None,
                    help="JSON {corpus: peso}; default = oversampling 24kHz+ 2x")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--no-wer", action="store_true",
                    help="desativa WER/CER automatico por checkpoint (faster-whisper CPU)")
    ap.add_argument("--sample-every-steps", type=int, default=500)
    ap.add_argument(
        "--resume-adapter",
        type=str,
        default=None,
        help="Caminho para pasta do adapter para continuar",
    )
    args = ap.parse_args()

    batch_size = args.batch or (2 if args.smoke else 4)
    grad_acc = args.grad_acc or (2 if args.smoke else 8)

    run_dir = CB.TRAINING / "runs" / args.run
    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for d in (run_dir, ckpt_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.csv"
    log_f = log_path.open("a", encoding="utf-8")
    if log_f.tell() == 0:
        log_f.write(
            "step,epoch,opt_step,total_loss,backbone_loss,depth_loss,"
            "val_loss,lr,vram_win_gb,vram_peak_gb,s_per_step\n"
        )

    print(
        f"[cfg] run={args.run} smoke={args.smoke} batch={batch_size} acc={grad_acc} "
        f"lr={args.lr} r={args.rank} a={args.alpha}"
    )

    # ---------------------------------------------------------- modelo base
    t0 = time.time()
    raw = CB.load_breeze_model(DEV)
    raw.gradient_checkpointing_enable()
    raw.enable_input_require_grads()
    print(f"[load] pronto em {time.time() - t0:.0f}s | gradient checkpointing ON")

    from peft import LoraConfig, get_peft_model

    if args.resume_adapter:
        from peft import PeftModel

        print(f"[resume] Carregando adapter existente de: {args.resume_adapter}")
        pm = PeftModel.from_pretrained(raw, args.resume_adapter, is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model

        lconf = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=TARGET_PRESETS[args.targets],
            use_rslora=args.use_rslora,
            exclude_modules=r".*codec_model.*",
        )
        pm = get_peft_model(raw, lconf)
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
    n_train = sum(stats.values())
    print(f"[G1] treinaveis por stack: {stats} | total={n_train:,}")
    (run_dir / "trainable_modules.txt").write_text(
        "\n".join(sorted(trainable_lines)), encoding="utf-8"
    )

    # ---------------------------------------------------------- dados
    tokenizer = CB.load_text_tokenizer()
    PD.set_ref_edit_frac(args.ref_edit_frac)
    ds = TrainDataset(load_split("train"), tokenizer)
    ds_val = TrainDataset(load_split("val"), tokenizer)
    from collections import Counter as _Counter
    _vdist = _Counter(PD.deterministic_variant(i, ds.records.get(i))
                      for i in ds.idxs[::max(1, len(ds.idxs) // 4000)])
    print(f"[data] ref_edit_frac={args.ref_edit_frac} | variantes (amostra)={dict(_vdist)}")
    cw = dict(DEFAULT_CORPUS_WEIGHTS)
    if args.corpus_weights:
        cw.update(json.loads(args.corpus_weights))
    weights = [float(cw.get(ds.records.get(i, {}).get("corpus", "tata"), 1.0))
               for i in ds.idxs]
    from collections import Counter as _Counter
    print(f"[data] corpus weights={cw} | "
          f"dist={dict(_Counter(ds.records.get(i, {}).get('corpus', 'tata') for i in ds.idxs))}")
    steps_per_epoch = math.ceil(len(ds) / (batch_size * grad_acc))
    opt_steps_total = args.steps if args.smoke else steps_per_epoch * args.epochs
    print(
        f"[data] itens={len(ds)} val={len(ds_val)} "
        f"steps/epoca={steps_per_epoch} total_alvo={opt_steps_total}"
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "batch_size": batch_size,
                "grad_acc": grad_acc,
                "n_train_items": len(ds),
                "opt_steps_total": opt_steps_total,
                "trainable_params": n_train,
                "per_stack": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    opt = torch.optim.AdamW(
        (p for p in pm.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-8,
    )

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, opt_steps_total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    trainable_params = [p for p in pm.parameters() if p.requires_grad]

    def save_checkpoint(tag_s: str):
        d = ckpt_dir / tag_s
        pm.save_pretrained(str(d), safe_serialization=True)
        print(f"[ckpt] salvo: {d}")
        return d

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
            wer = float(np.mean([r["wer"] for r in res]))
            cer = float(np.mean([r["cer"] for r in res]))
            print(f"[wer] {sample_dir.name}: WER={wer:.3f} CER={cer:.3f} "
                  f"({len(res)} amostras)")
            if tb is not None:
                tb.add_scalar("wer", wer, global_step)
                tb.add_scalar("cer", cer, global_step)
                tb.flush()
            (sample_dir / "wer.json").write_text(
                json.dumps({"wer": wer, "cer": cer, "per_sample": res},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[wer] falhou: {type(exc).__name__}: {str(exc)[:120]}")

    global_step = 0
    t_run = time.time()
    stop = False
    torch.cuda.reset_peak_memory_stats()

    epochs_to_run = 1 if args.smoke else args.epochs
    for epoch in range(epochs_to_run):
        if stop:
            break
        g = torch.Generator()
        g.manual_seed(42 + epoch)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(ds),
            replacement=True, generator=g)
        order = list(sampler)
        micro: list[dict] = []
        win = {"n_micro": 0, "loss": 0.0, "b": 0.0, "d": 0.0}
        win_peak0 = torch.cuda.max_memory_allocated()
        t_win = time.time()
        raw.train()
        for i in order:
            micro.append(ds[i])
            if len(micro) < batch_size:
                continue
            batch = CB.collate(micro)
            batch = {
                k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            micro.clear()
            out = raw(**batch)
            (out.loss / grad_acc).backward()
            win["n_micro"] += 1
            win["loss"] += out.loss.item()
            win["b"] += out.backbone_loss.item()
            win["d"] += out.depth_decoder_loss.item()
            if win["n_micro"] % grad_acc != 0:
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            mean_l = win["loss"] / win["n_micro"]
            if global_step <= 3 or global_step % 5 == 0:
                sp = (time.time() - t_win) / grad_acc
                win_peak = torch.cuda.max_memory_allocated()
                mean_b = win["b"] / win["n_micro"]
                mean_d = win["d"] / win["n_micro"]
                line = (
                    f"{global_step},{epoch},{global_step},{mean_l:.4f},"
                    f"{mean_b:.4f},{mean_d:.4f},nan,"
                    f"{sched.get_last_lr()[0]:.2e},"
                    f"{(win_peak - win_peak0) / 2**30:.2f},{win_peak / 2**30:.2f},{sp:.3f}\n"
                )
                log_f.write(line)
                log_f.flush()
                print(
                    f"[{global_step}/{opt_steps_total}] ep{epoch} loss={mean_l:.3f} "
                    f"(b{mean_b:.3f}/d{mean_d:.3f}) "
                    f"g={float(gnorm):.2f} vrwin={(win_peak - win_peak0) / 2**30:.2f}GB "
                    f"pk={win_peak / 2**30:.2f}GB {sp * 1000 / max(grad_acc, 1):.0f}ms/st"
                )
                if tb is not None:
                    tb.add_scalar("loss/total", mean_l, global_step)
                    tb.add_scalar("loss/backbone", mean_b, global_step)
                    tb.add_scalar("loss/depth", mean_d, global_step)
                    tb.add_scalar("lr", sched.get_last_lr()[0], global_step)
                    tb.add_scalar("grad_norm", float(gnorm), global_step)
                    tb.add_scalar("vram_peak_gb", win_peak / 2**30, global_step)
                    tb.flush()
                win = {"n_micro": 0, "loss": 0.0, "b": 0.0, "d": 0.0}
                win_peak0 = torch.cuda.max_memory_allocated()
                t_win = time.time()

            if args.smoke and global_step >= args.steps:
                stop = True
                break
            if (
                not args.smoke
                and global_step > 0
                and global_step % args.sample_every_steps == 0
            ):
                save_checkpoint(f"step{global_step}")
                generate_samples(
                    raw,
                    tokenizer,
                    samples_dir / f"checkpoint-{global_step}",
                    f"s{global_step}",
                    seed0=1000 + global_step,
                )

        if not args.smoke:
            vl, per_corpus = quick_val_loss(raw, ds_val, n_items=args.val_items)
            log_f.write(
                f"{global_step},{epoch},{global_step},nan,nan,nan,{vl:.4f},"
                f"nan,nan,{torch.cuda.max_memory_allocated() / 2**30:.2f},nan\n"
            )
            log_f.flush()
            msg = " ".join(f"{c}={v:.3f}" for c, v in sorted(per_corpus.items()))
            print(f"[epoca {epoch}] val_loss={vl:.4f} | {msg}")
            if tb is not None:
                tb.add_scalar("val/loss", vl, global_step)
                for c, v in per_corpus.items():
                    tb.add_scalar(f"val/{c}", v, global_step)
                tb.flush()
            save_checkpoint(f"epoch{epoch}_val{vl:.3f}".replace(".", "_"))
            ep_samples = samples_dir / f"checkpoint-epoch{epoch}"
            generate_samples(raw, tokenizer, ep_samples, f"ep{epoch}", seed0=2000 + epoch)
            run_wer(ep_samples)

    final_d = save_checkpoint("final")
    final_samples = samples_dir / "final"
    generate_samples(raw, tokenizer, final_samples, "final", seed0=9000)
    run_wer(final_samples)
    if tb is not None:
        tb.close()
    dt_min = (time.time() - t_run) / 60
    peak_all = torch.cuda.max_memory_allocated() / 2**30
    print(f"[FIM] steps={global_step} tempo={dt_min:.1f}min vrampico={peak_all:.2f}GB")
    print(f"[ENTREGA] adapter final: {final_d}")
    print(f"[AUDICAO] wavs em: {samples_dir}")


if __name__ == "__main__":
    main()
