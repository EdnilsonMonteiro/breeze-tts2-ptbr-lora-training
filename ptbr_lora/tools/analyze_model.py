"""FASE A - Analise somente-leitura do Breeze TTS 2 para planejamento LoRA.

Roda na venv: C:/IA/Breeze-tts/breeze-tts/venv/Scripts/python.exe analyze_model.py
- Introspeciona config.json do checkpoint.
- Instancia o modelo em meta device (sem alocar pesos) e mapeia modulos/shapes.
- Estima parametros treinaveis de LoRA para 3 variantes de target.
- Testa tokenizer de texto com frases PT-BR.
- Sonda Qwen3TTSTokenizer (CPU): sample rates + encode/decode smoke test.

Nao altera nenhum arquivo existente.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import paths  # noqa: E402
import common_breeze as CB  # noqa: E402

REPO = paths.BREEZE_REPO
CKPT = paths.CKPT
sys.path.insert(0, str(REPO))

SECTION = "===== %s ====="


def hr(title: str) -> None:
    print("\n" + SECTION % title)


def human(n: float) -> str:
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1024.0:
            return f"{n:8.2f}{unit}"
        n /= 1024.0
    return f"{n:8.2f}T"


# ------------------------------------------------------------------ 1. config
hr("1. CONFIG DIGEST")
cfg_raw = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
keys_top = [
    "model_type", "num_codebooks", "audio_vocab_size", "vocab_size",
    "text_vocab_size", "audio_token_id", "audio_eos_token_id",
    "codebook_eos_token_id", "codebook_pad_token_id",
    "depth_header_loss_weight", "tie_codebooks_embeddings",
    "tie_word_embeddings", "backbone_model_type", "text_encoder_proj_type",
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size", "dtype",
]
for k in keys_top:
    print(f"  {k}: {cfg_raw.get(k)}")
print("  depth_decoder_config.hidden/layers:",
      cfg_raw["depth_decoder_config"]["hidden_size"],
      cfg_raw["depth_decoder_config"]["num_hidden_layers"])
print("  text_encoder_config hidden/layers/type:",
      cfg_raw["text_encoder_config"]["hidden_size"],
      cfg_raw["text_encoder_config"]["num_hidden_layers"],
      cfg_raw["text_encoder_config"]["model_type"])
print("  text_encoder_lora_config (oficial, enabled=%s): r=%s alpha=%s targets=%s"
      % (cfg_raw["text_encoder_lora_config"]["enabled"],
         cfg_raw["text_encoder_lora_config"]["rank"],
         cfg_raw["text_encoder_lora_config"]["alpha"],
         cfg_raw["text_encoder_lora_config"]["target_modules"]))
print("  codec_config.model_type (IGNORAR p/ treino): %s sr=%s quantizers=%s"
      % (cfg_raw["codec_config"]["model_type"],
         cfg_raw["codec_config"]["sampling_rate"],
         cfg_raw["codec_config"]["num_quantizers"]))

hr("2. ARQUIVOS DO CHECKPOINT")
total_bytes = 0
for p in sorted(CKPT.rglob("*")):
    if p.is_file():
        total_bytes += p.stat().st_size
        if p.suffix in (".safetensors", ".json") and p.stat().st_size > 10_000_000:
            print(f"  {p.name:<44} {human(p.stat().st_size)}B")
print(f"  TOTAL checkpoint: {human(total_bytes)}B")

# ------------------------------------------------------- 3. modelo em meta
hr("3. INSTANCIACAO META DEVICE + MAPA DE MODULOS")
t0 = time.time()
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig

sys.path.insert(0, str(REPO))
from models.breeze import (  # noqa: E402
    BreezeForConditionalGeneration,
)
from models.breeze_config import (  # noqa: E402,F401
    BreezeConfig as RegisteredBreezeConfig,
)

config = AutoConfig.from_pretrained(CKPT)
with init_empty_weights():
    model = BreezeForConditionalGeneration._from_config(config)
print(f"  meta init OK em {time.time()-t0:.1f}s")

groups: dict[str, dict[str, object]] = {
    "codec_model": {"params": 0, "modules": 0},
    "text_encoder": {"params": 0, "modules": 0},
    "text_encoder_proj+embed_text_tokens": {"params": 0, "modules": 0},
    "backbone_model.embed_tokens": {"params": 0, "modules": 0},
    "backbone_model.layers(+norm)": {"params": 0, "modules": 0},
    "lm_head": {"params": 0, "modules": 0},
    "depth_decoder.model.layers(+norm)": {"params": 0, "modules": 0},
    "depth_decoder.model.embed_tokens": {"params": 0, "modules": 0},
    "depth_decoder.codebooks_head": {"params": 0, "modules": 0},
    "outros": {"params": 0, "modules": 0},
}


def group_of(name: str) -> str:
    if name.startswith("codec_model"):
        return "codec_model"
    if name.startswith("text_encoder."):
        return "text_encoder"
    if name.startswith(("text_encoder_proj.", "embed_text_tokens")):
        return "text_encoder_proj+embed_text_tokens"
    if name.startswith("backbone_model.embed_tokens"):
        return "backbone_model.embed_tokens"
    if name.startswith("backbone_model."):
        return "backbone_model.layers(+norm)"
    if name.startswith("depth_decoder.model.layers") or name.startswith(
        ("depth_decoder.model.norm", "depth_decoder.model.inputs_embeds_projector",
         "depth_decoder.model.backbone_hidden_state_projector")
    ):
        return "depth_decoder.model.layers(+norm)"
    if name.startswith("depth_decoder.model.embed_tokens"):
        return "depth_decoder.model.embed_tokens"
    if name.startswith("depth_decoder.codebooks_head"):
        return "depth_decoder.codebooks_head"
    if name.startswith("lm_head"):
        return "lm_head"
    return "outros"


lin_qkvo: list[tuple[str, int, int]] = []  # (module_name, in, out)
embed_shapes: list[tuple[str, tuple[int, ...]]] = []
for name, mod in model.named_modules():
    parts = name.split(".")
    if isinstance(mod, torch.nn.Linear):
        if len(parts) >= 2 and parts[-2] == "self_attn" and parts[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin_qkvo.append((name, mod.in_features, mod.out_features))
    elif isinstance(mod, torch.nn.Embedding):
        embed_shapes.append((name, tuple(mod.weight.shape)))

param_count = 0
group_params: dict[str, int] = {}
for pname, p in model.named_parameters():
    param_count += p.numel()
    g = group_of(pname)
    group_params[g] = group_params.get(g, 0) + p.numel()

print(f"  {'grupo':<40} {'params':>12}")
for g in groups:
    print(f"  {g:<40} {human(group_params.get(g, 0)):>12}")
print(f"  {'TOTAL':<40} {human(param_count):>12}")
print(f"  TOTAL bf16 ~= {human(param_count*2)}B")

hr("3b. ATENCOES q/k/v/o POR STACK (contagens)")
by_stack = Counter()
for name, _, _ in lin_qkvo:
    by_stack[name.split(".")[0]] += 1
for stack, c in sorted(by_stack.items()):
    per_type = Counter(n.split(".")[-1] for n, _, _ in lin_qkvo if n.split(".")[0] == stack)
    shapes = {(i, o) for s, i, o in lin_qkvo if s.split(".")[0] == stack}
    print(f"  {stack:<15} {c} Linears; por tipo {dict(per_type)}; shapes {sorted(shapes)}")
uniq_names_suffix = Counter(n.split(".")[-1] for n, _, _ in lin_qkvo)
print(f"  sufixos unicos: {dict(uniq_names_suffix)}  <- usados no target_modules PEFT")

hr("3c. EMBEDDINGS (para NAO tocar)")
for name, shape in embed_shapes:
    print(f"  {name:<55} {shape}")

hr("4. ESTIMATIVA LORA TREINAVEL (por variante)")
assert not hasattr(model, "_per_device_train")


def lora_estimate(target_pred, rank: int) -> tuple[int, int]:
    trainable = 0
    n_lin = 0
    for name, fin, fout in lin_qkvo:
        if target_pred(name):
            trainable += rank * (fin + fout)
            n_lin += 1
    return trainable, n_lin


variants = [
    ("V1 backbone+depth qkvo r=16",
     lambda n: not n.startswith("text_encoder"), 16),
    ("V2 V1 + text_encoder qkvo r=16",
     lambda n: True, 16),
    ("V3 V1 + text_encoder q,v,o r=8 (oficial-ish)",
     lambda n: not n.startswith("text_encoder")
     or n.split(".")[-1] in ("q_proj", "v_proj", "o_proj"), 8),
]
for label, pred, r in variants:
    tr, nl = lora_estimate(pred, r)
    print(f"  {label:<45} linears={nl:>4} params={human(tr):>9} "
          f"({100*tr/param_count:.2f}% do total)")

# ------------------------------------------------------- 5. tokenizer texto
hr("5. TOKENIZER TEXTO PT-BR (byte-level BPE)")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
frases = [
    "Ação, coração, pão e avô.",
    "A feijoada brasileira é um prato típico feito com feijão preto, carne-de-porco e temperos.",
    "[S0]<ins_bos>Fale com entusiasmo.<ins_eos>Olá! Tudo bem com você hoje?",
]
for frase in frases:
    ids = tok(frase, add_special_tokens=False)["input_ids"]
    dec = tok.decode(ids, skip_special_tokens=False)
    ratio = len(ids) / max(len(frase), 1)
    ok = "[OK]" if dec == frase else "[DIFF]"
    print(f"  {ok} chars={len(frase):>4} tokens={len(ids):>4} tok/char={ratio:.2f}")
    print(f"      dec: {dec[:90]}")
specials = ["[S0]", "<ins_bos>", "<ins_eos>", "<|AUDIO|>", "<|audio_eos|>"]
print("  tokens especiais -> id:")
for s in specials:
    tid = tok.convert_tokens_to_ids(s)
    print(f"      {s!r:<16} {tid}")
print(f"  pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id} "
      f"vocab={len(tok)} (<-> text_vocab_size 262158)")

# ------------------------------------------------------ 6. audio tokenizer
hr("6. AUDIO TOKENIZER (Qwen3TTSTokenizer, CPU) - smoke")
try:
    Qwen3TTSTokenizer = CB.import_qwen_tts()

    t0 = time.time()
    audio_tok = Qwen3TTSTokenizer.from_pretrained(str(CKPT / "audio_tokenizer"), device_map="cpu")
    print(f"  carregado em {time.time()-t0:.1f}s")
    print(f"  input_sr={audio_tok.get_input_sample_rate()} "
          f"output_sr={audio_tok.get_output_sample_rate()} "
          f"downsample={audio_tok.get_encode_downsample_rate()}")
    import numpy as np

    sr_in = audio_tok.get_input_sample_rate()
    wav = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr_in // 2) / sr_in)).astype(np.float32)
    ret = audio_tok.encode(wav, sr=sr_in)
    codes = np.asarray(ret["audio_codes"][0])
    print(f"  encode 0.5s@{sr_in}Hz -> audio_codes shape={codes.shape} "
          f"dtype={codes.dtype} min={codes.min()} max={codes.max()}")
    wavs, out_sr = audio_tok.decode({"audio_codes": [ret["audio_codes"]]})
    import numpy as _np

    arr = wavs[0] if isinstance(wavs, list) else wavs
    arr = _np.asarray(arr)
    print(f"  decode -> shape={arr.shape} sr={out_sr} "
          f"min={arr.min():.3f} max={arr.max():.3f}")
except Exception as exc:  # noqa: BLE001
    print(f"  FALHOU: {type(exc).__name__}: {exc}")

hr("FIM FASE A")
