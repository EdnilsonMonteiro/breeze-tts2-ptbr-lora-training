"""common_breeze.py — modulo compartilhado das fases B/C do treino LoRA PT-BR.

Fonte unica de verdade para:
- caminhos do projeto;
- pool de instrucoes PT-BR;
- construcao de exemplos de treino no formato EXATO do forward() do Breeze
  (reutiliza breeze_infer.templates._prepare_one verbatim, com patch de cache
  de codes pre-extraidos para nao recodificar audio a cada acesso);
- montagem de labels (marcadores 262144 / -101 / 262145 / -100);
- collate com left-padding replicando _collate_inputs oficial (+ labels).

Nao altera nenhum arquivo do repo breeze-tts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ paths
# Caminhos vem de ptbr_lora/core/paths.py (dirigido por PTBR_ARTIFACTS / .env).
# O engine (breeze-tts) esta na raiz deste fork; os artefatos ficam fora do git.
ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import paths  # noqa: E402

REPO = paths.BREEZE_REPO
CKPT = paths.CKPT
DATASET = paths.DATASET
TRAINING = paths.TRAINING
TOKENS_DIR = paths.TOKENS_DIR
WAVS24_DIR = paths.WAVS24_DIR
GOLD_DIR = paths.GOLD_DIR

TEXTS_CSV = paths.TEXTS_CSV

# ------------------------------------------------------------- multi-corpus
# Registro de corpora (datasets/corpora.json). Retrocompativel: sem o arquivo,
# cai no par legado (tata + podcast) dirigido por BREEZE_DATASET_DIR.
DATASETS_ROOT = paths.DATASETS_ROOT
CORPORA_JSON = paths.CORPORA_JSON
SCRAPING_WORK = paths.SCRAPING_WORK

_LEGACY_CORPORA = [
    {"name": "tata", "root": "TTS-Portuguese-Corpus", "csv": "texts.csv"},
    {"name": "podcast", "root": "podcast", "csv": "texts.csv"},
]


def load_corpora() -> list[dict]:
    """Lista de corpora ativos. Cada item: {name, root, csv[, speakers]}."""
    import json

    if CORPORA_JSON.exists():
        data = json.loads(CORPORA_JSON.read_text(encoding="utf-8"))
        return [c for c in data if c.get("enabled", True)]
    return list(_LEGACY_CORPORA)


def corpus_dir(corp: dict) -> Path:
    return DATASETS_ROOT / corp["root"]


def corpus_csv(corp: dict) -> Path:
    return corpus_dir(corp) / corp.get("csv", "texts.csv")
SR = 24_000                    # sample rate alvo do modelo
MAX_DUR_S = 10.2               # descarta clips acima disso (integridade texto-audio)
MIN_DUR_S = 0.98
PEAK_NORM = 0.95

sys.path.insert(0, str(REPO))

AUDIO_TOKEN_ID = 262144        # <|AUDIO|>       (marcador de frame)
AUDIO_EOS_TOKEN_ID = 262145    # <|audio_eos|>
BACKBONE_EOS_CLASS = None      # resolvido via config (vocab_size = 2051)
IGNORE_IDX = -100              # ignorado por backbone e depth decoder
DEPTH_IGNORE = -101            # frame usado SOMENTE pelo backbone (codebook 0)

INSTRUCTION_POOL = [
    "Fale de forma clara e natural.",
    "Leia o texto com dicao clara e ritmo calmo.",
    "Narre com voz neutra e naturalidade.",
    "Leia em voz alta com pronuncia cuidada.",
    "Fale com tom tranquilo e pausado.",
    "Leia o texto com entonacao de leitura informativa.",
    "Produza uma leitura limpa e expressiva.",
    "Fale com naturalidade, como um narrador brasileiro.",
]

# ------------------------------------------------------------------ loaders


def load_text_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


def load_audio_tokenizer(device: str = "cuda"):
    from qwen_tts import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer.from_pretrained(str(CKPT / "audio_tokenizer"), device_map=device)


def load_breeze_model(device: str = "cuda", attn: str = "eager"):
    from models.breeze import BreezeForConditionalGeneration

    model = BreezeForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation=attn
    )
    return model.to(device).eval()


# --------------------------------------------------------- patch de cache de codes
#
# templates._prepare_one -> _resolve_segment_audio_codes -> _encode_prompt_audio.
# Patch substitui APENAS a funcao final de encode dentro do namespace de
# templates; quando o path esta no cache devolvemos o tensor salvo na Fase B,
# caso contrario caimos no encode original (usado pela checagem de paridade).

_cache_store: dict[str, torch.Tensor] = {}
_orig_encode_prompt_audio = None


def enable_code_cache() -> None:
    """Ativa o cache global (registro manual com register_codes())."""
    global _orig_encode_prompt_audio
    import breeze_infer.templates as T

    if _orig_encode_prompt_audio is not None:
        return

    def _patched(audio_tokenizer, audio_path):
        key = str(audio_path)
        hit = _cache_store.get(key)
        if hit is not None:
            return hit.clone()
        return _orig_encode_prompt_audio(audio_tokenizer, audio_path)

    _orig_encode_prompt_audio = T._encode_prompt_audio
    T._encode_prompt_audio = _patched


def disable_code_cache() -> None:
    global _orig_encode_prompt_audio
    if _orig_encode_prompt_audio is None:
        return
    import breeze_infer.templates as T

    T._encode_prompt_audio = _orig_encode_prompt_audio
    _orig_encode_prompt_audio = None


def register_codes(wav_path: str | Path, codes_npz_path: str | Path) -> int:
    """Carrega codes (.npz) no cache sob a chave wav_path. Retorna n_frames."""
    enable_code_cache()
    arr = np.load(codes_npz_path)["codes"]
    assert arr.ndim == 2 and arr.shape[1] == 16 and arr.dtype == np.int16
    _cache_store[str(Path(wav_path))] = torch.from_numpy(np.ascontiguousarray(arr))
    return int(arr.shape[0])


class ConfigStub:
    """Substituto leve de model.config p/ _prepare_one (usa so num_codebooks)."""

    def __init__(self, num_codebooks: int = 16):
        self.num_codebooks = num_codebooks


@dataclass
class ExampleRequest:
    variant: str                 # 'tts_instruction' | 'ref_edit_tata'
    text: str                    # texto-alvo (transcricao)
    instruction: str
    ref_text: str = ""           # igual ao text no variante clone (transcricao do ref)
    ref_audio_path: str = ""     # wav 24 kHz processado (chave do cache)
    target_audio_path: str = ""  # wav da FALA SUPERVISORIA (o mesmo clip aqui)


def build_segments(req: ExampleRequest, include_target: bool = True) -> list[dict]:
    """Segmentos de treino = template oficial + BLOCO DE AUDIO ALVO anexado.

    Em inferencia os templates oficiais terminam no texto (o audio e GERADO).
    Em treino (estilo CSM) a sequencia precisa carregar o audio supervisionado:
      tts_instruction -> [texto]               + [audio-alvo]
      ref_edit_tata   -> [texto-ref, audio-ref,
                          texto]               + [audio-alvo]
    include_target=False devolve o template puro (usado na checagem de paridade).
    """
    from breeze_infer import templates as T

    r = {
        "speaker": "S0",
        "text": req.text,
        "instruction": req.instruction,
    }
    if req.variant == "tts_instruction":
        segments = T._tts_instruction_segments(r)
    elif req.variant == "ref_edit_tata":
        r["ref_audio_path"] = req.ref_audio_path
        r["ref_text"] = req.ref_text
        segments = T._ref_edit_tata_segments(r)
    else:
        raise ValueError(f"variant desconhecido: {req.variant}")

    if include_target:
        seg_path = req.target_audio_path or req.ref_audio_path
        if not seg_path:
            raise ValueError("bloco alvo exige target_audio_path/ref_audio_path")
        segments.append({"type": "audio", "append_eos": True, "drop_last_frame": False,
                         "audio_path": seg_path})
    return segments


def build_example(tokenizer, req: ExampleRequest) -> dict[str, torch.Tensor]:
    """Monta input_ids/masks/input_values exatamente como prepare_inputs oficial.

    Requer register_codes() do ref_audio_path quando variant=ref_edit_tata.
    Retorna tensores single-sample (batch dim 1).
    """
    from breeze_infer.templates import _prepare_one

    segments = build_segments(req)
    out = _prepare_one(tokenizer, None, ConfigStub(16), segments)
    out["input_values"] = out["audio_tokens"]          # alias claro
    return out


# ------------------------------------------------------------------ labels


def make_labels(example: dict, *, frame_policies: list[str]) -> torch.Tensor:
    """Constroi labels (1,S) para o treino LoRA.

    Gramatica VALIDADA EMPERICAMENTE no smoke (smoke_forward_b.py):
    - O lm_head do backbone tem SOMENTE 2052 classes (codes 0..2050 + EOS class 2051);
      ids de texto (262158) sao INVALIDOS como target -> TODO texto fica -100 e
      serve apenas de CONTEXTO (padrao codec-LM/CSM).
    - frames 'train'         -> 262144 (expansao interna usa os codes reais nos 16 slots);
    - frames 'backbone_only' -> -101   (codebook 0 treina no backbone; depth ignora 1..15);
    - posicao <|audio_eos|>  -> 262145 (slot0 vira EOS-class 2051, depth ignorado).
    """
    input_ids = example["input_ids"][0]
    labels = torch.full_like(input_ids, IGNORE_IDX)

    frame_positions = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    eos_positions = (input_ids == AUDIO_EOS_TOKEN_ID).nonzero(as_tuple=True)[0]

    counts = count_frames_per_block(example)
    assert len(counts) == len(frame_policies), (
        f"blocos de audio={len(counts)} != policies={len(frame_policies)}"
    )
    assert sum(counts) == len(frame_positions), (
        f"desalinhamento frames: ids={len(frame_positions)} vs blocos={sum(counts)}"
    )
    idx = 0
    for cnt, pol in zip(counts, frame_policies):
        assert pol in ("train", "backbone_only"), pol
        value = AUDIO_TOKEN_ID if pol == "train" else DEPTH_IGNORE
        for _ in range(cnt):
            labels[frame_positions[idx]] = value
            idx += 1
    for p in eos_positions:
        labels[p] = AUDIO_EOS_TOKEN_ID
    return labels.unsqueeze(0)


def count_frames_per_block(example: dict) -> list[int]:
    """Conta '<|AUDIO|>' consecutivos por bloco (separados por <|audio_eos|>)."""
    ids = example["input_ids"][0].tolist()
    blocks: list[int] = []
    cur = 0
    for t in ids:
        if t == AUDIO_TOKEN_ID:
            cur += 1
        elif t == AUDIO_EOS_TOKEN_ID:
            blocks.append(cur)
            cur = 0
    return blocks


TTS_INSTRUCTION_POLICIES = ["train"]
REF_EDIT_TATA_POLICIES = ["backbone_only", "train"]

POLICIES_BY_VARIANT = {
    "tts_instruction": TTS_INSTRUCTION_POLICIES,
    "ref_edit_tata": REF_EDIT_TATA_POLICIES,
    # ref_edit_auto usa o MESMO template/policies de ref_edit (ref = outro clipe
    # do MESMO locutor; internamente cai no template ref_edit_tata)
    "ref_edit_auto": REF_EDIT_TATA_POLICIES,
}


# ------------------------------------------------------------------ collate


def pad_left(t: torch.Tensor, pad_len: int, value) -> torch.Tensor:
    return F.pad(t, (pad_len, 0), value=value) if pad_len > 0 else t


def collate(items: list[dict], pad_token_id: int = 0):
    """Replica _collate_inputs oficial (left-pad) + labels (-100 no padding).

    items: lista de dicts single-sample vindos do Dataset:
      input_ids (L,), attention_mask (L,), text_ids_mask (L,), text_ids_len (nseg,),
      input_values (F,16), labels (L,)
    """
    max_len = max(it["input_ids"].shape[-1] for it in items)

    ids_l, att_l, mask_l, lab_l, tl_all = [], [], [], [], []
    for it in items:
        L = it["input_ids"].shape[-1]
        pad_len = max_len - L
        ids_l.append(pad_left(it["input_ids"].view(1, -1), pad_len, pad_token_id))
        att_l.append(pad_left(it["attention_mask"].view(1, -1), pad_len, 0))
        mask_l.append(pad_left(it["text_ids_mask"].view(1, -1), pad_len, False))
        lab_l.append(pad_left(it["labels"].view(1, -1), pad_len, IGNORE_IDX))
        tl_all.append(it["text_ids_len"])

    _ivs = []
    for it in items:
        iv = it["input_values"]
        if iv.dim() == 2:
            iv = iv.unsqueeze(0)
        _ivs.append(iv)
    input_values = torch.cat(_ivs, dim=1)  # (1, F_total, 16) sem pad

    batch = {
        "input_ids": torch.cat(ids_l, dim=0),
        "attention_mask": torch.cat(att_l, dim=0),
        "text_ids_mask": torch.cat(mask_l, dim=0),
        "labels": torch.cat(lab_l, dim=0),
        "text_ids_len": torch.cat(tl_all, dim=0),
        "input_values": input_values.long(),
    }
    return batch


# ------------------------------------------------------------------ dataset item


def instruction_for(idx: int) -> str:
    return INSTRUCTION_POOL[idx % len(INSTRUCTION_POOL)]
