---
language:
  - pt
license: other
license_name: breeze-tts-2-research-non-commercial
base_model: BreezeBlue/Breeze-TTS-2
library_name: peft
tags:
  - text-to-speech
  - lora
  - peft
  - brazilian-portuguese
  - voice-cloning
---

# Breeze TTS 2 — PT-BR LoRA (r=64, α=256, targets=all)

LoRA adapter for **Brazilian Portuguese** speech on top of
[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), including
reference-conditioned voice cloning. Trained on a ~204 h multi-corpus pt-BR dataset
with a single 16 GB consumer GPU.

> **Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
> non-commercial use only.**

> **NOT PUBLISHED YET.** This is a placeholder model card: the adapter has **not**
> been uploaded to the Hugging Face Hub. The best run so far is `r72_01`; release is
> planned after the final checks below.

---

## TL;DR

The single most important design choice was **`targets=all`** — applying LoRA to the
attention projections **and the MLP (feed-forward) projections** — not attention
alone. It is what gives the adapter enough capacity to learn Portuguese phonology and
prosody, which the EN/ZH base model does not have. Everything else (classic LoRA
scale, LR floor, canonical speaker references) makes that adaptation *stable*.

---

## Why `targets=all` (attention **and** MLP) is the key choice

### 1. Attention-only LoRA cannot rewrite the FFN — and language lives in the FFN

LoRA (Hu et al., 2021) freezes a weight `W` and adds a low-rank update
`ΔW = B·A`, `rank(B·A) ≤ r`. If a matrix is **not** targeted, its update is exactly
`0`. So an attention-only adapter leaves every feed-forward network untouched.

The FFN is where a transformer stores and writes *features*: Geva et al. (2021,
*Transformer Feed-Forward Layers Are Key-Value Memories*) show the FFN acts as a
key–value memory — hidden states match keys, and the value vectors write content into
the residual stream. For each layer,

```
FFN(x) = W2 · σ(W1 · x)
```

Adapting to a new **phonology** means rewriting that `feature → token` map (which
vowel quality, nasalization, /r/, /lh/, /nh/, "ão" the model should emit). Attention
merely *mixes* value vectors (a content-dependent convex re-weighting of context); it
cannot construct new nonlinear features. With attention-only LoRA the representable
update subspace for `W1` and `W2` is `{0}`, so the model can at best re-weight the
in-context reference — not learn the mapping into the pt-BR codebooks.

### 2. Capacity scales with the number of adapted matrices

Breeze TTS 2 is a codec language model (`backbone` → codebook 0; `depth decoder` →
codebooks 1..15; `text encoder` → semantics). The codec itself stays frozen. Our two
runs are **identical except for `--targets`**:

| `--targets` | adapted modules | trainable params | backbone | depth | text-encoder |
|---|---|---|---|---|---|
| `attn` | q, k, v, o | 42.7 M | 25.7 M | 5.1 M | 11.9 M |
| **`all`** | q, k, v, o **+ gate, up, down** | **148.3 M** | 69.7 M | 26.3 M | **52.2 M** |

The biggest jump is in the **text encoder** (T5Gemma2): its MLPs are where the
`Portuguese text → semantics` map is learned. Attention-only freezes ~40 M of those
FFN parameters. Roughly a 3.5× increase in trainable capacity, concentrated exactly
where the missing language knowledge has to be written.

### 3. The literature agrees

- **Hu et al. (2021), the LoRA paper**, report that adapting **more weight types**
  (adding MLP projections on top of attention) improves quality, and recommend
  applying LoRA to *all* linear layers rather than a subset.
- **Biderman et al. (2024), *LoRA Learns Less and Forgets Less***, reach the same
  conclusion for knowledge/domain adaptation: the FFN/MLP blocks carry the knowledge,
  and attention-only adaptation underfits it.
- **Cross-lingual adapters** (Pfeiffer et al., 2020, *MAD-X*) place *language*
  adapters inside the FFN — language-specific structure lives there.

### 4. Our controlled result

Same data, same schedule, same scale, same LR; **only `--targets` differs**
(`--rank 64 --alpha 256`, classic scale `α/r = 4.0`, `lr 3e-5` with a 15 % floor,
`--ref-edit-frac 1.0`). WER is measured on the **actual cloning clips**, text
normalized, with ASR-confounded sentences removed (see *Evaluation* below):

| run | `--targets` | WER ↓ (9 sentences) | CER ↓ |
|---|---|---|---|
| `r71_01` | `attn` | 0.167 | 0.065 |
| **`r72_01`** | **`all`** | **0.117** | **0.056** |
| frozen base (no LoRA) | — | 0.908 | 0.568 |

The `all` run wins on the metric **and** by ear (best Portuguese and best voice). The
frozen base is near-unintelligible (`0.908`; on one sentence it hallucinates
unrelated speech), so the metric does discriminate real quality.

---

## The rest of the recipe (what makes `targets=all` stable)

| Change | v1 (`r64_*`) | v2 (`r72_01`) | Why |
|---|---|---|---|
| LoRA scale | `rsLoRA` `α/√r` = 8.0 | classic `α/r` = **4.0** | 8.0 over-steers; lower training-time scale keeps prosody intact |
| LR schedule | one cosine → ~0 | **floor 15 %** + 3 cosine cycles | a collapsing LR froze the depth decoder (fine acoustics) and degraded Portuguese |
| LR | 3e-5 / 1e-5 | **3e-5** | 1e-5 was too weak; the depth loss never recovered |
| Reference data | `ref_edit_frac 0.9`, `REF_MIN_COS 0.55`, self-ref 2.9 % | `1.0`, **0.70**, **0 %** self-ref, **canonical reference per speaker** | clean identity signal: "copy *this voice*", not "copy this channel" |
| VRAM | `batch 4 × acc 8` | `batch 2 × acc 16` | same effective batch, half the peak → no OOM |

> **Note on inference-time adapter scale.** In the runtime, the adapter scale is a
> **multiplier** (1.0 = as trained). Empirically, values `< 1.0` make things *worse*,
> i.e. this adapter is **not** over-steering — the gain came from training at a
> well-behaved scale, not from shrinking the output.

---

## Evaluation (and a metric caveat worth reading)

Two traps were found while comparing runs — they are why early metrics pointed at the
wrong model:

1. **WER must be measured on the cloning clips, not on reference-free samples.**
   The training-time WER was computed on reference-free "language test" samples, and
   it was inflated by ASR **number/acronym formatting** (ref "quinze oh dois" → hyp
   "1502"; "CPF um dois três" → "CPF123.456.789-01"). Measuring instead on the
   cloning clips, with text normalization and the two ASR-confounded sentences
   removed, gives the table above.

2. **Speaker similarity (SECS/ECAPA) favors the *base* model here.** Measured with
   the same tooling and an other-speaker control (CETUC = 0.27 / 0.05, so the metric
   is not saturated):

   | model | median SECS | note |
   |---|---|---|
   | two real recordings of the same speaker (ceiling) | 0.825 | — |
   | **frozen base (no pt-BR LoRA)** | **0.685** | no Portuguese training at all |
   | `r70_01` (attn, lr 1e-5) | 0.725 | best SECS, **worst** Portuguese |
   | `r71_01` (attn) | 0.651 | |
   | `r72_01` (all) | 0.614 | **best Portuguese and voice by ear** |

   ECAPA tracks **timbre/channel**, not pronunciation. The base "clones" reasonably
   *in-context* by copying the reference's channel (our reference is 18 s / 48 kHz,
   out of the training distribution). The more the adapter learns pt-BR, the further
   it moves from that channel and the lower SECS reads — **without sounding worse.**
   **Conclusion: never rank models by SECS alone; read it only relative to the base
   and always with an other-speaker control.**

**Takeaway:** pick models by WER on cloning clips (normalized, confounds removed)
plus blind listening, and report SECS only as a base-relative control.

---

## Usage (PEFT)

```python
from peft import PeftModel
from models.breeze import BreezeForConditionalGeneration

base = BreezeForConditionalGeneration.from_pretrained(
    "BreezeBlue/Breeze-TTS-2", dtype="bfloat16"
)
model = PeftModel.from_pretrained(base, "SEU_USUARIO/Breeze-TTS-2-lora-ptbr")
```

Or point the inference repo (`breeze-tts2-ptbr`) at this adapter via `PTBR_ADAPTERS_DIR`.

## Training summary

- Base frozen; codec (Qwen3-TTS tokenizer, 24 kHz, 12.5 fps) frozen.
- LoRA `r=64, alpha=256` (classic scale 4.0, **no rsLoRA**), targets: q,k,v,o **+ gate,up,down** across all three stacks.
- **148,258,816** trainable params.
- 2 epochs (~7,018 optimizer steps); `batch 2 × grad-acc 16`.
- Reference-conditioned training (`ref_edit_frac=1.0`), LR `3e-5` with a 15 % floor and 3 cosine cycles.
- Dataset: 121,975 clips / 204.42 h, speaker-disjoint splits (train 112,303 / val 5,216 / test 4,456).

See the training repo for the full pipeline, the corrected evaluation protocol and the
per-axis numbers: [**breeze-tts2-ptbr-lora-training**](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training)
(`docs/ESTRATEGIA-PTBR.md`).

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Geva et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP.
- Biderman et al. (2024). *LoRA Learns Less and Forgets Less.* arXiv:2405.09673.
- Pfeiffer et al. (2020). *MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer.* EMNLP.

## License and responsible use

- Model Materials and any Derivative Model are governed by the **BreezeBlue Research
  and Non-Commercial License Agreement**
  (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). **No commercial use.**
- Do not clone a person's voice without explicit, legally sufficient consent, nor for
  deception, impersonation, or any prohibited use.
- No endorsement by BreezeBlue is implied.
