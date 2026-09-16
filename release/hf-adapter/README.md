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

# Breeze TTS 2 — LoRA pt-BR (r=64, step 4000)

LoRA adapter (rank 64, rsLoRA, attention + MLP projections) trained on top of
[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) for **Brazilian
Portuguese** speech, including reference-conditioned voice cloning.

> **Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
> non-commercial use only.**

> **NOT PUBLISHED YET.** This is a placeholder model card: the adapter has **not**
> been uploaded to the Hugging Face Hub. It is planned to be released after
> additional training runs.

## Files

- `adapter_model.safetensors` — LoRA weights.
- `adapter_config.json` — PEFT config (r=64, alpha=64, rsLoRA, targets all).

## Usage (PEFT)

```python
from peft import PeftModel
from models.breeze import BreezeForConditionalGeneration

base = BreezeForConditionalGeneration.from_pretrained(
    "BreezeBlue/Breeze-TTS-2", dtype="bfloat16"
)
model = PeftModel.from_pretrained(base, "SEU_USUARIO/Breeze-TTS-2-lora-ptbr")
```

Or point the inference repo (`breeze-tts2-ptbr`) at this adapter via
`PTBR_ADAPTERS_DIR`.

## Training summary

- Base frozen; codec (Qwen3-TTS tokenizer, 24 kHz, 12.5 fps) frozen.
- LoRA r=64, alpha=64, rsLoRA, targets: q,k,v,o + gate,up,down (all stacks).
- ~148.3 M trainable params (4.10 %).
- 2 epochs (6,862 optimizer steps); released checkpoint at **step 4000**.
- Reference-conditioned training (`ref_edit_frac=0.9`), LR 3e-5.

See the training repo for the full pipeline, metrics and the article draft.

## Evaluation (reported in the paper)

Speaker-embedding cosine similarity to a held-out reference (ECAPA):

| Model | cos (short ref) | cos (long ref) |
|---|---|---|
| Base Breeze TTS 2 (frozen) | 0.729 | 0.678 |
| LoRA, reference-conditioned (this adapter) | 0.541 | 0.698 |

**Caveats:** single held-out speaker; the comparison against an earlier
reference-free run also changed the learning rate, so the effect is a joint
data + LR intervention. Treat the numbers as indicative, not conclusive.

## License and responsible use

- Model Materials and any Derivative Model are governed by the **BreezeBlue
  Research and Non-Commercial License Agreement**
  (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
  **No commercial use.**
- Do not use to clone a person's voice without explicit, legally sufficient
  consent, nor for deception, impersonation, or any prohibited use.
- No endorsement by BreezeBlue is implied.
