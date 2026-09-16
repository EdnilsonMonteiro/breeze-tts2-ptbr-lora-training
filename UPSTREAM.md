# Upstream

Este repositorio e um **fork** de [`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts)
que adiciona o codigo de treino/avaliacao do LoRA PT-BR (`ptbr_lora/`).

| Campo | Valor |
|---|---|
| Upstream | `https://github.com/breezeblue-ai/breeze-tts.git` |
| Commit base | `008f769016b0a24711becd7a4925030bc93f608c` (`main`) |
| Engenharia PT-BR | `ptbr_lora/` (este fork) |

O engine (codigo Apache-2.0) fica na raiz do fork e **nao foi alterado**; o
codigo do projeto vive em `ptbr_lora/`. Os pesos (Breeze-TTS-2) **nao** estao no
repositorio — baixe-os oficialmente (ver `README.md`). A licenca do modelo
(`BreezeBlue Research and Non-Commercial`) esta em
https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE (o upstream
removeu o arquivo `MODEL_LICENSE` do repositorio de codigo).

## Sincronizar com o upstream

```bash
git remote add upstream https://github.com/breezeblue-ai/breeze-tts.git
git fetch upstream
git rebase upstream/main        # ou merge, conforme preferencia
```

Ao rebasear, confira `ptbr_lora/core/common_breeze.py`: ele depende de APIs
internas do engine (`breeze_infer.templates._prepare_one`,
`_encode_prompt_audio`) que podem mudar entre versoes.
