# Documentação — PT-BR LoRA (treino)

Documentação voltada ao **usuário**: como instalar, configurar e rodar o
pipeline de treino/avaliação do adaptador LoRA pt-BR sobre o Breeze TTS 2.

| Documento | Conteúdo |
|---|---|
| [INSTALL.md](INSTALL.md) | ambiente, dependências, checkpoint base |
| [CONFIGURATION.md](CONFIGURATION.md) | `PTBR_ARTIFACTS` e todas as variáveis de ambiente |
| [DATASETS.md](DATASETS.md) | download de corpora, ingestão e `prepare_dataset.py` |
| [TRAINING.md](TRAINING.md) | `train_lora.py`, `auto_train.py` e opções de LoRA |
| [EVALUATION.md](EVALUATION.md) | val loss, WER/CER e similaridade de locutor |

Visão geral do repositório: [`../README.md`](../README.md).

## Fluxo típico

```
[INSTALL] → [CONFIGURATION] → [DATASETS: process + finalize]
          → [TRAINING: smoke + full] → [EVALUATION] → adapter em training/runs/<run>/checkpoints/
```

Ao final, o adapter (pasta com `adapter_config.json` + `adapter_model.safetensors`)
pode ser usado no repo de inferência [`breeze-tts2-ptbr`](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr)
ou publicado no Hugging Face.

> **Licença:** pesos e derivados do Breeze TTS 2 são
> *research/non-commercial* (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
> Veja `../NOTICE`.
