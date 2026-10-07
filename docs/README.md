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
| [ESTRATEGIA-PTBR.md](ESTRATEGIA-PTBR.md) | receita v2: o que mudou, resultados (protocolo antigo — históricos) |
| [AUDITORIA-2026-09.md](AUDITORIA-2026-09.md) | **auditoria 29/09/2026**: achados e protocolo v3 |
| [SPEAKER_PURITY.md](SPEAKER_PURITY.md) | checagem de pureza dos rótulos de locutor antes do treino |

Visão geral do repositório: [`../README.md`](../README.md).

## Fluxo típico

```
[INSTALL] → [CONFIGURATION] → [DATASETS: process + finalize]
          → [TRAINING: smoke + full] → [EVALUATION] → adapter em training/runs/<run>/checkpoints/
```

Ao final, o adapter (pasta com `adapter_config.json` + `adapter_model.safetensors`)
pode ser usado no repo de inferência [`breeze-tts2-ptbr`](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr)
ou publicado no Hugging Face. O adaptador liberado (r76, passo 1500) está em
[`EdnilsonMonts/Breeze-tts-2-brazillian-lora`](https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora).

> **Licença:** pesos e derivados do Breeze TTS 2 são
> *research/non-commercial* (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
> Veja `../NOTICE`.
