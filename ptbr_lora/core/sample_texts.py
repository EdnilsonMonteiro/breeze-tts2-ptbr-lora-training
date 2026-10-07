"""sample_texts.py — frases de avaliacao (fonte unica: treino, eval_wer, gerar_amostras). Puro."""
from __future__ import annotations

# (nome, texto). Sufixo "-en" => texto em ingles (WER sem normalizacao pt, reportado a parte).
# Os textos passam por `text_norm` antes de gerar e ANTES de medir (ref e hyp).
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
    ("digitos-pt", "Foram vendidos 1.234 ingressos a R$ 25,50 cada, e o show começou às 23h30."),
    (
        "longo-pt",
        "Ontem à noite, depois de muito tempo sem se falarem, os dois amigos se encontraram na "
        "praça da cidade, conversaram sobre a infância, riram das velhas histórias e prometeram "
        "que não deixariam passar tanto tempo até o próximo encontro.",
    ),
]
