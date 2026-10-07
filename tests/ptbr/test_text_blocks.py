from text_blocks import blocks_for_seconds, dur_ok, split_blocks, split_sentences


def test_abbreviations_do_not_split():
    s = split_sentences("O Sr. Silva chegou. Falou com a Dra. Ana.")
    assert s == ["O Sr. Silva chegou.", "Falou com a Dra. Ana."]


def test_blocks_respect_budget_and_keep_text():
    text = ("Uma frase curta aqui. " * 3 + "Outra frase bem mais longa, com virgulas, "
            "que precisa ser fatiada porque excede o orcamento de palavras do bloco atual. ") * 3
    blocks = split_blocks(text, 12)
    assert all(len(b.split()) <= int(12 * 1.15) + 12 for b in blocks)
    assert " ".join(" ".join(blocks).split()) == " ".join(text.split())


def test_short_blocks_merged():
    blocks = split_blocks("Oi. Tudo bem? Sim. Vamos embora agora mesmo daqui.", 6, min_words=5)
    assert all(len(b.split()) >= 5 for b in blocks)


def test_blocks_for_seconds_and_duration_gate():
    assert len(blocks_for_seconds("palavra " * 100, 10.0)) >= 3
    assert dur_ok(4.0, 10)
    assert not dur_ok(13.0, 3)      # 3 palavras nao duram 13 s
