"""Testes que exigem torch (pulados se ausente): gramatica de labels e politicas por variante."""
import pytest

torch = pytest.importorskip("torch")
CB = pytest.importorskip("common_breeze")

A, E = CB.AUDIO_TOKEN_ID, CB.AUDIO_EOS_TOKEN_ID


def _ex(ids):
    return {"input_ids": torch.tensor([ids])}


def test_labels_train_only_block():
    ex = _ex([5, 5, 5, A, A, A, E])
    lab = CB.make_labels(ex, frame_policies=CB.POLICIES_BY_VARIANT["tts_instruction"])[0].tolist()
    assert lab == [-100, -100, -100, A, A, A, E]


def test_labels_ref_block_is_backbone_only():
    ex = _ex([5, 5, A, A, E, 5, 5, A, A, A, E])
    lab = CB.make_labels(ex, frame_policies=CB.POLICIES_BY_VARIANT["ref_edit_tata"])[0].tolist()
    assert lab == [-100, -100, -101, -101, E, -100, -100, A, A, A, E]


def test_all_variants_have_policies_matching_block_count():
    assert set(CB.POLICIES_BY_VARIANT) >= {"tts_plain", "tts_instruction", "ref_clone_tata",
                                           "ref_edit_tata", "ref_edit_auto"}
    assert len(CB.POLICIES_BY_VARIANT["ref_clone_tata"]) == 2
    assert len(CB.POLICIES_BY_VARIANT["tts_plain"]) == 1


def test_default_instruction_is_in_pool():
    assert CB.DEFAULT_INSTRUCTION in CB.INSTRUCTION_POOL
