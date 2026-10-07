import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptbr_lora" / "core"))
peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

import adapter_init as AI  # noqa: E402

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL = ATTN + ["gate_proj", "up_proj", "down_proj"]


class Layer(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.self_attn = nn.ModuleDict({n: nn.Linear(d, d) for n in ATTN})
        self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(d, 2 * d), "up_proj": nn.Linear(d, 2 * d),
                                  "down_proj": nn.Linear(2 * d, d)})

    def forward(self, x):
        a = sum(self.self_attn[n](x) for n in ATTN)
        h = torch.nn.functional.silu(self.mlp["gate_proj"](x + a)) * self.mlp["up_proj"](x + a)
        return x + a + self.mlp["down_proj"](h)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_model = nn.ModuleDict({"layers": nn.ModuleList([Layer(), Layer()])})
        self.text_encoder = nn.ModuleDict({"layers": nn.ModuleList([Layer()])})
        self.codec_model = nn.Linear(16, 16)

    def forward(self, x):
        for l in self.backbone_model["layers"]:
            x = l(x)
        for l in self.text_encoder["layers"]:
            x = l(x)
        return self.codec_model(x)


def _make(targets, r=4, alpha=8, seed=0):
    torch.manual_seed(seed)
    base = Tiny()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, exclude_modules=r".*codec_model.*")
    return get_peft_model(base, cfg)


def _train_a_bit(pm):
    opt = torch.optim.SGD([p for p in pm.parameters() if p.requires_grad], lr=0.05)
    x = torch.randn(8, 16)
    for _ in range(20):
        opt.zero_grad()
        pm(x).pow(2).mean().backward()
        opt.step()


def test_warm_start_output_identical_and_mlp_neutral(tmp_path):
    old = _make(ATTN)
    _train_a_bit(old)
    old.save_pretrained(tmp_path / "old")
    x = torch.randn(5, 16)
    old.eval()
    y_old = old(x).detach()

    new = _make(ALL, seed=0)          # mesma base (seed 0)
    AI.check_compat(tmp_path / "old", rank=4, alpha=8, use_rslora=False)
    rep = AI.load_into(new, tmp_path / "old")
    assert rep["loaded_tensors"] == 3 * 4 * 2            # 3 camadas x 4 projecoes de atencao x (A,B)
    assert rep["new_lora_tensors"] == 3 * 2 * 3          # gate/up/down x (A,B) x 3 camadas
    new.eval()
    assert torch.allclose(new(x).detach(), y_old, atol=1e-6)
    # os modulos novos treinam: depois de um passo, a saida muda
    _train_a_bit(new)
    assert not torch.allclose(new(x).detach(), y_old, atol=1e-4)


def test_incompatible_alpha_or_rank_is_rejected(tmp_path):
    old = _make(ATTN, r=4, alpha=8)
    old.save_pretrained(tmp_path / "old")
    with pytest.raises(ValueError):
        AI.check_compat(tmp_path / "old", rank=4, alpha=16, use_rslora=False)
    with pytest.raises(ValueError):
        AI.check_compat(tmp_path / "old", rank=8, alpha=8, use_rslora=False)


def test_adapter_with_modules_missing_in_new_model_fails(tmp_path):
    old = _make(ALL)
    old.save_pretrained(tmp_path / "old")
    new = _make(ATTN)             # o novo tem MENOS alvos que o adapter
    with pytest.raises(KeyError):
        AI.load_into(new, tmp_path / "old")


def test_shape_mismatch_fails(tmp_path):
    old = _make(ATTN, r=4, alpha=8)
    old.save_pretrained(tmp_path / "old")
    new = _make(ALL, r=8, alpha=8)
    with pytest.raises(ValueError):
        AI.load_into(new, tmp_path / "old")
