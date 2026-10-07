import json
import sys
from pathlib import Path

import numpy as np
import pytest

import purity as P
import speaker_identity as SI
import splits as SPL

PTBR = Path(__file__).resolve().parents[2] / "ptbr_lora"


def _voice(seed, n, d=64, noise=0.12, rng=None):
    rng = rng or np.random.default_rng(seed)
    base = np.random.default_rng(1000 + seed).normal(size=d)
    return P.l2n(base + noise * np.linalg.norm(base) / np.sqrt(d) * rng.normal(size=(n, d)))


def test_edge_bleed():
    rttm = [(0.0, 5.0, "A"), (5.1, 9.0, "B")]
    assert not SI.edge_bleed([[1.0, 4.8]], 0.175, "A", rttm)          # margem cai no silencio/proprio
    assert SI.edge_bleed([[1.0, 5.0]], 0.175, "A", rttm)              # margem final pega B (5.1-5.175)
    assert not SI.edge_bleed([[5.3, 8.0]], 0.175, "B", rttm)


def test_refine_label_clean_split_and_intruder():
    cfg = P.PurityConfig()
    E = _voice(1, 20)
    r = SI.refine_label("ep:S0", [f"c{i}" for i in range(20)], E, cfg)
    assert list(r.kept) == ["ep:S0"] and len(r.kept["ep:S0"]) == 20 and not r.dropped

    E = np.concatenate([_voice(1, 12), _voice(2, 10)])
    r = SI.refine_label("ep:S1", [f"m{i}" for i in range(22)], E, cfg)
    assert sorted(r.kept) == ["ep:S1#a", "ep:S1#b"]
    assert sorted(len(v) for v in r.kept.values()) == [10, 12]

    E = np.concatenate([_voice(1, 19), _voice(9, 1)])
    r = SI.refine_label("ep:S2", [f"x{i}" for i in range(20)], E, cfg)
    assert r.dropped == {"x19": "destoa do locutor"}


def test_complete_linkage_merges_same_person_and_resists_chaining():
    rng = np.random.default_rng(0)
    C = np.concatenate([_voice(1, 3, rng=rng), _voice(2, 2, rng=rng), _voice(3, 1, rng=rng)])
    lab = SI.complete_linkage(C, 0.8)
    assert len(set(lab[:3])) == 1 and len(set(lab[3:5])) == 1 and len({lab[0], lab[3], lab[5]}) == 3
    # cadeia A~B, B~C, mas A !~ C: ligacao completa NAO junta os tres
    a = np.array([1.0, 0, 0])
    b = P.l2n(np.array([1.0, 1.0, 0]))
    c = np.array([0, 1.0, 0])
    lab = SI.complete_linkage(np.stack([a, b, c]), 0.7)
    assert len(set(lab)) >= 2


def test_calibration_and_groups():
    assert SI.calibrate_link_threshold(np.array([0.1, 0.5, 0.6]), floor=0.75) == 0.75
    assert SI.calibrate_link_threshold(np.full(1000, 0.8), floor=0.75) == pytest.approx(0.83)
    g = SI.episode_groups({"P1": ["ep1", "ep3"], "P2": ["ep2"], "P3": ["ep3", "ep4"]})
    assert g["ep1"] == g["ep3"] == g["ep4"] != g["ep2"]


def test_group_key_uses_explicit_group():
    assert SPL.group_key({"idx": "a", "corpus": "podcast", "speaker": "podcast:P001", "group": "G002"}) == "podcast/G002"
    assert SPL.group_key({"idx": "a", "corpus": "podcast", "speaker": "ep1:SPEAKER_00"}) == "podcast/ep1"


def test_podcast_identity_end_to_end(tmp_path, monkeypatch):
    """Dois episodios com o mesmo apresentador + um rotulo com 2 vozes + clipe com troca de voz + borda."""
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    for p in (str(PTBR / "core"), str(PTBR / "tools"), str(PTBR / "data")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import podcast_identity as PI
    import speaker_purity as SPU

    sr = 24000
    wavs = tmp_path / "wavs24"
    wavs.mkdir()
    rng = np.random.default_rng(0)

    def tone(name, f1, f2=None, dur=3.6):
        t = np.arange(int(dur * sr)) / sr
        f = np.where(t < dur / 2, f1, f2 or f1) + rng.normal(0, 2)
        y = 0.3 * np.sin(2 * np.pi * f * t) + 0.005 * rng.normal(size=len(t))
        sf.write(wavs / f"{name}.wav", y.astype(np.float32), sr)

    meta, index, rttm = [], [], {"ep1": [], "ep2": []}
    t0 = {"ep1": 0.0, "ep2": 0.0}

    def add(tag, spk, name, f1, f2=None):
        tone(name, f1, f2)
        a = t0[tag] + 0.5
        t0[tag] += 10.0
        index.append({"chunk": f"{name}.wav", "tag": tag, "speaker": spk, "dur": 3.6,
                      "spans": [[a, a + 3.25]]})
        rttm[tag].append((a, a + 3.25, spk))
        meta.append({"idx": name, "text": "frase", "dur_proc_s": 3.6})

    for i in range(10):
        add("ep1", "SPEAKER_00", f"ep1_host_{i}", 300)        # apresentador
        add("ep2", "SPEAKER_01", f"ep2_host_{i}", 300)        # mesmo apresentador, outro rotulo
        add("ep1", "SPEAKER_01", f"ep1_guest_{i}", 700)       # convidado
    for i in range(10):
        add("ep2", "SPEAKER_00", f"ep2_mix_{i}", 1100 if i < 5 else 1500)   # rotulo com 2 vozes
    add("ep1", "SPEAKER_00", "ep1_host_troca", 300, 1900)  # troca de voz no meio do clipe
    add("ep1", "SPEAKER_00", "ep1_host_borda", 300)        # margem invade outro locutor
    a = index[-1]["spans"][0][1]
    rttm["ep1"].append((a + 0.05, a + 2.0, "SPEAKER_01"))
    for s in range(6):                                     # calibracao: pessoas diferentes
        for c in range(6):
            tone(f"cal{s}_{c}", 2300 + 400 * s)
            meta.append({"idx": f"cal{s}_{c}", "corpus": "cetuc", "speaker": f"cetuc:S{s}", "text": "x",
                         "dur_proc_s": 3.6})
    (tmp_path / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta), encoding="utf-8")
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index), encoding="utf-8")
    (tmp_path / "rttm").mkdir()
    for tag, segs in rttm.items():
        (tmp_path / "rttm" / f"{tag}.rttm").write_text("".join(
            f"SPEAKER {tag} 1 {s:.3f} {e - s:.3f} <NA> <NA> {k} <NA> <NA>\n" for s, e, k in segs))

    class Fake:
        name = "fake"

        def __init__(self, *a, **k):
            self.n = 0

        def __call__(self, ws):
            out = []
            for w in ws:
                f0 = int(np.argmax(np.abs(np.fft.rfft(w, n=16000))))
                self.n += 1
                out.append(_voice(round(f0 / 100), 1, rng=np.random.default_rng(self.n))[0])
            return P.l2n(np.array(out))

    monkeypatch.setattr(SPU, "EcapaEmbedder", Fake)
    ds = tmp_path / "podcast"
    assert PI.main(["--index", str(tmp_path / "index.jsonl"), "--rttm-dir", str(tmp_path / "rttm"),
                    "--meta", str(tmp_path / "meta.jsonl"), "--wavs", str(wavs), "--dataset-dir", str(ds),
                    "--device", "cpu", "--calib-speakers", "6", "--calib-clips", "6"]) == 0
    rows = {r["idx"]: r for r in map(json.loads, (ds / "speakers.jsonl").read_text().splitlines())}
    host = {rows[f"ep1_host_{i}"]["speaker"] for i in range(10)} | {rows[f"ep2_host_{i}"]["speaker"] for i in range(10)}
    assert len(host) == 1 and host != {"podcast:99"}                       # mesma pessoa nos 2 episodios
    assert rows["ep1_host_0"]["group"] == rows["ep2_host_0"]["group"]       # episodios viram um grupo so
    guest = rows["ep1_guest_0"]["speaker"]
    assert guest not in host and guest != "podcast:99"
    mix = {rows[f"ep2_mix_{i}"]["speaker"] for i in range(10)}
    assert len(mix) <= 2 and not (mix & host)                               # rotulo de 2 vozes nao vira 1 pessoa
    assert rows["ep1_host_troca"]["status"] == "drop_mixed"
    assert rows["ep1_host_borda"]["status"] == "drop_bleed"
    assert (ds / "identity" / "report.html").is_file()
    summ = json.loads((ds / "identity" / "summary.json").read_text())
    assert summ["n_multi_episode"] == 1
