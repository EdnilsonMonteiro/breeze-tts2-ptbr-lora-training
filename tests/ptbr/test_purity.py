import json
import sys
from pathlib import Path

import numpy as np
import pytest

import purity as P

TOOLS = Path(__file__).resolve().parents[2] / "ptbr_lora" / "tools"


def _voice(seed, n, d=64, noise=0.15, rng=None):
    rng = rng or np.random.default_rng(seed)
    base = np.random.default_rng(1000 + seed).normal(size=d)
    return P.l2n(base + noise * np.linalg.norm(base) / np.sqrt(d) * rng.normal(size=(n, d)))


def test_clean_speaker_is_ok():
    E = _voice(1, 20)
    r = P.analyze_speaker("a", [f"a{i}" for i in range(20)], E, P.PurityConfig())
    assert r.status == "OK", r.reasons
    assert r.intra > 0.8 and r.n_outliers == 0 and not r.split_suspect


def test_two_people_under_one_label_is_flagged():
    E = np.concatenate([_voice(1, 12), _voice(2, 10)])
    r = P.analyze_speaker("mix", [f"m{i}" for i in range(22)], E, P.PurityConfig())
    assert r.split_suspect and r.status == "RUIM"
    assert sorted(r.split_sizes) == [10, 12]
    assert any("2 vozes" in x for x in r.reasons)


def test_single_intruder_clip_is_outlier_not_split():
    E = np.concatenate([_voice(1, 19), _voice(7, 1)])
    ids = [f"c{i}" for i in range(20)]
    r = P.analyze_speaker("a", ids, E, P.PurityConfig())
    flagged = [cid for cid, _, o, _ in r.clip_loo if o]
    assert flagged == ["c19"]
    assert not r.split_suspect
    assert r.status in ("OK", "SUSPEITO")


def test_few_clips_get_no_verdict():
    r = P.analyze_speaker("a", ["x", "y", "z"], _voice(1, 3), P.PurityConfig())
    assert r.status == "POUCOS"


def test_duplicates_and_cross_split_leak():
    clips, emb = [], {}
    for spk, seed, split in (("ep1:host", 1, "train"), ("ep2:host", 1, "val"), ("ep1:guest", 2, "train")):
        E = _voice(seed, 10, rng=np.random.default_rng(hash(spk) % 2**32))
        for i, e in enumerate(E):
            cid = f"{spk}_{i}"
            clips.append(P.Clip(id=cid, audio=Path(cid), speaker=spk, group="pod", split=split))
            emb[cid] = e
    results, dups, summ = P.analyze(clips, emb, P.PurityConfig())
    assert len(dups) == 1
    d = dups[0]
    assert {d["speaker_a"], d["speaker_b"]} == {"ep1:host", "ep2:host"} and d["cross_split"]
    by = {r.speaker: r for r in results}
    assert by["ep1:host"].nearest == "ep2:host" and by["ep1:host"].status == "SUSPEITO"
    assert any("VAZAMENTO" in x for x in by["ep2:host"].reasons)
    assert summ["pod"]["speakers"] == 3


def test_render_templates_and_manifest(tmp_path):
    (tmp_path / "w").mkdir()
    rows = [{"chunk": "a_1.wav", "tag": "ep1", "speaker": "SPEAKER_00"},
            {"chunk": "a_2.wav", "tag": "ep2", "speaker": "SPEAKER_00"}]
    m = tmp_path / "idx.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cl = P.clips_from_manifest(m, audio="{chunk}", speaker="{tag}:{speaker}", group="{tag}",
                               root=tmp_path / "w")
    assert [c.speaker for c in cl] == ["ep1:SPEAKER_00", "ep2:SPEAKER_00"]
    assert cl[0].audio == tmp_path / "w" / "a_1.wav" and cl[0].id == "a_1"
    with pytest.raises(KeyError, match="nao existe"):
        P.render("{nada}", rows[0])


def test_csv_pipe_and_training_dir(tmp_path):
    t = tmp_path / "tr"
    (t / "wavs24").mkdir(parents=True)
    (t / "manifest.csv").write_text("idx,corpus,speaker,wav_rel\nx1,cetuc,cetuc:A,wavs/x1.wav\n", encoding="utf-8")
    (t / "speaker_table.csv").write_text("split,corpus,group,speaker,n_clips,hours\nval,cetuc,g,cetuc:A,1,0.1\n",
                                         encoding="utf-8")
    c = P.clips_from_training_dir(t)[0]
    assert (c.id, c.group, c.split, c.audio) == ("x1", "cetuc", "val", t / "wavs24" / "x1.wav")
    p = tmp_path / "texts.csv"
    p.write_text("audio|speaker\nf/a.wav|s1\n", encoding="utf-8")
    assert P.clips_from_manifest(p, audio="{audio}", speaker="{speaker}")[0].speaker == "s1"


def test_sample_per_speaker_is_deterministic():
    clips = [P.Clip(id=f"{s}{i}", audio=Path("x"), speaker=s) for s in "ab" for i in range(50)]
    a = P.sample_per_speaker(clips, 10, seed=3)
    b = P.sample_per_speaker(clips, 10, seed=3)
    assert [c.id for c in a] == [c.id for c in b] and len(a) == 20


def test_cli_end_to_end_with_fake_embedder(tmp_path, monkeypatch):
    """Roda o CLI inteiro (leitura de pastas, cache, relatorios, exemplos) sem torch."""
    sf = pytest.importorskip("soundfile")
    sys.path.insert(0, str(TOOLS))
    import speaker_purity as SP

    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    rng = np.random.default_rng(0)
    freqs = {"ana": [300] * 8, "bia": [700] * 8, "mix": [300] * 4 + [1500] * 4}
    for spk, fs in freqs.items():
        (tmp_path / "in" / spk).mkdir(parents=True)
        for i, f in enumerate(fs):
            y = 0.3 * np.sin(2 * np.pi * (f + rng.normal(0, 3)) * t) + 0.01 * rng.normal(size=len(t))
            sf.write(tmp_path / "in" / spk / f"{i}.wav", y.astype(np.float32), sr)

    class Fake:
        name = "fake-fft"

        def __init__(self, *a, **k):
            pass

        def __call__(self, wavs):
            # "voz" = frequencia dominante -> vetor fixo por voz + ruido pequeno (imita um embedding real)
            out = []
            for w in wavs:
                f0 = int(np.argmax(np.abs(np.fft.rfft(w[:sr]))))
                v = _voice(round(f0 / 100), 1, rng=np.random.default_rng(len(out) + f0))[0]
                out.append(v)
            return P.l2n(np.array(out))

    monkeypatch.setattr(SP, "EcapaEmbedder", Fake)
    out = tmp_path / "out"
    assert SP.main(["--folders", str(tmp_path / "in"), "--out", str(out), "--device", "cpu"]) == 0
    spk = {r.split(",")[0]: r.split(",")[3] for r in (out / "speakers.csv").read_text().splitlines()[1:]}
    assert spk["ana"] == "OK" and spk["bia"] == "OK" and spk["mix"] == "RUIM"
    assert (out / "exclude_speakers.txt").read_text().split() == ["mix"]
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "RUIM — mix" in html and "listen/001_mix/" in html
    assert len(list((out / "listen" / "001_mix").glob("*.wav"))) >= 3
    # segunda execucao: tudo vem do cache
    assert SP.main(["--folders", str(tmp_path / "in"), "--out", str(out), "--device", "cpu"]) == 0
    assert json.loads((out / "summary.json").read_text())["status"]["RUIM"] == 1
