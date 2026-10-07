import json

import unseen_eval as UE


def _plan(tmp_path):
    ref = tmp_path / "ref_a.wav"
    ref.write_bytes(b"RIFFfake")
    return {"voices": [{"name": "voz_a", "corpus": "cetuc", "prompt": {"audio": str(ref), "text": "x"},
                        "heldout": []}]}


def test_listen_page_lists_reference_and_each_step(tmp_path):
    plan = _plan(tmp_path)
    run = tmp_path / "run"
    for step in (0, 500, 1000):
        d = run / "eval_unseen" / f"step{step}"
        d.mkdir(parents=True)
        for k in range(2):
            (d / f"voz_a__p{k}.wav").write_bytes(b"RIFFfake")
        items = [{"voice": "voz_a", "phrase": k, "secs_heldout": 0.5 + step / 10000, "wer": 0.1}
                 for k in range(2)]
        (d / "unseen_scores.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    page = UE.write_listen_page(run, plan, 2)
    assert page == run / "ouvir" / "index.html"
    html = page.read_text(encoding="utf-8")
    assert (run / "ouvir" / "refs" / "voz_a.wav").is_file()
    assert "refs/voz_a.wav" in html
    for step in (0, 500, 1000):
        assert f"../eval_unseen/step{step}/voz_a__p0.wav" in html
    assert html.index("passo 0") < html.index("passo 500") < html.index("passo 1000")
    assert "SECS 0.55" in html


def test_listen_page_none_without_steps(tmp_path):
    assert UE.write_listen_page(tmp_path / "run", _plan(tmp_path), 2) is None
