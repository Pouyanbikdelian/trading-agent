"""Telegram offset persistence — the crash-replay guard."""

from __future__ import annotations


def test_offset_round_trip(tmp_path, monkeypatch) -> None:
    from trading.bot import telegram as tg

    class _S:
        state_dir = tmp_path

    monkeypatch.setattr(tg, "settings", _S())
    assert tg._load_offset() == 0  # missing file → old behavior
    tg._save_offset(4242)
    assert tg._load_offset() == 4242
    (tmp_path / "telegram_offset.json").write_text("{corrupt")
    assert tg._load_offset() == 0  # corrupt file degrades, never raises
