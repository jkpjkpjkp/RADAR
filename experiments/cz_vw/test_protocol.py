"""Accounting and timing fixtures are tests, never empirical portfolio evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from .data import Panel, histories, prior_month
from .evaluate import fit_alpha
from .portfolio import backtest, decile_targets


def test_resume_preserves_checkpoint_and_does_not_retrain(tmp_path, monkeypatch):
    from . import run

    # A tiny real Panel exercises NumPy date loading and restricted checkpoint I/O.
    dates = (
        pd.bdate_range("2011-06-02", periods=1260)
        .strftime("%Y-%m-%d")
        .to_numpy(dtype="U10")
    )
    for name, value in {
        "dates": dates,
        "symbols": np.array(["P1"]),
        "monthly_features": np.zeros((1, 1, 202), np.float32),
        "month_index": np.zeros(len(dates), int),
        **{
            n: np.ones((1, len(dates)))
            for n in (
                "member",
                "tradable",
                "totalret",
                "terminal_events",
                "price_return",
                "lagged_cap",
            )
        },
    }.items():
        np.save(tmp_path / f"{name}.npy", value)
    torch.save({"data": {}}, tmp_path / "news.pt")
    factors = tmp_path / "factors.parquet"
    factors.write_bytes(b"unused fixture factors")
    audit = {"news_sha256": run.sha(tmp_path / "news.pt"), "manifest_sha256": "fixture"}
    (tmp_path / "data_audit.json").write_text(json.dumps(audit))
    monkeypatch.setattr(run, "verify_cache", lambda _: {"manifest_sha256": "fixture"})
    calls = []

    def train(k, window, data, cfg, out, **kwargs):
        calls.append(k)
        model = run.RADAR(cfg)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": run.asdict(cfg),
            "train_period": [data.dates[0], data.dates[1007]],
        }
        torch.save(checkpoint, Path(out) / "window_1.pt")
        return {"checkpoint": "window_1.pt", "best_val_loss": 0.1}

    monkeypatch.setattr(run, "train_window", train)
    monkeypatch.setattr(run, "histories", lambda *args: ([], [], []))
    monkeypatch.setattr(run.RADAR, "build_context_bank", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run, "predict", lambda *args: pd.DataFrame({"date": ["2015-06-08"]})
    )
    monkeypatch.setattr(run, "evaluate", lambda *args: None)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--shared_data",
            str(tmp_path),
            "--data_dir",
            str(tmp_path),
            "--factors",
            str(factors),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--epochs",
            "1",
        ],
    )
    run.main()
    first_hash = run.sha(output / "window_1.pt")
    run.main()
    assert calls == [1]
    assert run.sha(output / "window_1.pt") == first_hash


def fixture_panel():
    n = 20
    return SimpleNamespace(
        symbols=np.array([f"P{i:02}" for i in range(n)]),
        dates=np.array(["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"]),
        member=np.ones((n, 4), bool),
        tradable=np.ones((n, 4), bool),
        terminal_events=np.zeros((n, 4), bool),
        lagged_cap=np.ones((n, 4)),
        totalret=np.zeros((n, 4)),
        price_return=np.zeros((n, 4)),
    )


def predictions(panel):
    return pd.DataFrame(
        [
            {"date": d, "symbol": s, "prediction": i}
            for d in panel.dates[:2]
            for i, s in enumerate(panel.symbols)
        ]
    )


def test_cap_weighted_deciles_and_deterministic_ties():
    ids = np.arange(20)
    caps = np.arange(1, 21)
    a = decile_targets(ids, np.zeros(20), caps, 20)
    b = decile_targets(ids[::-1], np.zeros(20), caps[::-1], 20)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[9, 18:], [19 / 39, 20 / 39])
    assert set(np.flatnonzero(a[0])) == {0, 1}
    np.testing.assert_allclose((a[9] - a[0]).sum(), 0, atol=1e-15)


def test_execution_lag_costs_and_final_liquidation():
    p = fixture_panel()
    p.totalret[18:, 1] = 100.0  # Must not earn execution-day return.
    p.totalret[18:, 2] = 0.1
    p.price_return[18:, 2] = 0.1
    gross, _, _ = backtest(p, predictions(p), 0)
    net, _, _ = backtest(p, predictions(p), 10)
    np.testing.assert_allclose(gross.return_gross, [0.1, 0])
    assert net.date.tolist() == ["2016-01-06", "2016-01-07"]
    assert net.fees.iloc[0] == pytest.approx(0.002)
    assert net.final_liquidation_dollars.iloc[-1] > 0
    np.testing.assert_allclose(
        net.return_net, net.return_gross - net.fees / net.start_nav
    )
    np.testing.assert_allclose(net.end_nav, net.start_nav + net.pnl - net.fees)


def test_halted_holdings_not_fabricated_sales():
    p = fixture_panel()
    p.totalret[18:, 2] = 0.1
    p.price_return[18:, 2] = 0.1
    p.tradable[19, 2:] = False
    ledger, w, _ = backtest(p, predictions(p), 0)
    assert w[1, 19] * ledger.start_nav.iloc[1] == pytest.approx(w[0, 19] * 1.1)
    assert ledger.final_unliquidated_dollars.iloc[-1] > 0


def test_calendar_lag_and_future_independent_eligibility():
    np.testing.assert_array_equal(
        prior_month(["2016-01-31", "2016-02-01", "2017-01-03"]),
        [201512, 201601, 201612],
    )
    p = fixture_panel()
    before = Panel.ids(p, 0)
    p.totalret[:, 2] = np.nan
    p.tradable[:, 2] = False
    np.testing.assert_array_equal(Panel.ids(p, 0), before)


def test_retrieval_bank_excludes_future_index_entrants():
    days = ["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"]
    p = SimpleNamespace(
        symbols=np.array(["P1", "P2"]),
        tickers=["P1", "P2"],
        dates=days,
        member=np.array([[True] * 4, [False, False, True, True]]),
        news={s: {d: np.ones(768, np.float32) for d in days} for s in ["P1", "P2"]},
        features=lambda ids, start, end: torch.ones(2, end - start, 202),
    )
    features, news, masks = histories(p, 0, 4, 768)
    assert masks[1].tolist() == [True, True, False, False]
    assert features[1][:2].count_nonzero() == news[1][:2].count_nonzero() == 0


def test_hac_alpha_units_and_factor_date_validation():
    rng = np.random.default_rng(14)
    n = 300
    names = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    x = rng.normal(0, 0.01, (n, 6))
    design = np.column_stack([np.ones(n), x])
    noise = rng.normal(0, 0.005, n)
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    dates = pd.bdate_range("2017-01-01", periods=n).astype(str)
    factors = pd.DataFrame(x, columns=names)
    factors["date"] = dates
    ledger = pd.DataFrame(
        {"date": dates, "return_net": 0.0004 + x @ np.arange(6) / 10 + noise}
    )
    r = fit_alpha(ledger, factors)
    assert r["annual_alpha"] == pytest.approx(0.0004 * 252)
    assert r["ci_low"] < r["annual_alpha"] < r["ci_high"] and 0 <= r["p"] <= 1
    with pytest.raises(ValueError, match="Missing factor"):
        fit_alpha(ledger, factors.iloc[1:])
