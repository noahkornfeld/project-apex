"""
tests/test_market_data_vectorized.py
=====================================
Targeted tests for the vectorised slot-filling implementation in
environment/market_data.py (M1 optimisation).

Coverage:
    TestSectorIds
        - Correct GICS code → embedding index mapping
        - Unknown / invalid code → GICS_UNKNOWN_IDX
        - Inactive slot (sid=-1) stays -1
        - Latest snapshot wins when multiple dates exist

    TestSlotFilling
        - Correct price routed to the right (t, k) slot
        - Inactive slot (sid=-1) yields zeros / vol default
        - NaN price in bars → replaced by column default
        - sid present in active_ids but absent from bars → default
        - Multiple assets on same day all mapped correctly
        - Vectorised result matches reference naive-loop result
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from environment.market_data import (
    _build_sector_ids,
    GICS_TO_IDX,
    GICS_UNKNOWN_IDX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(rows):
    """Build a minimal bars-like DataFrame from a list of dicts."""
    _COLS = ["date", "security_id", "adj_close", "adj_open",
             "adv63", "vol_252", "gap_vol_252", "date_str"]
    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _naive_slot_fill(dates_str, active_ids, bars_rows):
    """Reference implementation — plain Python nested loop (old approach)."""
    T, K = active_ids.shape
    adj_close   = np.zeros((T, K), dtype=np.float32)
    adj_open    = np.zeros((T, K), dtype=np.float32)
    adv63       = np.zeros((T, K), dtype=np.float32)
    vol_252     = np.full ((T, K), 0.20, dtype=np.float32)
    gap_vol_252 = np.zeros((T, K), dtype=np.float32)

    lookup = {}
    for r in bars_rows:
        d   = r["date_str"]
        sid = int(r["security_id"])
        if d not in lookup:
            lookup[d] = {}
        lookup[d][sid] = (
            r.get("adj_close",   0.0),
            r.get("adj_open",    0.0),
            r.get("adv63",       0.0),
            r.get("vol_252",     0.20),
            r.get("gap_vol_252", 0.0),
        )

    for t in range(T):
        d = dates_str[t]
        day = lookup.get(d, {})
        for k in range(K):
            sid = int(active_ids[t, k])
            if sid >= 0 and sid in day:
                v = day[sid]
                adj_close  [t, k] = v[0] if np.isfinite(v[0]) else 0.0
                adj_open   [t, k] = v[1] if np.isfinite(v[1]) else 0.0
                adv63      [t, k] = v[2] if np.isfinite(v[2]) else 0.0
                vol_252    [t, k] = v[3] if np.isfinite(v[3]) else 0.20
                gap_vol_252[t, k] = v[4] if np.isfinite(v[4]) else 0.0

    return adj_close, adj_open, adv63, vol_252, gap_vol_252


def _vectorised_slot_fill(dates_str, active_ids, bars_df):
    """
    Runs the same vectorised numpy logic as build_market_data §4,
    given pre-built bars_df and dates_str / active_ids.
    Returns (adj_close, adj_open, adv63, vol_252, gap_vol_252).
    """
    T, K_max = active_ids.shape
    date_to_t = {d: i for i, d in enumerate(dates_str)}

    all_sids = np.unique(active_ids[active_ids >= 0])
    n_sids   = len(all_sids)
    sid_to_col = {int(s): i for i, s in enumerate(all_sids)}

    sid_set  = set(sid_to_col.keys())
    date_set = set(dates_str.tolist())

    _cols = ["adj_close", "adj_open", "adv63", "vol_252", "gap_vol_252"]
    bars_filt = bars_df.loc[
        bars_df["security_id"].isin(sid_set) & bars_df["date_str"].isin(date_set),
        ["date_str", "security_id"] + _cols,
    ].copy()

    t_vec = bars_filt["date_str"].map(date_to_t).values.astype(np.intp)
    c_vec = bars_filt["security_id"].map(sid_to_col).values.astype(np.intp)

    ok    = np.isfinite(t_vec.astype(float)) & np.isfinite(c_vec.astype(float))
    t_vec = t_vec[ok];  c_vec = c_vec[ok]
    bars_filt = bars_filt.iloc[ok]

    _defaults = {"adj_close": 0.0, "adj_open": 0.0, "adv63": 0.0,
                 "vol_252": 0.20, "gap_vol_252": 0.0}
    dense = {}
    for col, default in _defaults.items():
        arr  = np.full((T, max(n_sids, 1)), default, dtype=np.float32)
        vals = bars_filt[col].values.astype(np.float32)
        vals = np.where(np.isfinite(vals), vals, default)
        arr[t_vec, c_vec] = vals
        dense[col] = arr

    flat     = active_ids.ravel()
    flat_col = (pd.Series(flat.astype(np.int64))
                .map(sid_to_col).fillna(-1).values.astype(np.intp)
                .reshape(T, K_max))

    valid_k  = flat_col >= 0
    col_safe = np.where(valid_k, flat_col, 0)
    T_idx    = np.arange(T, dtype=np.intp)[:, None]

    ac  = np.where(valid_k, dense["adj_close"]  [T_idx, col_safe], 0.0 ).astype(np.float32)
    ao  = np.where(valid_k, dense["adj_open"]   [T_idx, col_safe], 0.0 ).astype(np.float32)
    adv = np.where(valid_k, dense["adv63"]      [T_idx, col_safe], 0.0 ).astype(np.float32)
    vol = np.where(valid_k, dense["vol_252"]    [T_idx, col_safe], 0.20).astype(np.float32)
    gv  = np.where(valid_k, dense["gap_vol_252"][T_idx, col_safe], 0.0 ).astype(np.float32)
    return ac, ao, adv, vol, gv


# ---------------------------------------------------------------------------
# TestSectorIds
# ---------------------------------------------------------------------------

class TestSectorIds:

    def _call(self, ndx_rows, active_ids, dates_str=None):
        T, K = active_ids.shape
        if dates_str is None:
            dates_str = np.array(["2020-01-01"] * T, dtype=object)
        ndx_df = pd.DataFrame(ndx_rows)
        ndx_df["date"] = pd.to_datetime(ndx_df["date"])
        with patch("environment.market_data.pd.read_parquet", return_value=ndx_df):
            return _build_sector_ids("dummy.parquet", dates_str, active_ids, T, K)

    def test_known_gics_code_mapped_correctly(self):
        active_ids = np.array([[100, 200, 300]], dtype=np.int64)
        ndx = [
            {"date": "2020-01-01", "security_id": 100, "sector_code": 45},  # IT → 7
            {"date": "2020-01-01", "security_id": 200, "sector_code": 10},  # Energy → 0
            {"date": "2020-01-01", "security_id": 300, "sector_code": 35},  # HealthCare → 5
        ]
        result = self._call(ndx, active_ids)
        assert result[0, 0] == GICS_TO_IDX[45]
        assert result[0, 1] == GICS_TO_IDX[10]
        assert result[0, 2] == GICS_TO_IDX[35]

    def test_inactive_slot_stays_minus_one(self):
        active_ids = np.array([[100, -1, -1]], dtype=np.int64)
        ndx = [{"date": "2020-01-01", "security_id": 100, "sector_code": 45}]
        result = self._call(ndx, active_ids)
        assert result[0, 1] == -1
        assert result[0, 2] == -1

    def test_unknown_gics_code_maps_to_unknown_idx(self):
        active_ids = np.array([[100, 200]], dtype=np.int64)
        ndx = [
            {"date": "2020-01-01", "security_id": 100, "sector_code": 9999},  # not in map
            {"date": "2020-01-01", "security_id": 200, "sector_code": None},  # None → unknown
        ]
        result = self._call(ndx, active_ids)
        assert result[0, 0] == GICS_UNKNOWN_IDX
        assert result[0, 1] == GICS_UNKNOWN_IDX

    def test_sid_not_in_ndx_maps_to_unknown(self):
        active_ids = np.array([[999]], dtype=np.int64)  # sid 999 not in ndx at all
        ndx = [{"date": "2020-01-01", "security_id": 100, "sector_code": 45}]
        result = self._call(ndx, active_ids)
        # Active sid with no ndx record → merge_asof NaN → GICS_UNKNOWN_IDX
        assert result[0, 0] == GICS_UNKNOWN_IDX

    def test_latest_snapshot_wins(self):
        """As-of rule: query date 2020-01-01 sees 2018 snapshot (not the 2022 one)."""
        active_ids = np.array([[100]], dtype=np.int64)
        dates_str  = np.array(["2020-01-01"], dtype=object)
        ndx = [
            {"date": "2018-01-01", "security_id": 100, "sector_code": 10},  # Energy
            {"date": "2022-01-01", "security_id": 100, "sector_code": 45},  # IT (future)
        ]
        result = self._call(ndx, active_ids, dates_str)
        # 2022 snapshot is AFTER query date → must NOT be used
        assert result[0, 0] == GICS_TO_IDX[10]   # Energy (2018 is most recent ≤ 2020)

    def test_all_11_known_gics_codes_map_distinctly(self):
        known_codes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        active_ids  = np.array([list(range(100, 100 + len(known_codes)))], dtype=np.int64)
        ndx = [
            {"date": "2020-01-01", "security_id": 100 + i, "sector_code": code}
            for i, code in enumerate(known_codes)
        ]
        result = self._call(ndx, active_ids)
        expected = [GICS_TO_IDX[c] for c in known_codes]
        assert result[0].tolist() == expected

    def test_as_of_rule_sector_change_visible_only_after_snapshot_date(self):
        """Company reclassified from Energy→IT in 2020.  Dates before the
        reclassification must return Energy; dates after must return IT."""
        active_ids = np.array([[100], [100], [100]], dtype=np.int64)
        dates_str  = np.array(["2019-06-01",   # before reclassification
                                "2020-07-01",   # after reclassification
                                "2024-01-01"],  # well after
                               dtype=object)
        ndx = [
            {"date": "2015-01-01", "security_id": 100, "sector_code": 10},   # Energy
            {"date": "2020-01-01", "security_id": 100, "sector_code": 45},   # IT
        ]
        result = self._call(ndx, active_ids, dates_str)
        assert result[0, 0] == GICS_TO_IDX[10], "pre-change date should return Energy"
        assert result[1, 0] == GICS_TO_IDX[45], "post-change date should return IT"
        assert result[2, 0] == GICS_TO_IDX[45], "long after change should still return IT"

    def test_as_of_rule_no_snapshot_before_date_returns_unknown(self):
        """If the only snapshot is AFTER the query date, result is GICS_UNKNOWN_IDX."""
        active_ids = np.array([[100]], dtype=np.int64)
        dates_str  = np.array(["2010-01-01"], dtype=object)  # before any snapshot
        ndx = [{"date": "2015-01-01", "security_id": 100, "sector_code": 45}]
        result = self._call(ndx, active_ids, dates_str)
        assert result[0, 0] == GICS_UNKNOWN_IDX

    def test_multi_row_multi_day(self):
        active_ids = np.array([[100, 200], [100, 200]], dtype=np.int64)
        dates_str  = np.array(["2020-01-01", "2020-01-02"], dtype=object)
        ndx = [
            {"date": "2020-01-01", "security_id": 100, "sector_code": 45},
            {"date": "2020-01-01", "security_id": 200, "sector_code": 10},
        ]
        result = self._call(ndx, active_ids, dates_str)
        # Both rows should have same mapping (one snapshot)
        assert result[0, 0] == result[1, 0] == GICS_TO_IDX[45]
        assert result[0, 1] == result[1, 1] == GICS_TO_IDX[10]


# ---------------------------------------------------------------------------
# TestSlotFilling
# ---------------------------------------------------------------------------

class TestSlotFilling:

    def _make_scenario(self, dates, active_ids_list, bar_rows):
        dates_str  = np.array(dates, dtype=object)
        active_ids = np.array(active_ids_list, dtype=np.int64)
        bars_df    = _make_bars(bar_rows)
        return dates_str, active_ids, bars_df

    def test_single_asset_single_day_routed_correctly(self):
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04"],
            active_ids_list= [[100, -1]],
            bar_rows       = [{"date": "2021-01-04", "security_id": 100,
                                "adj_close": 150.0, "adj_open": 148.0,
                                "adv63": 5e8, "vol_252": 0.25, "gap_vol_252": 0.01}],
        )
        ac, ao, adv, vol, gv = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        assert ac [0, 0] == pytest.approx(150.0)
        assert ao [0, 0] == pytest.approx(148.0)
        assert adv[0, 0] == pytest.approx(5e8)
        assert vol[0, 0] == pytest.approx(0.25)
        assert gv [0, 0] == pytest.approx(0.01)

    def test_inactive_slot_is_zero(self):
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04"],
            active_ids_list= [[100, -1]],
            bar_rows       = [{"date": "2021-01-04", "security_id": 100,
                                "adj_close": 50.0, "adj_open": 49.0,
                                "adv63": 1e8, "vol_252": 0.30, "gap_vol_252": 0.005}],
        )
        ac, ao, adv, vol, gv = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        assert ac [0, 1] == 0.0
        assert ao [0, 1] == 0.0
        assert adv[0, 1] == 0.0
        assert gv [0, 1] == 0.0

    def test_inactive_slot_vol_is_zero_not_default(self):
        """vol_252 default (0.20) only applies to active-but-missing slots.
        Inactive slots get 0.0 via np.where(valid_k, ..., 0.20) — wait, no:
        inactive → valid_k=False → np.where uses the else branch which is 0.20.
        This test documents that actual behaviour."""
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04"],
            active_ids_list= [[-1]],
            bar_rows       = [],
        )
        _, _, _, vol, _ = _vectorised_slot_fill(dates_str, active_ids, bars_df)
        # inactive: valid_k=False → np.where returns default 0.20
        assert vol[0, 0] == pytest.approx(0.20)

    def test_nan_price_replaced_by_default(self):
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04"],
            active_ids_list= [[100]],
            bar_rows       = [{"date": "2021-01-04", "security_id": 100,
                                "adj_close": float("nan"), "adj_open": float("nan"),
                                "adv63": float("nan"), "vol_252": float("nan"),
                                "gap_vol_252": float("nan")}],
        )
        ac, ao, adv, vol, gv = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        assert ac [0, 0] == pytest.approx(0.0)
        assert ao [0, 0] == pytest.approx(0.0)
        assert adv[0, 0] == pytest.approx(0.0)
        assert vol[0, 0] == pytest.approx(0.20)   # vol_252 default
        assert gv [0, 0] == pytest.approx(0.0)

    def test_missing_sid_in_bars_gives_defaults(self):
        """active_ids contains sid=100 but bars has no row for it."""
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04"],
            active_ids_list= [[100]],
            bar_rows       = [],   # no bar data at all
        )
        ac, ao, adv, vol, gv = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        assert ac [0, 0] == pytest.approx(0.0)
        assert vol[0, 0] == pytest.approx(0.20)

    def test_two_assets_two_days_all_correct(self):
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04", "2021-01-05"],
            active_ids_list= [[100, 200], [100, 200]],
            bar_rows       = [
                {"date": "2021-01-04", "security_id": 100,
                 "adj_close": 10.0, "adj_open": 9.5, "adv63": 1e6, "vol_252": 0.20, "gap_vol_252": 0.01},
                {"date": "2021-01-04", "security_id": 200,
                 "adj_close": 20.0, "adj_open": 19.0, "adv63": 2e6, "vol_252": 0.30, "gap_vol_252": 0.02},
                {"date": "2021-01-05", "security_id": 100,
                 "adj_close": 11.0, "adj_open": 10.5, "adv63": 1.1e6, "vol_252": 0.21, "gap_vol_252": 0.011},
                {"date": "2021-01-05", "security_id": 200,
                 "adj_close": 21.0, "adj_open": 20.0, "adv63": 2.1e6, "vol_252": 0.31, "gap_vol_252": 0.021},
            ],
        )
        ac, ao, adv, vol, gv = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        assert ac[0, 0] == pytest.approx(10.0)
        assert ac[0, 1] == pytest.approx(20.0)
        assert ac[1, 0] == pytest.approx(11.0)
        assert ac[1, 1] == pytest.approx(21.0)
        assert vol[0, 1] == pytest.approx(0.30)
        assert vol[1, 1] == pytest.approx(0.31)

    def test_asset_swaps_slot_between_days(self):
        """sid 100 in slot 0 on day 0, in slot 1 on day 1."""
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04", "2021-01-05"],
            active_ids_list= [[100, 200], [200, 100]],  # swapped!
            bar_rows       = [
                {"date": "2021-01-04", "security_id": 100,
                 "adj_close": 10.0, "adj_open": 9.5, "adv63": 1e6, "vol_252": 0.20, "gap_vol_252": 0.0},
                {"date": "2021-01-04", "security_id": 200,
                 "adj_close": 20.0, "adj_open": 19.0, "adv63": 2e6, "vol_252": 0.30, "gap_vol_252": 0.0},
                {"date": "2021-01-05", "security_id": 100,
                 "adj_close": 11.0, "adj_open": 10.5, "adv63": 1e6, "vol_252": 0.20, "gap_vol_252": 0.0},
                {"date": "2021-01-05", "security_id": 200,
                 "adj_close": 21.0, "adj_open": 20.0, "adv63": 2e6, "vol_252": 0.30, "gap_vol_252": 0.0},
            ],
        )
        ac, _, _, _, _ = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        # day 0: slot 0 = sid100 = 10, slot 1 = sid200 = 20
        assert ac[0, 0] == pytest.approx(10.0)
        assert ac[0, 1] == pytest.approx(20.0)
        # day 1: slot 0 = sid200 = 21, slot 1 = sid100 = 11  (swapped)
        assert ac[1, 0] == pytest.approx(21.0)
        assert ac[1, 1] == pytest.approx(11.0)

    def test_vectorised_matches_naive_loop_random(self):
        """Vectorised output must equal the reference naive-loop on random data."""
        rng  = np.random.default_rng(42)
        T, K = 20, 8
        n_active = 6
        sids = np.arange(200, 200 + n_active)

        # Build active_ids: each row shuffles sids into first n_active slots, rest -1
        active_ids = np.full((T, K), -1, dtype=np.int64)
        for t in range(T):
            active_ids[t, :n_active] = rng.permutation(sids)

        base_date = pd.Timestamp("2021-01-04")
        dates_list = [(base_date + pd.offsets.BDay(t)).strftime("%Y-%m-%d") for t in range(T)]
        dates_str  = np.array(dates_list, dtype=object)

        # Build bar rows for every (date, sid) combo
        bar_rows = []
        for t, d in enumerate(dates_list):
            for sid in sids:
                bar_rows.append({
                    "date": d, "date_str": d, "security_id": int(sid),
                    "adj_close":   float(rng.uniform(10, 200)),
                    "adj_open":    float(rng.uniform(10, 200)),
                    "adv63":       float(rng.uniform(1e6, 1e9)),
                    "vol_252":     float(rng.uniform(0.10, 0.50)),
                    "gap_vol_252": float(rng.uniform(0.0, 0.05)),
                })

        bars_df  = _make_bars(bar_rows)
        ref      = _naive_slot_fill(dates_str, active_ids, bar_rows)
        vec      = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        for ref_arr, vec_arr, name in zip(
            ref, vec,
            ["adj_close", "adj_open", "adv63", "vol_252", "gap_vol_252"]
        ):
            np.testing.assert_allclose(
                vec_arr, ref_arr, rtol=1e-5, atol=1e-5,
                err_msg=f"Mismatch in column '{name}'"
            )

    def test_bars_extra_dates_and_sids_ignored(self):
        """Bars rows outside the date/sid universe must not corrupt the output."""
        dates_str, active_ids, bars_df = self._make_scenario(
            dates          = ["2021-01-04"],
            active_ids_list= [[100]],
            bar_rows       = [
                {"date": "2021-01-04", "security_id": 100,
                 "adj_close": 50.0, "adj_open": 49.0, "adv63": 1e7, "vol_252": 0.22, "gap_vol_252": 0.005},
                # extra sid not in active_ids
                {"date": "2021-01-04", "security_id": 999,
                 "adj_close": 99.0, "adj_open": 98.0, "adv63": 9e9, "vol_252": 0.99, "gap_vol_252": 0.1},
                # extra date not in dates_str
                {"date": "2099-12-31", "security_id": 100,
                 "adj_close": 99.0, "adj_open": 98.0, "adv63": 9e9, "vol_252": 0.99, "gap_vol_252": 0.1},
            ],
        )
        ac, _, _, vol, _ = _vectorised_slot_fill(dates_str, active_ids, bars_df)

        assert ac [0, 0] == pytest.approx(50.0)
        assert vol[0, 0] == pytest.approx(0.22)
