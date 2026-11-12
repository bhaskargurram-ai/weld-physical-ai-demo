"""Read / query helpers over Mosaico (or local files in offline mode).

Every function returns the same shapes online (mosaicod) and offline (MOSAICO_OFFLINE=1,
reading data/generated/ directly).

    python -m mosaico_io.query list --process FCAW --spool_id SP-1043
    python -m mosaico_io.query anomalies current_a --gt 250
    python -m mosaico_io.query violations SP-1043_J07_P3_FCAW
"""
from __future__ import annotations

import argparse
import json
import sys
from fnmatch import fnmatch
from typing import Any, Optional

import numpy as np
import pandas as pd

from collect.schema import ARC_HZ, TOPIC_ARC, TOPIC_THERMAL, TOPIC_TRAVEL
from mosaico_io.common import (ARC_COLS, SIGNALS, THERMAL_COLS, TRAVEL_COLS, assemble_signals, avg_signals, connect,
                               flat_keys,
                               is_offline, json_default, load_meta, local_sequences, meta_row, offline_signals)

_OPS = ("eq", "neq", "gt", "geq", "lt", "leq", "match", "in_", "between")


def _split_filter(key: str, value: Any) -> tuple[str, str, Any]:
    """'heat_input_mean_kj_mm__gt' -> (key, 'gt', value); list -> in_; '*' in str -> match."""
    if "__" in key:
        k, op = key.rsplit("__", 1)
        op = {"in": "in_", "gte": "geq", "lte": "leq", "ne": "neq"}.get(op, op)
        if op in _OPS:
            return k, op, value
    if isinstance(value, (list, tuple, set)):
        return key, "in_", list(value)
    if isinstance(value, str) and any(ch in value for ch in "*?["):
        return key, "match", value
    return key, "eq", value


def _match_local(row: dict, key: str, op: str, value: Any) -> bool:
    v = row
    for part in key.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    if op == "eq":
        return v == value
    if op == "neq":
        return v != value
    if op == "in_":
        return v in value
    if op == "match":
        return v is not None and fnmatch(str(v), str(value))
    if v is None:
        return False
    if op == "gt":
        return v > value
    if op == "geq":
        return v >= value
    if op == "lt":
        return v < value
    if op == "leq":
        return v <= value
    if op == "between":
        return value[0] <= v <= value[1]
    raise ValueError(op)


# ------------------------------------------------------------------ metadata


def get_meta(seq: str) -> dict:
    """Full pass metadata (meta.json content) for a sequence."""
    if is_offline():
        return load_meta(seq)
    h = connect().sequence_handler(seq)
    if h is None:
        raise KeyError(f"sequence not found in mosaicod: {seq}")
    return dict(h.user_metadata)


def list_passes(**filters) -> list[dict]:
    """Metadata rows for weld passes, filtered on sequence user metadata.

    Filters: ``process="FCAW"``, ``spool_id="SP-1043"``, ``pass_type=["fill", "cap"]`` (one-of),
    ``has_defect=True``, ``heat_input_mean_kj_mm__gt=1.0``, ``sequence_name="SP-1043_*"`` (glob),
    nested keys with dots via ``**{"wps.max_interpass_c__lt": 255}``.
    Online this is a server-side ``QuerySequence().with_user_metadata(...)`` query.
    """
    parsed = [_split_filter(k, v) for k, v in filters.items() if v is not None]
    rows: list[dict] = []
    if is_offline():
        for seq in local_sequences():
            m = load_meta(seq)
            m = {**flat_keys(m), **m}
            if all(_match_local(m, k, op, v) for k, op, v in parsed):
                rows.append(meta_row(m))
    else:
        from mosaicolabs import QuerySequence
        client = connect()
        if parsed:
            q = QuerySequence()
            for k, op, v in parsed:
                if k == "sequence_name" and op in ("eq", "match"):
                    q = q.with_name(v) if op == "eq" else q.with_name_match(v)
                else:
                    q = q.with_user_metadata(k, **{op: v})
            resp = client.query(q)
            names = [it.sequence.name for it in resp] if resp else []
        else:
            names = client.list_sequences()
        for n in names:
            h = client.sequence_handler(n)
            if h is None:
                continue
            m = dict(h.user_metadata)
            if "spool_id" not in m or "wps_id" not in m:      # not a weld pass
                continue
            m.setdefault("sequence_name", n)
            rows.append(meta_row(m))
    rows.sort(key=lambda r: (r.get("t0_ns") or 0, r.get("sequence_name") or ""))
    return rows


# ------------------------------------------------------------------ signals


def _online_signals(seq: str, t0_s=None, t1_s=None) -> pd.DataFrame:
    from mosaicolabs.ml import DataFrameExtractor
    client = connect()
    h = client.sequence_handler(seq)
    if h is None:
        raise KeyError(f"sequence not found in mosaicod: {seq}")
    t0_ns = int(h.user_metadata.get("t0_ns", h.timestamp_ns_min))
    start = None if t0_s is None else int(t0_ns + t0_s * 1e9)
    end = None if t1_s is None else int(t0_ns + t1_s * 1e9)
    # pad so interpolation at the edges has neighbours
    pad = int(0.2e9)
    chunks = list(DataFrameExtractor(h).to_pandas_chunks(
        topics=[TOPIC_ARC, TOPIC_TRAVEL, TOPIC_THERMAL], window_sec=1e6,
        timestamp_ns_start=None if start is None else start - pad,
        timestamp_ns_end=None if end is None else end + pad))
    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["timestamp_ns"])

    def part(topic: str, cols: list[str]) -> pd.DataFrame:
        sel = {c: c.rsplit(".", 1)[-1] for c in raw.columns if c.startswith(topic + ".")}
        sel = {c: f for c, f in sel.items() if f in cols}
        df = raw[["timestamp_ns", *sel]].rename(columns={"timestamp_ns": "t_ns", **sel})
        # rows of other topics have all of this topic's fields null; a NaN gas column (SAW) must survive
        df = df.dropna(subset=list(sel.values()), how="all").drop_duplicates("t_ns").reset_index(drop=True)
        return df.drop(columns=[c for c in sel.values() if df[c].isna().all() and c not in cols[:1]])

    def part_or_none(topic, cols):
        if not any(c.startswith(topic + ".") for c in raw.columns):
            return None
        return part(topic, cols)

    arc = part(TOPIC_ARC, list(ARC_COLS))
    trv = part_or_none(TOPIC_TRAVEL, list(TRAVEL_COLS))
    thm = part_or_none(TOPIC_THERMAL, list(THERMAL_COLS))
    return assemble_signals(arc, trv, thm, t0_ns, t0_s, t1_s)


def get_signals(seq: str, t0_s: Optional[float] = None, t1_s: Optional[float] = None) -> pd.DataFrame:
    """Time-synced signals @ARC_HZ: columns t_s (seconds from arc start) + SIGNALS."""
    return offline_signals(seq, t0_s, t1_s) if is_offline() else _online_signals(seq, t0_s, t1_s)


# ------------------------------------------------------------------ WPS compliance

WPS_PARAMS = ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm", "gas_lpm", "heat_input_kj_mm",
              "current2_a", "voltage2_v", "wfs2_mpm")   # *2*: trail electrode of tandem processes


def _runs(mask: np.ndarray, t: np.ndarray, min_duration_s: float, gap_s: float = 0.0):
    """Contiguous True runs as (i0, i1) index pairs (inclusive), merging gaps < gap_s."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out = []
    s = p = idx[0]
    for i in idx[1:]:
        if t[i] - t[p] > max(gap_s, 1.5 / ARC_HZ):
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return [(a, b) for a, b in out if t[b] - t[a] + 1.0 / ARC_HZ >= min_duration_s]


def _arc_window(meta: dict, df: pd.DataFrame) -> tuple[float, float]:
    arc_on_s = float(meta.get("arc_on_s", df["t_s"].iloc[-1]))
    return 0.5, arc_on_s - 1.0      # skip hot-start ramp and programmed crater fill


def _has_range(wps: dict, p: str) -> bool:
    r = wps.get(p)
    return (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r)
            and all(np.isfinite(r)))


def violations_from_signals(df: pd.DataFrame, meta: dict, min_duration_s: float = 0.2) -> list[dict]:
    """WPS excursions of the 0.5 s averages. Parameters whose WPS range or signal is missing (no gas on SAW,
    no trail electrode on single-wire passes, ...) are skipped, not reported."""
    wps = meta.get("wps", {})
    if df.empty or not wps:
        return []
    avg = avg_signals(df)
    t = avg["t_s"].to_numpy()
    a0, a1 = _arc_window(meta, df)
    cur = avg["current_a"].to_numpy(dtype=float)
    on = (t >= a0) & (t <= a1) & (np.nan_to_num(cur, nan=0.0) > 10)
    out = []
    for p in WPS_PARAMS:
        if not _has_range(wps, p) or p not in avg or not avg[p].notna().any():
            continue
        lo, hi = (float(v) for v in wps[p])
        x = avg[p].to_numpy(dtype=float)
        for kind, mask in (("high", on & (x > hi)), ("low", on & (x < lo))):
            for i0, i1 in _runs(mask, t, min_duration_s):
                seg = x[i0:i1 + 1]
                out.append({"param": p, "t_start_s": round(float(t[i0]), 2), "t_end_s": round(float(t[i1]), 2),
                            "value": round(float(seg.max() if kind == "high" else seg.min()), 3),
                            "limit_lo": lo, "limit_hi": hi, "kind": kind})
    if "interpass_c" in avg and avg["interpass_c"].notna().any() and isinstance(wps.get("max_interpass_c"), (int, float)):
        x = avg["interpass_c"].to_numpy()
        lim = float(wps["max_interpass_c"])
        for i0, i1 in _runs(x > lim, t, min_duration_s):
            out.append({"param": "interpass_c", "t_start_s": round(float(t[i0]), 2),
                        "t_end_s": round(float(t[i1]), 2), "value": round(float(x[i0:i1 + 1].max()), 1),
                        "limit_lo": None, "limit_hi": lim, "kind": "high"})
    out.sort(key=lambda v: (v["t_start_s"], v["param"]))
    return out


def wps_violations(seq: str, min_duration_s: float = 0.2) -> list[dict]:
    """WPS excursions (0.5 s monitor averages, arc-on only):
    [{param, t_start_s, t_end_s, value, limit_lo, limit_hi, kind}]"""
    return violations_from_signals(get_signals(seq), get_meta(seq), min_duration_s)


# ------------------------------------------------------------------ query by physical value


def _anomaly_rows(seq, param, field, intervals, df, meta, gt, lt, source):
    rows = []
    a0, a1 = _arc_window(meta, df) if param != "interpass_c" else (-1e9, 1e9)
    avg = avg_signals(df)
    t = avg["t_s"].to_numpy()
    x = avg[param].to_numpy() if param in avg else None
    for s0, s1 in intervals:
        s0, s1 = max(s0, a0), min(s1, a1)
        if s1 < s0:
            continue
        peak = None
        if x is not None:
            m = (t >= s0 - 1e-6) & (t <= s1 + 1e-6)
            if m.any():
                peak = float(x[m].max()) if gt is not None and lt is None else float(x[m].min()) \
                    if lt is not None and gt is None else float(np.median(x[m]))
        rows.append({"sequence_name": seq, "param": param, "field": field, "t_start_s": round(s0, 2),
                     "t_end_s": round(s1, 2), "duration_s": round(s1 - s0, 2),
                     "peak": None if peak is None else round(peak, 3), "gt": gt, "lt": lt,
                     "process": meta.get("process"), "spool_id": meta.get("spool_id"),
                     "joint_id": meta.get("joint_id"), "pass_type": meta.get("pass_type"),
                     "defect_types": meta.get("defect_types"), "rt_result": meta.get("rt_result"),
                     "source": source})
    return rows


def _normal_satisfies(meta: dict, param: str, gt, lt) -> bool:
    """True if the WPS window itself already meets the condition (e.g. gas<12 on SAW, I>245 on SAW):
    such passes are not anomalous for that threshold."""
    lim = meta.get("wps", {}).get(param)
    if not lim or param == "interpass_c":
        return False
    lo, hi = lim
    if gt is not None and lt is None:
        return lo > gt
    if lt is not None and gt is None:
        return hi < lt
    return lo > gt and hi < lt


def find_anomalies(param: str, gt: Optional[float] = None, lt: Optional[float] = None,
                   min_duration_s: float = 0.2, raw: bool = False, relative_to_wps: bool = True,
                   **filters) -> list[dict]:
    """Find time intervals where a physical signal crossed a threshold, across passes.

    ``find_anomalies("current_a", gt=250)``, ``find_anomalies("gas_lpm", lt=10, process="FCAW")``,
    ``find_anomalies("interpass_c", gt=250)``. Both ``gt`` and ``lt`` -> value in (gt, lt).
    Online: server-side ``QueryOntologyCatalog`` on the custom ontology field (0.5 s monitor
    average unless ``raw=True``) + ``QuerySequence`` metadata filters, intervals from
    ``topic.clusterize()``; offline: the same computation over local parquet files.
    ``relative_to_wps`` skips passes whose WPS window already satisfies the condition
    (e.g. ``gas_lpm < 12`` on gasless SAW). Returns
    [{sequence_name, param, t_start_s, t_end_s, duration_s, peak, ...}].
    """
    from mosaico_io.ontology import PARAM_INDEX
    if param not in PARAM_INDEX:
        raise ValueError(f"unknown param {param!r}; one of {sorted(PARAM_INDEX)}")
    if gt is None and lt is None:
        raise ValueError("give gt= and/or lt=")
    topic, cls, avg_field = PARAM_INDEX[param]
    field = param if raw else avg_field
    out: list[dict] = []
    gap_s = 0.25
    if is_offline():
        for row in list_passes(**filters):
            seq = row["sequence_name"]
            meta = load_meta(seq)
            if relative_to_wps and _normal_satisfies(meta, param, gt, lt):
                continue
            df = offline_signals(seq)
            if param not in df or not df[param].notna().any():
                continue                      # signal not recorded for this pass (e.g. gas on SAW)
            src = df if raw else avg_signals(df)
            x, t = src[param].to_numpy(dtype=float), src["t_s"].to_numpy()
            mask = np.ones_like(x, dtype=bool)
            if gt is not None:
                mask &= x > gt
            if lt is not None:
                mask &= x < lt
            iv = [(float(t[a]), float(t[b])) for a, b in _runs(mask, t, 0.0, gap_s)]
            rows = _anomaly_rows(seq, param, field, iv, df, meta, gt, lt, "local")
            out += [r for r in rows if r["duration_s"] + 1.0 / ARC_HZ >= min_duration_s]
        return out

    from mosaicolabs import Query, QueryOntologyCatalog, QuerySequence, QueryTopic
    client = connect()
    q_field = getattr(cls.Q, field)
    if gt is not None and lt is not None:
        expr = q_field.between([gt, lt])
    elif gt is not None:
        expr = q_field.gt(float(gt))
    else:
        expr = q_field.lt(float(lt))
    parts = [QueryTopic().with_name(topic), QueryOntologyCatalog(expr)]
    parsed = [_split_filter(k, v) for k, v in filters.items() if v is not None]
    if parsed:
        qs = QuerySequence()
        for k, op, v in parsed:
            qs = qs.with_name_match(v) if (k == "sequence_name" and op == "match") else \
                qs.with_name(v) if k == "sequence_name" else qs.with_user_metadata(k, **{op: v})
        parts.insert(0, qs)
    resp = client.query(query=Query(*parts))
    for item in (resp or []):
        seq = item.sequence.name
        meta = get_meta(seq)
        if "spool_id" not in meta or (relative_to_wps and _normal_satisfies(meta, param, gt, lt)):
            continue
        t0_ns = int(meta.get("t0_ns", 0))
        iv = []
        for tp in item.topics:
            if tp.name != topic:
                continue
            for cl in tp.clusterize(int(gap_s * 1e9)):
                iv.append(((cl.timerange.start - t0_ns) / 1e9, (cl.timerange.end - t0_ns) / 1e9))
        if not iv:
            continue
        df = _online_signals(seq)
        rows = _anomaly_rows(seq, param, field, sorted(iv), df, meta, gt, lt, "mosaico")
        out += [r for r in rows if r["duration_s"] + 1.0 / ARC_HZ >= min_duration_s]
    out.sort(key=lambda r: (r["sequence_name"], r["t_start_s"]))
    return out


# ------------------------------------------------------------------ CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description="Query weld passes (Mosaico or offline)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    for k in ("process", "spool_id", "joint_id", "pass_type", "wps_id", "rt_result", "defect_types", "source_id"):
        pl.add_argument(f"--{k}")
    pl.add_argument("--has_defect", choices=["true", "false"])
    pa = sub.add_parser("anomalies")
    pa.add_argument("param")
    pa.add_argument("--gt", type=float)
    pa.add_argument("--lt", type=float)
    pa.add_argument("--process")
    pa.add_argument("--raw", action="store_true")
    pv = sub.add_parser("violations")
    pv.add_argument("sequence")
    ps = sub.add_parser("signals")
    ps.add_argument("sequence")
    ps.add_argument("--t0", type=float)
    ps.add_argument("--t1", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        f = {k: getattr(a, k) for k in ("process", "spool_id", "joint_id", "pass_type", "wps_id", "rt_result",
                                        "defect_types", "source_id") if getattr(a, k)}
        if a.has_defect:
            f["has_defect"] = a.has_defect == "true"
        for r in list_passes(**f):
            hi = r.get("heat_input_mean_kj_mm")
            print(f"{r['sequence_name']:<22} {r.get('source_id') or '-':<13} {r['process']:<5} {r['pass_type'] or '-':<5} "
                  f"HI={'%.2f' % hi if isinstance(hi, (int, float)) else '  - '} "
                  f"defect={r['defect_types'] or '-':<16} RT={r['rt_result']}")
    elif a.cmd == "anomalies":
        f = {"process": a.process} if a.process else {}
        for r in find_anomalies(a.param, gt=a.gt, lt=a.lt, raw=a.raw, **f):
            print(json.dumps(r, default=json_default))
    elif a.cmd == "violations":
        for v in wps_violations(a.sequence):
            print(json.dumps(v, default=json_default))
    else:
        print(get_signals(a.sequence, a.t0, a.t1).describe().T.to_string())
    print(f"[{'offline' if is_offline() else 'mosaicod'}]", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
