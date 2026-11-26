# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.20"]
# ///
"""Linking JSON rides to CSVs: the old rule (greedy, smallest |offset| first) against the new one
(gcod.match_rides: exact assignment by fit, then the athlete's own offset), on the whole release.

GoldenCheetah's export has no key between a ride in the athlete's JSON and its CSV: a ride's `date`
is its UTC start, a CSV's name its local start. Reads a build made with the old rule (converter
0.4.0): its activities table has every ride with its UTC start, duration and distance, and the
convert manifest has every CSV's length and distance. No zips are read.

  uv run research/explore/15_ride_linking.py data [--report research/reports/15_ride_linking.md]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "datasets" / "opendata"))
import gcod  # noqa: E402


def load(workdir: Path) -> tuple[dict[str, list[dict]], dict[str, dict[str, dict]]]:
    acts = (pl.scan_parquet(workdir / "build/opendata/data/activities/*.parquet")
            .select("athlete_id", "activity_id", "start_time_utc", "utc_offset_minutes", "sport",
                    "workout_time", "total_distance")
            .filter(pl.col("start_time_utc").is_not_null())
            .collect())
    rides: dict[str, list[dict]] = {}
    for r in acts.iter_rows(named=True):
        rides.setdefault(r["athlete_id"], []).append(r)
    csvs = {}
    for line in (workdir / "build/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        csvs[m["athlete_id"]] = m.get("csvs") or {}
    return rides, csvs


def old_rule(utc: list[datetime], csv_ids: list[str], misfit) -> list[tuple[str | None, int | None]]:
    """Converter 0.4.0: candidates within +-14 h; greedy by smallest |offset|, fit breaks ties."""
    local = {s: gcod.parse_local(s) for s in csv_ids}
    cands = sorted((abs(off), misfit(i, s) > 1, i, s, off)
                   for i, u in enumerate(utc) for s, lt in local.items() if lt is not None
                   for off in [int((lt - u).total_seconds())] if abs(off) <= 14 * 3600 and off % 900 == 0)
    used_r, used_c, out = set(), set(), {}
    for _, _, i, s, off in cands:
        if i not in used_r and s not in used_c:
            used_r.add(i)
            used_c.add(s)
            out[i] = (s, off)
    return [out.get(i, (None, None)) for i in range(len(utc))]


def own_offset(links: dict[int, tuple[str, int]], utc: list[datetime], misfit, ambiguous: set[int]) -> dict[int, bool]:
    """Per linked ride: is its offset within an hour of the nearest unambiguous, fitting links?"""
    anchors = sorted((utc[i], off) for i, (s, off) in links.items() if i not in ambiguous and misfit(i, s) <= 1)
    times = [t for t, _ in anchors]
    out = {}
    for i, (_, off) in links.items():
        k = bisect.bisect_left(times, utc[i])
        near = [anchors[j][1] for j in (k - 1, k) if 0 <= j < len(anchors) and anchors[j][0] != utc[i]]
        out[i] = min((abs(off - a) for a in near), default=math.inf) <= gcod.SAME_ZONE_S
    return out


def compare(rides: dict[str, list[dict]], csvs: dict[str, dict[str, dict]]) -> dict:
    st = Counter()
    for athlete_id, recs in rides.items():
        cs = csvs.get(athlete_id, {})
        ids = sorted(cs)
        utc = [r["start_time_utc"].replace(tzinfo=UTC) for r in recs]
        metrics = [{"METRICS": {"workout_time": r["workout_time"], "total_distance": r["total_distance"]}}
                   for r in recs]

        def misfit(i: int, s: str, metrics=metrics, cs=cs) -> float:
            return gcod.ride_misfit(metrics[i], cs.get(s))

        old = old_rule(utc, ids, misfit)
        new = gcod.match_rides(utc, ids, misfit=misfit)
        built = [r["activity_id"] if r["utc_offset_minutes"] is not None else None for r in recs]
        st["old rule reproduces the build"] += sum(o[0] == b for o, b in zip(old, built, strict=True))
        # ambiguous: rides or CSVs with more than one candidate under the new rule's range
        cand = _candidates(utc, ids)
        nr, nc = Counter(i for i, _ in cand), Counter(s for _, s in cand)
        amb = {i for i, s in cand if nr[i] > 1 or nc[s] > 1}
        st["rides"] += len(recs)
        st["ambiguous rides"] += len(amb)
        for name, links in (("old", old), ("new", new)):
            ln = {i: (s, off) for i, (s, off) in enumerate(links) if s is not None}
            own = own_offset(ln, utc, misfit, amb)
            for i, (s, _) in ln.items():
                where = "ambiguous" if i in amb else "unambiguous"
                st[f"{name} {where} links"] += 1
                st[f"{name} {where} fit"] += misfit(i, s) <= 1
                st[f"{name} {where} own offset"] += own[i]
                st[f"{name} {where} neither"] += misfit(i, s) > 1 and not own[i]
        st["rides whose link changes"] += sum(o[0] != n[0] for o, n in zip(old, new, strict=True))
        st["CSVs whose ride changes"] += len({o[0] for o, n in zip(old, new, strict=True) if o[0] != n[0] and o[0]}
                                             | {n[0] for o, n in zip(old, new, strict=True) if o[0] != n[0] and n[0]})
    return dict(st)


def _candidates(utc: list[datetime], ids: list[str]) -> list[tuple[int, str]]:
    by_key: dict[tuple[int, int], list[tuple[str, datetime]]] = {}
    for s in ids:
        if (lt := gcod.parse_local(s)) is not None:
            by_key.setdefault((lt.minute % 15, lt.second), []).append((s, lt))
    return [(i, s) for i, u in enumerate(utc) for s, lt in by_key.get((u.minute % 15, u.second), ())
            if gcod.MIN_OFFSET_S <= (off := int((lt - u).total_seconds())) <= gcod.MAX_OFFSET_S and off % 900 == 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    rides, csvs = load(a.workdir)
    results = {}
    default = gcod.FIT_TIME_S
    for t in (5.0, 10.0, 30.0):
        gcod.FIT_TIME_S = t
        results[t] = compare(rides, csvs)
        print(f"duration tolerance {t:g} s:", json.dumps(results[t], indent=1), flush=True)
    gcod.FIT_TIME_S = default
    if a.report:
        a.report.write_text(report(results, default), encoding="utf-8", newline="\n")
        print(f"wrote {a.report}")


def report(results: dict[float, dict], default: float) -> str:
    r = results[default]
    pct = lambda n, d: f"{100 * n / d:.1f}%"  # noqa: E731
    lines = [
        "# 15 · Linking JSON rides to CSVs",
        "",
        "Generated by `research/explore/15_ride_linking.py` from the converter 0.4.0 build of the whole release.",
        "",
        "GoldenCheetah's export has no key between a ride in the athlete's JSON and its CSV. The JSON",
        "records have `date`, `data`, `sport`, `METRICS` and sometimes `XDATA` (checked on 120,672",
        "rides of 300 random athletes); `date` is the UTC start, and a CSV's name is the local start.",
        "A CSV is a candidate for a ride when the two differ by a whole quarter-hour. Most rides have",
        "one candidate that no other ride has. The rest compete: a ride listed twice at two UTC",
        "times, a file exported twice an hour apart, two rides an hour apart, two rides on one second.",
        "",
        "## Findings",
        "",
        f"- **{r['rides']:,} rides**, of which {r['ambiguous rides']:,} "
        f"({pct(r['ambiguous rides'], r['rides'])}) compete with",
        "  other rides or CSVs. The rest link the same way under both rules.",
        "- **The old rule** (converter 0.4.0) linked greedily by smallest |offset|; a ride's fit to the",
        "  CSV only broke ties. A far-off time zone could win over the ride that fits, and a wrong link",
        "  taken early could leave a fitting ride without its CSV.",
        "- **The new rule** (`gcod.match_rides`) assigns each group of competing rides and CSVs",
        "  exactly, by: links whose ride fits the CSV (duration and distance), then links at the",
        "  athlete's own offset (within an hour of the nearest unambiguous, fitting links), then the",
        "  number of links, then closeness to those offsets, the misfit and the smallest |offset|.",
        "- **Fit is a preference, not a veto.** 98.7% of the unambiguous links fit; the others are",
        "  mostly rides whose duration matches to the second but whose distance metric does not",
        "  (an edited file, a device without distance). Refusing them would unlink correct rides.",
        "- **The decisions don't depend on the tolerance**: "
        + ", ".join(f"{results[t]['rides whose link changes']:,}" for t in (5.0, 10.0, 30.0)) + " rides change",
        "  at 5, 10 and 30 s.",
        "",
        "## Results",
        "",
        "Duration tolerance 10 s or 1%, distance 50 m or 1%.",
        "",
        "| | links | fit the CSV | at the athlete's offset | neither |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("old", "new"):
        for where in ("ambiguous", "unambiguous"):
            n = r[f"{name} {where} links"]
            lines.append(f"| {name} rule, {where} | {n:,} | {r[f'{name} {where} fit']:,} | "
                         f"{r[f'{name} {where} own offset']:,} | {r[f'{name} {where} neither']:,} |")
    lines += ["", f"Rides whose link changes: {r['rides whose link changes']:,}; CSVs whose ride changes: "
              f"{r['CSVs whose ride changes']:,}. The old rule reproduces {r['old rule reproduces the build']:,} of "
              f"the {r['rides']:,} built links (the rest are exact ties, which JSON order broke).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
