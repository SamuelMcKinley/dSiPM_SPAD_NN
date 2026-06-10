#!/usr/bin/env python3
"""Aggregate photon_stats.csv files written by simSPADs.py.

No pandas. Reads per-job CSVs from parallel tensor jobs and writes the
photon_study_master.csv / photon_study_summary.csv files expected by plot.py.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

MASTER_FIELDS = [
    "energy", "event", "spad_um",
    "n_raw", "n_lost_qe", "n_after_qe",
    "n_lost_dt", "n_final",
    "frac_lost_qe_pct", "frac_lost_dt_pct", "frac_final_pct",
    "mean_lam_nm", "mean_eta", "n_oob",
]

SUMMARY_FIELDS = [
    "energy", "spad_um", "n_events",
    "mean_n_raw", "mean_n_lost_qe", "mean_n_after_qe",
    "mean_n_lost_dt", "mean_n_final",
    "mean_frac_lost_qe_pct", "mean_frac_lost_dt_pct", "mean_frac_final_pct",
    "mean_lam_nm", "mean_eta",
]


def iter_stat_files(input_roots):
    for root in input_roots:
        root = os.path.abspath(root)
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if ".bad_" not in d]
            if ".bad_" in os.path.basename(dirpath):
                continue
            if "photon_stats.csv" in filenames:
                yield os.path.join(dirpath, "photon_stats.csv")


def as_float(value, default=math.nan):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_records(input_roots):
    records = []
    seen = set()
    for path in iter_stat_files(input_roots):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                tensor_file = row.get("tensor_file", "")
                source_root = row.get("source_root", "")
                key = (tensor_file, source_root, row.get("spad_um", ""), row.get("channel_um", ""), row.get("event", ""))
                if key in seen:
                    continue
                seen.add(key)

                raw = as_int(row.get("n_raw"))
                n_lost_qe = as_int(row.get("n_lost_qe"))
                n_lost_dt = as_int(row.get("n_lost_dt"))
                n_final = as_int(row.get("n_final"))
                rec = {
                    "energy": as_float(row.get("energy")),
                    "event": as_int(row.get("event")),
                    "spad_um": as_float(row.get("spad_um")),
                    "n_raw": raw,
                    "n_lost_qe": n_lost_qe,
                    "n_after_qe": as_int(row.get("n_after_qe")),
                    "n_lost_dt": n_lost_dt,
                    "n_final": n_final,
                    "frac_lost_qe_pct": round(n_lost_qe / raw * 100.0, 3) if raw > 0 else 0.0,
                    "frac_lost_dt_pct": round(n_lost_dt / raw * 100.0, 3) if raw > 0 else 0.0,
                    "frac_final_pct": round(n_final / raw * 100.0, 3) if raw > 0 else 0.0,
                    "mean_lam_nm": as_float(row.get("mean_lam_nm")),
                    "mean_eta": as_float(row.get("mean_eta")),
                    "n_oob": as_int(row.get("n_oob")),
                }
                if math.isnan(rec["energy"]) or math.isnan(rec["spad_um"]):
                    continue
                records.append(rec)
    records.sort(key=lambda r: (r["energy"], r["spad_um"], r["event"]))
    return records


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def mean_non_nan(values):
    vals = [v for v in values if not math.isnan(v)]
    return mean(vals) if vals else math.nan


def fmt(value, ndigits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return round(float(value), ndigits)


def write_outputs(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, "photon_study_master.csv")
    summary_path = os.path.join(output_dir, "photon_study_summary.csv")

    with open(master_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["mean_lam_nm"] = fmt(row["mean_lam_nm"], 2)
            row["mean_eta"] = fmt(row["mean_eta"], 5)
            writer.writerow(row)

    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["energy"], rec["spad_um"])].append(rec)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for (energy, spad), recs in sorted(grouped.items()):
            raw = mean(r["n_raw"] for r in recs)
            lost_qe = mean(r["n_lost_qe"] for r in recs)
            lost_dt = mean(r["n_lost_dt"] for r in recs)
            final = mean(r["n_final"] for r in recs)
            writer.writerow({
                "energy": energy,
                "spad_um": spad,
                "n_events": len(recs),
                "mean_n_raw": fmt(raw),
                "mean_n_lost_qe": fmt(lost_qe),
                "mean_n_after_qe": fmt(mean(r["n_after_qe"] for r in recs)),
                "mean_n_lost_dt": fmt(lost_dt),
                "mean_n_final": fmt(final),
                "mean_frac_lost_qe_pct": fmt(lost_qe / raw * 100.0 if raw > 0 else 0.0, 2),
                "mean_frac_lost_dt_pct": fmt(lost_dt / raw * 100.0 if raw > 0 else 0.0, 2),
                "mean_frac_final_pct": fmt(final / raw * 100.0 if raw > 0 else 0.0, 2),
                "mean_lam_nm": fmt(mean_non_nan(r["mean_lam_nm"] for r in recs), 4),
                "mean_eta": fmt(mean_non_nan(r["mean_eta"] for r in recs), 4),
            })

    return master_path, summary_path, len(grouped)


def main():
    parser = argparse.ArgumentParser(description="Aggregate parallel simSPAD photon stats without ROOT or pandas.")
    parser.add_argument("output_dir", help="Directory to write photon_study_master.csv and photon_study_summary.csv")
    parser.add_argument("input_roots", nargs="+", help="SPAD tensor roots such as /.../SPAD_results/predict_10x10")
    args = parser.parse_args()

    records = load_records(args.input_roots)
    if not records:
        raise SystemExit("No photon_stats.csv records found. Re-run SPAD tensor jobs with the updated simSPADs.py.")

    master_path, summary_path, n_groups = write_outputs(records, args.output_dir)
    print(f"Aggregated {len(records)} photon event records into {n_groups} energy/SPAD groups")
    print(f"Master:  {master_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
