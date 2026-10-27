"""
Register and quantify every SmartSPIM sample under a folder.

For each *_Destripe_DONE folder (searched a few levels deep):
  1. registration (run_custom_atlas.py, AIND route by default) of the
     registration channel, skipped when a finished one exists;
  2. quantification (quantify_registered.py) of its seg-*.ome.zarr channels
     (or --channels), skipped when done unless --redo-quant.
A failure in one sample is logged and the batch moves on. At the end every
sample's region tables are combined into one CSV per label set:
  <root>/quantification_summary/all_<labels>_regions.csv
plus batch_status.csv (what happened to each sample) and batch.log.

Usage (from the repo's code/ folder):
  uv run python batch_register_quantify.py "\\\\10.158.246.229\\DataCommon\\SmartSPIM2\\Ken\\LINCS_series" ^
      --aind-template "<folder with the AIND template files>" ^
      --allen-dir "<folder with average_template_25.nrrd, annotation_25.nrrd>" ^
      --labels kim="<Kim_to_allen25_ann_fwdtransforms.tif>" ^
      --labels-info kim="<atlas_info_KimRef_FPbasedLabel_v4.0.csv>"
  Add --dry-run first to see what it would do.
"""

import argparse
import csv
import datetime as dt
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSFORMS = ("ls_to_template_SyN_0GenericAffine.mat", "ls_to_template_SyN_1Warp.nii.gz")
SKIP_DIRS = {"registration", "quantification", "metadata", "annotation_precomputed"}
logger = logging.getLogger("batch")


def find_samples(root, suffix, max_depth):
    out = []
    stack = [(Path(root), 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as e:
            logger.warning(f"cannot list {d}: {e}")
            continue
        for e in entries:
            try:
                if not e.is_dir():
                    continue
            except OSError:
                continue
            if e.name.endswith(suffix):
                out.append(Path(e.path))
            elif (depth + 1 < max_depth and not e.name.endswith(".zarr") and not e.name.startswith("registration_")
                  and e.name not in SKIP_DIRS and not e.name.lower().startswith(("ex_", "seg-"))):
                stack.append((Path(e.path), depth + 1))
    return sorted(out)


def run(cmd, log_path):
    """Run a step in its own process (memory is freed between samples); output to the log."""
    logger.info("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(map(str, cmd))}\n")
        lf.flush()
        p = subprocess.run([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    return p.returncode


def last_error(log_path, n=3):
    try:
        lines = [x.rstrip() for x in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return " | ".join(lines[-n:])[-400:]
    except OSError:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing the samples (searched --max-depth levels)")
    ap.add_argument("--aind-template", help="AIND template folder (AIND route, recommended)")
    ap.add_argument("--register-args", nargs=argparse.REMAINDER, default=[],
                    help="instead of --aind-template: arguments passed to run_custom_atlas.py "
                         "(must come last), e.g. --atlas-template X --atlas-labels Y --atlas-res 25")
    ap.add_argument("--reg-channel", default="Ex_488_Ch0_stitched.ome.zarr",
                    help="channel to register (default: %(default)s)")
    ap.add_argument("--allen-dir", help="folder with average_template_25.nrrd and annotation_25.nrrd")
    ap.add_argument("--labels", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--labels-info", nargs="*", default=[], metavar="NAME=CSV")
    ap.add_argument("--channels", nargs="+", help="zarrs to quantify (default: all seg-*.ome.zarr)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sample-suffix", default="_Destripe_DONE")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="only samples whose folder name contains one of these")
    ap.add_argument("--redo-registration", action="store_true")
    ap.add_argument("--redo-quant", action="store_true")
    ap.add_argument("--no-quant", action="store_true", help="register only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.aind_template and not args.register_args:
        raise SystemExit("Give --aind-template DIR (or --register-args ... for a custom atlas)")
    root = Path(args.root.rstrip("/\\"))
    summ = root / "quantification_summary"
    summ.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(summ / "batch.log", encoding="utf-8")])
    reg_name = "ccf25_aind" if args.aind_template else None
    ch_stem = args.reg_channel.replace(".ome.zarr", "").replace(".zarr", "")

    samples = find_samples(root, args.sample_suffix, args.max_depth)
    if args.only:
        samples = [s for s in samples if any(o in s.name for o in args.only)]
    logger.info(f"{len(samples)} sample folders under {root}")
    status = []
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        row = {"sample": s.name, "path": str(s), "registration": "", "quantification": "", "seconds": 0, "note": ""}
        logger.info(f"[{i}/{len(samples)}] {s.name}")
        zarr = s / args.reg_channel
        if not zarr.is_dir():
            row.update(registration="skipped", note=f"no {args.reg_channel}")
            logger.info(f"  skipped: no {args.reg_channel}")
            status.append(row)
            continue
        if reg_name is None:  # custom atlas: same folder naming as run_custom_atlas.py
            tmpl = args.register_args[args.register_args.index("--atlas-template") + 1]
            reg_name = Path(tmpl).stem.split(".")[0]
        reg_dir = s / f"registration_{reg_name}" / ch_stem
        log_path = s / f"registration_{reg_name}" / "batch_steps.log"
        done = (reg_dir / "annotation_precomputed" / "info").is_file() and all((reg_dir / t).is_file() for t in TRANSFORMS)

        # 1) registration
        if done and not args.redo_registration:
            row["registration"] = "existing"
        elif args.dry_run:
            row["registration"] = "would run"
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, HERE / "run_custom_atlas.py", zarr]
            cmd += ["--aind-template", args.aind_template] if args.aind_template else args.register_args
            if args.redo_registration:
                cmd.append("--redo")
            rc = run(cmd, log_path)
            row["registration"] = "done" if rc == 0 else "FAILED"
            if rc:
                row["note"] = last_error(log_path)
                logger.error(f"  registration failed: {row['note']}")

        # 2) quantification
        segs = args.channels or sorted(e.name for e in s.iterdir()
                                       if e.is_dir() and e.name.lower().startswith("seg-") and e.name.endswith(".zarr"))
        qdone = (reg_dir / "quantification" / "quantification.json").is_file()
        if args.no_quant:
            row["quantification"] = "not requested"
        elif not segs:
            row["quantification"] = "skipped"
            row["note"] = (row["note"] + " no seg-*.zarr").strip()
        elif row["registration"] == "FAILED":
            row["quantification"] = "skipped"
        elif qdone and not args.redo_quant:
            row["quantification"] = "existing"
        elif args.dry_run:
            row["quantification"] = f"would run: {', '.join(segs)}"
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, HERE / "quantify_registered.py", s,
                   "--registration", f"registration_{reg_name}/{ch_stem}", "--threshold", args.threshold,
                   "--channels", *segs]
            if args.allen_dir:
                cmd += ["--allen-dir", args.allen_dir]
            if args.labels:
                cmd += ["--labels", *args.labels]
            if args.labels_info:
                cmd += ["--labels-info", *args.labels_info]
            rc = run(cmd, log_path)
            row["quantification"] = "done" if rc == 0 else "FAILED"
            if rc:
                row["note"] = (row["note"] + " " + last_error(log_path)).strip()
                logger.error(f"  quantification failed: {last_error(log_path)}")
        row["seconds"] = round(time.time() - t0)
        logger.info(f"  registration: {row['registration']}, quantification: {row['quantification']} "
                    f"({row['seconds']} s)")
        status.append(row)

    with open(summ / "batch_status.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(status[0]) if status else ["sample"])
        w.writeheader()
        w.writerows(status)

    # combine region tables across samples, one file per label set
    combined = {}
    for s in samples:
        q = s / f"registration_{reg_name}" / ch_stem / "quantification" if reg_name else None
        if not q or not q.is_dir():
            continue
        for p in sorted(q.glob("*_regions.csv")):
            labels = p.stem.rsplit("_", 2)[-2]   # <channel>_<labels>_regions
            with open(p, newline="", encoding="utf-8") as f:
                combined.setdefault(labels, []).extend(csv.DictReader(f))
    for labels, rows in combined.items():
        out = summ / f"all_{labels}_regions.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"wrote {out} ({len(rows)} rows)")
    n_fail = sum("FAILED" in (r["registration"] + r["quantification"]) for r in status)
    logger.info(f"Finished: {len(status)} samples, {n_fail} with failures. Status: {summ / 'batch_status.csv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
