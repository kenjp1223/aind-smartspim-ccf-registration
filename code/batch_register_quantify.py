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
import re
import shutil
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


# ---------------------------------------------------------------------------
# progress display
# ---------------------------------------------------------------------------
# share of one sample's time spent in each registration stage (rough, from test runs)
STAGE_SHARE = [("loading atlas", 0.02), ("copy in", 0.03), ("preprocess", 0.05), ("rigid", 0.05),
               ("affine", 0.07), ("SyN", 0.58), ("warp + labels", 0.15), ("copy out", 0.05)]
RE_ITERS = re.compile(r"^\s*iterations = ([\dx]+)")
RE_SHRINK = re.compile(r"Shrink factors \(level (\d+) out of (\d+)\): \[([\d, ]+)\]")
RE_DIAG = re.compile(r"^\s*\d*DIAGNOSTIC,\s+(\d+),")
RE_PLANES = re.compile(r"planes (\d+)/(\d+)")


class StepWatcher:
    """Follows a step's log as it grows and works out where the step is."""

    def __init__(self, kind):
        self.kind = kind            # "register" or "quantify"
        self.stage = "loading atlas"
        self.detail = ""
        self.iters, self.shrink = [], {}
        self.level = 0
        self.it = 0
        self.frac_in_stage = 0.0
        self.reused = False

    def feed(self, line):
        if self.kind == "quantify":
            m = RE_PLANES.search(line)
            if m:
                self.stage, self.frac_in_stage = "reading seg", int(m.group(1)) / int(m.group(2))
                self.detail = f"planes {m.group(1)}/{m.group(2)}"
            elif "region rows" in line:
                self.stage, self.detail = "region tables", line.split(": ")[-1].strip()
            elif "-> Allen nrrd layout" in line or "-> output grid" in line:
                self.stage, self.detail = "atlas grids", ""
            return
        m = RE_ITERS.match(line)
        if m:
            self.iters = [int(x) for x in m.group(1).split("x")]
            self.shrink = {}
        m = RE_SHRINK.search(line)
        if m:
            f = [int(x) for x in m.group(3).split(",")]
            self.shrink[int(m.group(1))] = f[0] * f[1] * f[2] if len(f) == 3 else f[0] ** 3
        if "(network path)" in line and "Copying" in line:
            self._set("copy in")
        elif "Resample OMEZarr" in line:
            self._set("preprocess")
        elif "*** Running Euler3DTransform" in line:
            self._set("rigid")
        elif "*** Running AffineTransform" in line and self.stage in ("loading atlas", "copy in", "preprocess", "rigid"):
            self._set("affine")
        elif "*** Running SyN" in line:
            self._set("SyN")
        elif "Reusing the transforms" in line:
            self.reused = True
            self._set("warp + labels")
        elif "Elapsed time (stage 1)" in line and self.stage == "SyN":
            self._set("warp + labels")
        elif "Copying results to" in line:
            self._set("copy out")
        elif "XXDIAGNOSTIC" in line:
            self.level += 1
            self.it = 0
        else:
            m = RE_DIAG.match(line)
            if m:
                self.it = int(m.group(1))
        if self.stage in ("rigid", "affine", "SyN") and self.iters and self.level:
            n = len(self.iters)
            k = min(self.level, n)
            # cost per iteration ~ voxels at that level ~ 1 / shrink^3
            w = [self.iters[i] / self.shrink.get(i + 1, 1) for i in range(n)]
            done = sum(w[: k - 1]) + w[k - 1] * min(self.it / max(self.iters[k - 1], 1), 1.0)
            self.frac_in_stage = done / max(sum(w), 1e-9)
            self.detail = f"level {k}/{n}, iteration {self.it}/{self.iters[k - 1]}"

    def _set(self, stage):
        self.stage, self.level, self.it, self.frac_in_stage, self.detail = stage, 0, 0, 0.0, ""

    def fraction(self):
        if self.kind == "quantify":
            return self.frac_in_stage
        names = [n for n, _ in STAGE_SHARE]
        i = names.index(self.stage) if self.stage in names else 0
        return sum(sh for _, sh in STAGE_SHARE[:i]) + STAGE_SHARE[i][1] * self.frac_in_stage


def fmt_time(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    return f"{h}:{r // 60:02d}:{r % 60:02d}" if h else f"{r // 60:02d}:{r % 60:02d}"


class Progress:
    """One status line, redrawn in place: overall bar, sample, stage, times."""

    def __init__(self, n_samples, enabled):
        self.n = n_samples
        self.enabled = enabled
        self.t0 = time.time()
        self.durations = []          # seconds per sample that really ran a registration
        self.last_len = 0

    def show(self, i, name, step, watcher, t_sample):
        if not self.enabled:
            return
        frac_sample = watcher.fraction() if watcher else 0.0
        overall = (i - 1 + frac_sample) / max(self.n, 1)
        width = 24
        bar = "#" * int(round(overall * width)) + "-" * (width - int(round(overall * width)))
        elapsed = time.time() - t_sample
        if self.durations:
            per = sum(self.durations) / len(self.durations)
        elif frac_sample > 0.05:
            per = elapsed / frac_sample
        else:
            per = None
        eta = ""
        if per:
            left = max(per - elapsed, 0) + per * (self.n - i)
            eta = f" | batch left ~{fmt_time(left)}"
        stage = f"{step}: {watcher.stage}" if watcher else step
        detail = f" ({watcher.detail})" if watcher and watcher.detail else ""
        short = name.replace("_Destripe_DONE", "")
        line = (f"[{bar}] {overall * 100:3.0f}% | {i}/{self.n} {short} | {stage}{detail}"
                f" | {fmt_time(elapsed)}{eta}")
        cols = shutil.get_terminal_size((120, 20)).columns - 1
        line = line[:cols]
        sys.stdout.write("\r" + line + " " * max(self.last_len - len(line), 0))
        sys.stdout.flush()
        self.last_len = len(line)

    def clear(self):
        if self.enabled and self.last_len:
            sys.stdout.write("\r" + " " * self.last_len + "\r")
            sys.stdout.flush()
            self.last_len = 0


def run(cmd, log_path, progress=None, i=0, name="", step="", kind="register"):
    """
    Run a step in its own process (memory is freed between samples). Its
    output goes to log_path; while it runs, the log is followed to update the
    progress line.
    """
    logger.info("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    watcher = StepWatcher(kind)
    t_start = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(map(str, cmd))}\n")
        lf.flush()
        start = lf.tell()
        p = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        pos, buf = start, ""
        while True:
            rc = p.poll()
            try:
                with open(log_path, encoding="utf-8", errors="replace") as rf:
                    rf.seek(pos)
                    chunk = rf.read()
                    pos = rf.tell()
            except OSError:
                chunk = ""
            buf += chunk
            *lines, buf = buf.split("\n")
            for ln in lines:
                watcher.feed(ln)
            if progress:
                progress.show(i, name, step, watcher, t_start)
            if rc is not None:
                break
            time.sleep(1.0)
    if progress:
        progress.clear()
    return p.returncode, watcher


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
    ap.add_argument("--only", nargs="*", help="only samples whose folder name contains one of these "
                                              "(case-insensitive), e.g. --only Calcr Pdyn_NAc")
    ap.add_argument("--no-progress", action="store_true", help="no live progress line")
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
        samples = [s for s in samples if any(o.lower() in s.name.lower() for o in args.only)]
    logger.info(f"{len(samples)} sample folders under {root}"
                + (f" matching {' / '.join(args.only)}" if args.only else ""))
    for s in samples:
        logger.info(f"  {s.relative_to(root)}")
    progress = Progress(len(samples), enabled=not args.no_progress and not args.dry_run and sys.stdout.isatty())
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
            t_reg = time.time()
            rc, w = run(cmd, log_path, progress, i, s.name, "register", "register")
            row["registration"] = "done" if rc == 0 else "FAILED"
            if rc == 0 and not w.reused:
                progress.durations.append(time.time() - t_reg)
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
            rc, _ = run(cmd, log_path, progress, i, s.name, "quantify", "quantify")
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
