"""
Region quantification in each brain's own space: atlas labels are warped onto
the brain, the signal is never resampled.

Why this direction: warping the signal into the atlas interpolates it (each
25 um atlas voxel gets a blend of neighbouring values), so sums and counts are
not preserved. Warping the labels instead (nearest-neighbour, "genericLabel")
keeps every signal voxel as acquired.

Steps, per registered sample (run_custom_atlas.py, AIND route):
  1. label volume in CCF 25 um space (e.g. Kim_to_allen25_ann_fwdtransforms.tif,
     same grid as AIND's ccf_average_template_25.nii.gz, any axis order)
     -> AIND SmartSPIM template (AIND's fixed CCF -> template transform)
     -> this brain's registration grid (inverse of the brain -> template
     registration). Labels are never averaged. The template-space step is the
     same for every brain and is cached.
  2. every full-resolution voxel gets the label of the registration-grid voxel
     that contains it (the registration grid, 16 um here, is finer than the
     25 um atlas, so no label detail is lost).
  3. per region (and hemisphere):
       intensity channels: sum of voxel values, voxel count, mean
       seg-* channels:     count of voxels with probability >= --threshold,
                           voxel count, fraction
     all at the channel's finest level (level 0).

Outputs in <registration folder>/sample_space_<labels>/:
  <labels>_labels_regrid.tif     labels on the registration grid (z, y, x of the
                                 zarr level used for registration), atlas IDs
  hemisphere_regrid.tif          1 = left, 2 = right (same grid)
  <channel>_regions.csv          one row per region x hemisphere (left/right/both)
  quantification.json, .log
and with --summary DIR, all samples combined: DIR/<labels>_<kind>_regions.csv

Level-0 voxel (z, y, x) has label regrid[z // fz, y // fy, x // fx], with
(fz, fy, fx) = registration voxel / level-0 voxel (written to quantification.json).

Usage (from the repo's code/ folder):
  uv run python quantify_sample_space.py "<sample folder>" ["<sample folder>" ...] ^
      --labels "kim=<Kim_to_allen25_ann_fwdtransforms.tif>" ^
      --labels-info "kim=<atlas_info_KimRef_FPbasedLabel_v4.0.csv>" ^
      --channels Ex_639_Ch2_stitched.ome.zarr seg-Ex_639_Ch2_stitched.ome.zarr ^
      --summary "<folder for combined CSVs>"
  First time: add --check-transforms (needs annotation_25.nrrd and
  average_template_25.nrrd next to the AIND files) to verify the CCF -> template
  step against AIND's own template-space annotation.
"""

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from quantify_registered import (
    LevelReader,
    apply_orient,
    block_factors,
    dice_mask,
    is_network,
    match_orientation,
    ome_levels,
    read_any,
    read_names,
    small_int,
)

logger = logging.getLogger("sample_space")
LS_AFFINE = "ls_to_template_SyN_0GenericAffine.mat"
LS_INVERSE_WARP = "ls_to_template_SyN_1InverseWarp.nii.gz"


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
    for name in ("sample_space", "quantify"):
        lg = logging.getLogger(name)
        lg.disabled = False
        lg.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# atlas side (same for every brain)
# --------------------------------------------------------------------------
def lr_axis_of(ccf_img):
    """
    Left/right axis of the CCF grid = the axis the brain is mirror-symmetric
    across; which end is left comes from the NIfTI header (ITK physical
    space is LPS: +x = left).
    """
    a = ccf_img.numpy()
    sub = a[::2, ::2, ::2]
    scores = []
    for ax in range(3):
        f = np.flip(sub, ax)
        x, y = sub.ravel() - sub.mean(), f.ravel() - f.mean()
        scores.append(float(x @ y / np.sqrt((x @ x) * (y @ y))))
    lr = int(np.argmax(scores))
    d = np.asarray(ccf_img.direction)[0, lr]   # physical x (LPS: +left) per index step
    high_is_left = bool(d > 0)
    logger.info(f"CCF grid: left/right axis {lr} (mirror symmetry {[round(s, 3) for s in scores]}); "
                f"{'higher' if high_is_left else 'lower'} index = left (header direction x {d:+.2f})")
    if abs(d) < 0.5:
        logger.warning("header does not say clearly which side is left; hemisphere names may be swapped")
    return lr, high_is_left


def hemisphere_volume(shape, lr, high_is_left):
    """1 = left, 2 = right."""
    idx = np.arange(shape[lr])
    left = idx >= shape[lr] // 2 if high_is_left else idx < shape[lr] // 2
    side = np.where(left, 1, 2).astype(np.uint8)
    view = [1, 1, 1]
    view[lr] = shape[lr]
    return np.broadcast_to(side.reshape(view), shape).copy()


def labels_on_ccf_grid(path, ccf_img):
    """Label file -> array in the CCF NIfTI's own axis order (found by brain-mask overlap)."""
    arr = small_int(read_any(path, labels=True))
    ref = ccf_img.numpy()
    brain = ref > np.percentile(ref[ref > 0], 20)
    res = match_orientation(arr, brain.astype(np.uint8), dice_mask)
    best, nxt = res[0], (res[1] if len(res) > 1 else (float("nan"), None, None))
    logger.info(f"{Path(path).name} {arr.shape} -> CCF grid {ref.shape}: axes {best[1]}, flips {best[2]}, "
                f"brain-mask Dice {best[0]:.3f} (next {nxt[0]:.3f}, axes {nxt[1]}, flips {nxt[2]})")
    if best[0] < 0.8:
        raise SystemExit(f"{path} does not overlap the CCF template brain (Dice {best[0]:.2f})")
    if best[0] - nxt[0] < 0.005:
        logger.warning("the next orientation fits almost as well (typically the left/right mirror); "
                       "kept the one with the fewest flips")
    return np.ascontiguousarray(apply_orient(arr, best[1], best[2])), {
        "perm": list(best[1]), "flips": list(best[2]), "dice": best[0], "next": nxt[0]}


def compact(arr):
    ids, inv = np.unique(arr.ravel(), return_inverse=True)
    if ids[0] != 0:
        ids = np.concatenate([[0], ids])
        inv = inv + 1
    return inv.reshape(arr.shape).astype(np.float32), ids


def ccf_to_template(img_ccf, reg):
    """CCF-space label image -> SmartSPIM template space (AIND's reverse transform, labels kept)."""
    import ants

    tmpl = ants.image_read(reg["template_path"])
    c2t = list(reg["ccf_to_template_transform_path"])
    if len(c2t) != 2 or "Inverse" not in Path(c2t[1]).name or not Path(c2t[1]).is_file():
        raise SystemExit(f"Need AIND's inverse warp (spim_template_to_ccf_syn_1InverseWarp_25.nii.gz) next to the "
                         f"other AIND files; the registration used {c2t}. Download it and re-run "
                         f"run_custom_atlas.py --aind-template (registration is reused).")
    return ants.apply_transforms(fixed=tmpl, moving=img_ccf, transformlist=c2t,
                                 whichtoinvert=[True, False], interpolator="genericLabel")


def template_space_labels(name, label_path, reg, cache_dir):
    """(labels in template space as compact float image, id table, hemisphere image, info), cached."""
    import ants

    ccf_img = ants.image_read(reg["ccf_reference_path"])
    st = Path(label_path).stat()
    key = hashlib.sha1(f"{Path(label_path).resolve()}|{st.st_size}|{st.st_mtime}|"
                       f"{reg['ccf_reference_path']}|{reg['ccf_to_template_transform_path']}".encode()).hexdigest()[:12]
    cdir = Path(cache_dir) / f"{name}_{key}"
    files = [cdir / "labels_template.nii.gz", cdir / "hemisphere_template.nii.gz", cdir / "info.json"]
    if all(f.is_file() for f in files):
        info = json.loads(files[2].read_text())
        logger.info(f"{name}: template-space labels from cache {cdir}")
        return ants.image_read(str(files[0])), np.array(info["ids"], np.uint64), ants.image_read(str(files[1])), info
    t0 = time.time()
    arr, orient = labels_on_ccf_grid(label_path, ccf_img)
    lr, high_is_left = lr_axis_of(ccf_img)
    comp, ids = compact(arr)
    hemi = hemisphere_volume(arr.shape, lr, high_is_left).astype(np.float32)
    hemi[arr == 0] = 0
    hdr = dict(origin=ccf_img.origin, spacing=ccf_img.spacing, direction=ccf_img.direction)
    lab_t = ccf_to_template(ants.from_numpy(comp, **hdr), reg)
    hemi_t = ccf_to_template(ants.from_numpy(hemi, **hdr), reg)
    cdir.mkdir(parents=True, exist_ok=True)
    ants.image_write(lab_t, str(files[0]))
    ants.image_write(hemi_t, str(files[1]))
    info = {"label_file": str(label_path), "orientation_to_ccf": orient, "lr_axis": lr,
            "higher_index_is_left": high_is_left, "ids": [int(i) for i in ids]}
    files[2].write_text(json.dumps(info))
    logger.info(f"{name}: {len(ids) - 1} labels warped CCF -> template in {time.time() - t0:.0f} s (cached in {cdir})")
    return lab_t, ids, hemi_t, info


def check_transforms(reg, allen_dir):
    """
    Warp Allen's annotation_25.nrrd CCF -> template the same way and compare
    with AIND's ccf_annotation_to_template_moved_25.nii.gz.
    """
    import ants

    allen_dir = Path(allen_dir)
    aind_dir = Path(reg["template_path"]).parent
    theirs_p = aind_dir / "ccf_annotation_to_template_moved_25.nii.gz"
    for p in (allen_dir / "annotation_25.nrrd", allen_dir / "average_template_25.nrrd", theirs_p):
        if not p.is_file():
            raise SystemExit(f"--check-transforms needs {p}")
    ccf_img = ants.image_read(reg["ccf_reference_path"])
    ref = ccf_img.numpy()
    avg = read_any(allen_dir / "average_template_25.nrrd")
    from quantify_registered import corr

    best = match_orientation(avg, ref, corr)[0]
    ann = small_int(read_any(allen_dir / "annotation_25.nrrd", labels=True))
    ann = np.ascontiguousarray(apply_orient(ann, best[1], best[2]))
    comp, ids = compact(ann)
    mine = ccf_to_template(ants.from_numpy(comp, origin=ccf_img.origin, spacing=ccf_img.spacing,
                                           direction=ccf_img.direction), reg).numpy()
    mine = ids[np.rint(mine).astype(np.int64)]
    theirs = ants.image_read(str(theirs_p), pixeltype="unsigned int").numpy().astype(np.uint64)
    brain = (mine > 0) | (theirs > 0)
    agree = float((mine[brain] == theirs[brain]).mean())
    logger.info(f"Transform check: Allen annotation warped CCF -> template here vs AIND's own template-space "
                f"annotation: same label in {agree * 100:.1f}% of brain voxels "
                f"(Allen nrrd -> CCF grid match corr {best[0]:.3f})")
    if agree < 0.85:
        logger.error("Low agreement: the CCF -> template step does not reproduce AIND's annotation. "
                     "Do not trust the region values; send me this log.")
    return agree


# --------------------------------------------------------------------------
# brain side
# --------------------------------------------------------------------------
def to_sample_grid(img_t, reg, reg_dir, grid_shape):
    """Template-space label image -> this brain's registration grid, in the zarr's own axis order."""
    import ants
    from aind_ccf_reg.utils import check_orientation, rotate_image

    ap = reg["ants_params"]
    tmpl = ants.image_read(reg["template_path"])
    dummy = np.zeros(grid_shape, np.float32)
    rot, in_mat, _ = check_orientation(dummy, reg["input_orientation"], ap["template_orientations"])
    order = np.where(in_mat)[1]
    fixed = ants.from_numpy(rot, spacing=tuple(ap["spacing"][s] for s in order))
    fixed.set_direction(tmpl.direction)
    fixed.set_origin(tmpl.origin)
    moved = ants.apply_transforms(fixed=fixed, moving=img_t, interpolator="genericLabel",
                                  transformlist=[str(reg_dir / LS_AFFINE), str(reg_dir / LS_INVERSE_WARP)],
                                  whichtoinvert=[True, False])
    back, _ = rotate_image(moved.numpy(), in_mat, reverse=True)
    if back.shape != tuple(grid_shape):
        raise RuntimeError(f"labels came back as {back.shape}, registration grid is {grid_shape}")
    return np.rint(back).astype(np.int64)


def block_sums(zarr_path, grid_shape, grid_um, kind, threshold, level=None):
    """
    Per registration-grid block: number of level-0 voxels, and either the sum
    of values (intensity) or the count >= threshold (seg).
    """
    levels = ome_levels(zarr_path)
    p, sp = levels[0] if level is None else next(x for x in levels if str(x[0]) == str(level))
    f = block_factors(sp, grid_um)
    rd = LevelReader(Path(zarr_path) / str(p))
    for ax in range(3):
        if -(-rd.shape[ax] // f[ax]) != grid_shape[ax]:
            raise ValueError(f"{Path(zarr_path).name} level {p} {rd.shape} / {f} does not tile the "
                             f"registration grid {grid_shape}")
    thr = threshold * (255.0 if rd.dtype == np.uint8 else 1.0)
    logger.info(f"{Path(zarr_path).name}: level {p} {rd.shape} @ {sp} um, {rd.dtype}, blocks {f}"
                + (f", counting values >= {thr:g}" if kind == "seg" else ", summing values"))
    sums = np.zeros(grid_shape, np.float64)
    nvox = np.zeros(grid_shape, np.int64)
    step = int(np.lcm(rd.outer[0], f[0]))
    rows = rd.slab_rows()
    groups = {}
    for a, b in rows:
        groups.setdefault(a // step, []).append((a, b))
    t0 = time.time()
    for gi, rs in sorted(groups.items()):
        z0, z1 = rs[0][0], rs[-1][1]
        slab = rd.read(z0, z1)
        vals = (slab >= thr) if kind == "seg" else slab
        oz0, oz1 = z0 // f[0], -(-z1 // f[0])
        sub = (oz1 - oz0, grid_shape[1], grid_shape[2])
        pad = [(0, o * k - n) for n, k, o in zip(slab.shape, f, sub)]
        shp = [sub[0], f[0], sub[1], f[1], sub[2], f[2]]
        acc = np.int64 if kind == "seg" or np.issubdtype(slab.dtype, np.integer) else np.float64
        sums[oz0:oz1] = np.pad(vals, pad).reshape(shp).sum(axis=(1, 3, 5), dtype=acc)
        cz, cy, cx = (np.minimum(k, n - np.arange(o) * k).clip(0)
                      for n, k, o in zip(slab.shape, f, sub))
        nvox[oz0:oz1] = cz[:, None, None] * cy[None, :, None] * cx[None, None, :]
        if gi % 5 == 0:
            logger.info(f"  planes {z1}/{rd.shape[0]} ({time.time() - t0:.0f} s)")
    return sums, nvox, {"level": str(p), "level_voxel_um": list(sp), "block": f}


def region_rows(labels, hemi, sums, nvox, ids, names, kind, voxel_mm3):
    """Sum block values per region x hemisphere (labels/hemi on the registration grid)."""
    n_ids = len(ids)
    key = labels.ravel() * 3 + hemi.ravel().astype(np.int64)
    w_s = np.bincount(key, weights=sums.ravel(), minlength=3 * n_ids)
    w_n = np.bincount(key, weights=nvox.ravel().astype(np.float64), minlength=3 * n_ids)
    rows = []
    for c in range(1, n_ids):
        rid = int(ids[c])
        acr, nm = names.get(str(rid), ("", ""))
        for hemi_name, hs in (("left", [1]), ("right", [2]), ("both", [0, 1, 2])):
            k = [3 * c + h for h in hs]
            n = float(w_n[k].sum())
            if n == 0:
                continue
            s = float(w_s[k].sum())
            row = {"region_id": rid, "acronym": acr, "name": nm, "hemisphere": hemi_name,
                   "voxels": int(n), "volume_mm3": n * voxel_mm3}
            if kind == "seg":
                row.update(positive_voxels=int(round(s)), positive_volume_mm3=s * voxel_mm3, positive_fraction=s / n)
            else:
                row.update(sum_intensity=s, mean_intensity=s / n)
            rows.append(row)
    return rows


def write_csv(path, rows, lead):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lead) + list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**lead, **r})


def default_channels(sample, reg_channel):
    segs = sorted(e.name for e in Path(sample).iterdir()
                  if e.is_dir() and e.name.lower().startswith("seg-") and e.name.endswith(".zarr"))
    raws = [s[4:] for s in segs if (Path(sample) / s[4:]).is_dir() and s[4:] != reg_channel]
    return raws + segs


def quantify_one(sample, args, label_sets, names):
    import tifffile

    sample = Path(sample)
    reg_dir = sample / args.registration
    for f in ("run_config.json", LS_AFFINE, LS_INVERSE_WARP):
        if not (reg_dir / f).is_file():
            raise RuntimeError(f"{reg_dir} has no {f}: register this sample first")
    reg = json.loads((reg_dir / "run_config.json").read_text())
    grid_path = Path(reg["input_image_path"])
    grid_shape = tuple(json.loads((grid_path / "zarr.json").read_text())["shape"])
    grid_um = tuple(s * 1000 for s in reg["ants_params"]["spacing"])
    reg_channel = Path(reg.get("sources", {}).get("zarr", grid_path.parent)).name
    channels = args.channels or default_channels(sample, reg_channel)
    if not channels:
        raise RuntimeError("no channels to quantify (no seg-*.zarr; use --channels)")
    results = {}
    for lname, (lpath, _) in label_sets.items():
        out = reg_dir / f"sample_space_{lname}"
        work = Path(tempfile.mkdtemp(prefix="ssq_")) / out.name if is_network(out) else out
        work.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(work / "quantification.log", encoding="utf-8")
        logging.getLogger().addHandler(fh)
        try:
            t0 = time.time()
            lab_t, ids, hemi_t, linfo = template_space_labels(lname, lpath, reg, args.cache)
            labels = to_sample_grid(lab_t, reg, reg_dir, grid_shape)
            hemi = to_sample_grid(hemi_t, reg, reg_dir, grid_shape)
            hemi[labels == 0] = 0
            logger.info(f"{sample.name}: {lname} labels on the registration grid {grid_shape}: "
                        f"{len(np.unique(labels)) - 1} regions present ({time.time() - t0:.0f} s)")
            tifffile.imwrite(work / f"{lname}_labels_regrid.tif", ids[labels].astype(np.uint32), compression="zlib")
            tifffile.imwrite(work / "hemisphere_regrid.tif", hemi.astype(np.uint8), compression="zlib")
            summary = {"sample": str(sample), "labels": lname, "label_file": str(lpath),
                       "registration_grid": {"shape": list(grid_shape), "voxel_um": list(grid_um),
                                             "zarr_level": str(grid_path)},
                       "threshold": args.threshold, "label_info": {k: v for k, v in linfo.items() if k != "ids"},
                       "channels": {}}
            for ch in channels:
                zp = sample / ch
                if not zp.is_dir():
                    logger.warning(f"{sample.name}: {ch} not found, skipped")
                    continue
                kind = "seg" if ch.lower().startswith("seg-") else "intensity"
                sums, nvox, cinfo = block_sums(zp, grid_shape, grid_um, kind, args.threshold, args.level)
                vox_mm3 = float(np.prod(cinfo["level_voxel_um"])) * 1e-9
                rows = region_rows(labels, hemi, sums, nvox, ids, names.get(lname, {}), kind, vox_mm3)
                label = ch.replace(".ome.zarr", "").replace(".zarr", "")
                write_csv(work / f"{label}_regions.csv", rows,
                          {"sample": sample.name, "channel": label, "labels": lname})
                outside = float(sums[labels == 0].sum())
                cinfo.update(kind=kind, rows=len(rows), total=float(sums.sum()), outside_atlas=outside)
                summary["channels"][ch] = cinfo
                logger.info(f"{sample.name}: {label} -> {len(rows)} rows; "
                            f"{'positive voxels' if kind == 'seg' else 'intensity'} outside the atlas: "
                            f"{outside / max(float(sums.sum()), 1e-12) * 100:.1f}%")
                results.setdefault((lname, kind), []).extend(
                    {"sample": sample.name, "channel": label, "labels": lname, **r} for r in rows)
            (work / "quantification.json").write_text(json.dumps(summary, indent=2, default=str))
        finally:
            logging.getLogger().removeHandler(fh)
            fh.close()
        if work != out:
            shutil.copytree(work, out, dirs_exist_ok=True)
            shutil.rmtree(work.parent, ignore_errors=True)
        logger.info(f"{sample.name}: results in {out}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="sample folders (*_Destripe_DONE)")
    ap.add_argument("--registration", default="registration_ccf25_aind/Ex_488_Ch0_stitched")
    ap.add_argument("--labels", nargs="+", required=True, metavar="NAME=PATH",
                    help="label volume(s) in CCF 25 um space, e.g. kim=Kim_to_allen25_ann_fwdtransforms.tif")
    ap.add_argument("--labels-info", nargs="*", default=[], metavar="NAME=CSV")
    ap.add_argument("--channels", nargs="+", help="zarrs to quantify (default: seg-*.zarr and their raw channels)")
    ap.add_argument("--threshold", type=float, default=0.5, help="seg probability threshold (0-1)")
    ap.add_argument("--level", help="zarr level to quantify (default: finest)")
    ap.add_argument("--summary", help="folder for CSVs combining all samples")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "ccfreg_label_cache"),
                    help="where template-space labels are cached (default: %(default)s)")
    ap.add_argument("--check-transforms", metavar="ALLEN_DIR",
                    help="first verify the CCF -> template step with annotation_25.nrrd in this folder")
    args = ap.parse_args(argv)

    import aind_ccf_reg.utils  # noqa: F401  (its import reconfigures logging)

    setup_logging()
    label_sets = {}
    for item in args.labels:
        k, p = item.split("=", 1)
        label_sets[k] = (p, None)
    names = {k: read_names(v) for k, v in (x.split("=", 1) for x in args.labels_info)}

    if args.check_transforms:
        first = Path(args.samples[0].rstrip("/\\")) / args.registration
        check_transforms(json.loads((first / "run_config.json").read_text()), args.check_transforms)

    combined, status = {}, []
    for s in args.samples:
        s = s.rstrip("/\\")
        t0 = time.time()
        try:
            res = quantify_one(s, args, label_sets, names)
            for k, rows in res.items():
                combined.setdefault(k, []).extend(rows)
            status.append((Path(s).name, "done", round(time.time() - t0)))
        except Exception as e:  # noqa: BLE001  (one bad sample must not stop the others)
            logger.exception(f"{Path(s).name}: FAILED: {e}")
            status.append((Path(s).name, f"FAILED: {e}", round(time.time() - t0)))
    if args.summary:
        sd = Path(args.summary)
        sd.mkdir(parents=True, exist_ok=True)
        for (lname, kind), rows in combined.items():
            p = sd / f"{lname}_{kind}_regions.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            logger.info(f"wrote {p} ({len(rows)} rows)")
    for name, st, sec in status:
        logger.info(f"{name}: {st} ({sec} s)")
    return 1 if any(st != "done" for _, st, _ in status) else 0


if __name__ == "__main__":
    sys.exit(main())
