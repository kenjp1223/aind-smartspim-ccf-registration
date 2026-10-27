"""
Signal channels of registered brains in atlas space, on the grid of a CCF
25 um label file (e.g. Kim_to_allen25_ann_fwdtransforms.tif): same array size
and axis order, so its labels index the maps directly.

Sums are preserved (no interpolation of the signal):
  1. The atlas's own voxel coordinates are carried CCF -> AIND template ->
     brain (the transform chain used for labels, nearest neighbour), so every
     voxel of the brain's registration grid (16 um) knows the 25 um atlas
     voxel it belongs to.
  2. For each 16 um block the full-resolution (level 0) totals are computed:
       intensity channels: sum of voxel values
       seg-* channels:     number of voxels with probability >= --threshold
     plus the number of level-0 voxels.
  3. Block totals are added into their atlas voxel. Summing a map over a
     region therefore gives that region's total in the brain.

Outputs, per sample, in <registration folder>/atlas_space/ (all float32,
in the reference tif's shape and axis order):
  voxels_ccf25.tif                 level-0 voxels mapped into each atlas voxel
  <channel>_sum_ccf25.tif          intensity sum          (intensity channels)
  <channel>_mean_ccf25.tif         sum / voxels
  <channel>_count_ccf25.tif        voxels >= threshold    (seg-* channels)
  <channel>_fraction_ccf25.tif     count / voxels
  atlas_space.json                 settings, totals, how much fell outside the atlas

Usage (from the repo's code/ folder):
  uv run python warp_signal_to_atlas.py "<sample>" ["<sample>" ...] ^
      --reference "<Kim_to_allen25_ann_fwdtransforms.tif>" ^
      --channels Ex_639_Ch2_stitched.ome.zarr seg-Ex_639_Ch2_stitched.ome.zarr
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from quantify_registered import invert_orient, is_network, read_any
from quantify_sample_space import (
    LS_AFFINE,
    LS_INVERSE_WARP,
    block_sums,
    ccf_to_template,
    default_channels,
    labels_on_ccf_grid,
    setup_logging,
    to_sample_grid,
)

logger = logging.getLogger("sample_space")


def template_space_coords(reg, cache_dir):
    """
    CCF voxel coordinates (index + 1 per axis, 0 = outside) carried into the
    AIND template space; the same for every brain, so cached.
    """
    import ants

    key = hashlib.sha1(f"{reg['ccf_reference_path']}|{reg['ccf_to_template_transform_path']}|"
                       f"{reg['template_path']}".encode()).hexdigest()[:12]
    cdir = Path(cache_dir) / f"ccf_coords_{key}"
    files = [cdir / f"coord{k}_template.nii.gz" for k in range(3)]
    if all(f.is_file() for f in files):
        logger.info(f"CCF coordinates in template space from cache {cdir}")
        return [ants.image_read(str(f)) for f in files]
    ccf = ants.image_read(reg["ccf_reference_path"])
    shape = ccf.shape
    hdr = dict(origin=ccf.origin, spacing=ccf.spacing, direction=ccf.direction)
    out = []
    t0 = time.time()
    cdir.mkdir(parents=True, exist_ok=True)
    for k in range(3):
        view = [1, 1, 1]
        view[k] = shape[k]
        coord = np.broadcast_to((np.arange(shape[k], dtype=np.float32) + 1).reshape(view), shape).copy()
        img_t = ccf_to_template(ants.from_numpy(coord, **hdr), reg, interpolator="nearestNeighbor")
        ants.image_write(img_t, str(files[k]))
        out.append(img_t)
    logger.info(f"CCF coordinates -> template space in {time.time() - t0:.0f} s (cached in {cdir})")
    return out


def atlas_index_on_brain(reg, reg_dir, grid_shape, cache_dir):
    """Linear CCF-grid index for every registration-grid voxel of the brain (-1 = outside the atlas)."""
    import ants

    ccf_shape = ants.image_header_info(reg["ccf_reference_path"])["dimensions"]
    ccf_shape = tuple(int(x) for x in ccf_shape)
    coords_t = template_space_coords(reg, cache_dir)
    lin = np.zeros(grid_shape, np.int32)   # CCF grids have < 2**31 voxels
    inside = np.ones(grid_shape, bool)
    stride = [ccf_shape[1] * ccf_shape[2], ccf_shape[2], 1]
    for k in range(3):
        c = to_sample_grid(coords_t[k], reg, reg_dir, grid_shape, interpolator="nearestNeighbor").astype(np.int32)
        inside &= c > 0
        lin += (c - 1) * np.int32(stride[k])
        del c
    lin[~inside] = -1
    return lin, ccf_shape


def warp_one(sample, args, ref_orient, ref_shape):
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
        raise RuntimeError("no channels (no seg-*.zarr; use --channels)")

    out = reg_dir / args.out_name
    work = Path(tempfile.mkdtemp(prefix="atl_")) / out.name if is_network(out) else out
    work.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(work / "atlas_space.log", encoding="utf-8")
    logging.getLogger().addHandler(fh)
    try:
        t0 = time.time()
        lin, ccf_shape = atlas_index_on_brain(reg, reg_dir, grid_shape, args.cache)
        n_ccf = int(np.prod(ccf_shape))
        logger.info(f"{sample.name}: atlas voxel of every {grid_um} um voxel found ({time.time() - t0:.0f} s); "
                    f"{(lin >= 0).mean() * 100:.1f}% of the registration grid lies inside the atlas")
        sel = lin >= 0
        idx = lin[sel]

        def to_ref(flat):
            """CCF-grid vector -> array in the reference tif's shape and axis order."""
            a = invert_orient(flat.reshape(ccf_shape), ref_orient["perm"], ref_orient["flips"])
            a = np.ascontiguousarray(a).astype(np.float32)
            if a.shape != tuple(ref_shape):
                raise RuntimeError(f"atlas map {a.shape} != reference {ref_shape}")
            return a

        info = {"sample": str(sample), "registration": str(reg_dir), "reference": args.reference,
                "reference_orientation_to_ccf": ref_orient, "registration_grid_um": list(grid_um),
                "threshold": args.threshold, "channels": {}}
        voxels_written = False
        for ch in channels:
            zp = sample / ch
            if not zp.is_dir():
                logger.warning(f"{sample.name}: {ch} not found, skipped")
                continue
            kind = "seg" if ch.lower().startswith("seg-") else "intensity"
            sums, nvox, cinfo = block_sums(zp, grid_shape, grid_um, kind, args.threshold, args.level)
            s_atlas = np.bincount(idx, weights=sums[sel], minlength=n_ccf)
            n_atlas = np.bincount(idx, weights=nvox[sel].astype(np.float64), minlength=n_ccf)
            label = ch.replace(".ome.zarr", "").replace(".zarr", "")
            if not voxels_written:
                tifffile.imwrite(work / "voxels_ccf25.tif", to_ref(n_atlas), compression="zlib")
                voxels_written = True
            ratio = np.divide(s_atlas, n_atlas, out=np.zeros_like(s_atlas), where=n_atlas > 0)
            names = ("count", "fraction") if kind == "seg" else ("sum", "mean")
            tifffile.imwrite(work / f"{label}_{names[0]}_ccf25.tif", to_ref(s_atlas), compression="zlib")
            tifffile.imwrite(work / f"{label}_{names[1]}_ccf25.tif", to_ref(ratio), compression="zlib")
            total, mapped = float(sums.sum()), float(s_atlas.sum())
            cinfo.update(kind=kind, total=total, in_atlas=mapped,
                         outside_atlas_fraction=(total - mapped) / total if total else 0.0)
            info["channels"][ch] = cinfo
            logger.info(f"{sample.name}: {label}: {names[0]} in atlas {mapped:.6g} of {total:.6g} "
                        f"({(total - mapped) / max(total, 1e-12) * 100:.1f}% maps outside the atlas)")
        info["seconds"] = round(time.time() - t0, 1)
        (work / "atlas_space.json").write_text(json.dumps(info, indent=2, default=str))
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
    if work != out:
        shutil.copytree(work, out, dirs_exist_ok=True)
        shutil.rmtree(work.parent, ignore_errors=True)
    logger.info(f"{sample.name}: results in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="sample folders (*_Destripe_DONE)")
    ap.add_argument("--reference", required=True,
                    help="CCF 25 um volume whose size and axis order the maps should have "
                         "(e.g. Kim_to_allen25_ann_fwdtransforms.tif)")
    ap.add_argument("--registration", default="registration_ccf25_aind/Ex_488_Ch0_stitched")
    ap.add_argument("--channels", nargs="+", help="zarrs to warp (default: seg-*.zarr and their raw channels)")
    ap.add_argument("--threshold", type=float, default=0.5, help="seg probability threshold (0-1)")
    ap.add_argument("--level", help="zarr level summed (default: finest)")
    ap.add_argument("--out-name", default="atlas_space", help="output folder name in the registration folder")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "ccfreg_label_cache"))
    args = ap.parse_args(argv)

    import aind_ccf_reg.utils  # noqa: F401  (its import reconfigures logging)
    import ants

    setup_logging()
    first = Path(args.samples[0].rstrip("/\\")) / args.registration
    reg0 = json.loads((first / "run_config.json").read_text())
    ccf = ants.image_read(reg0["ccf_reference_path"])
    _, ref_orient = labels_on_ccf_grid(args.reference, ccf)
    ref_shape = tuple(read_any(args.reference, labels=True).shape)

    status = []
    for s in args.samples:
        s = s.rstrip("/\\")
        t0 = time.time()
        try:
            warp_one(s, args, ref_orient, ref_shape)
            status.append((Path(s).name, "done", round(time.time() - t0)))
        except Exception as e:  # noqa: BLE001
            logger.exception(f"{Path(s).name}: FAILED: {e}")
            status.append((Path(s).name, f"FAILED: {e}", round(time.time() - t0)))
    for name, st, sec in status:
        logger.info(f"{name}: {st} ({sec} s)")
    return 1 if any(st != "done" for _, st, _ in status) else 0


if __name__ == "__main__":
    sys.exit(main())
