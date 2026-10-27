"""
Quantify TRAILMAP segmentation (seg-*.ome.zarr) and other channels of a
registered SmartSPIM brain per atlas region, in Allen CCF 25 um space.

Needs a finished registration from run_custom_atlas.py (AIND route or custom
atlas) in <sample>/registration_<atlas>/<registration channel>/.

For each channel:
  seg-*  : axon density = fraction of full-resolution voxels with TRAILMAP
           probability >= --threshold, averaged into the registration grid
           (the Allen "projection density" idea), plus mean probability.
  others : mean intensity at the registration resolution.
These maps are warped into the atlas (brain -> template -> CCF, linear
interpolation), written as 25 um tifs in the Allen annotation_25.nrrd layout
(planes = anterior->posterior, rows = dorsal->ventral, columns = left->right,
the same layout as ClearMap's / the Kim-to-Allen tifs), and summarised per
region and hemisphere for each label set given (Allen CCF, Kim, ...).

Outputs in <registration folder>/quantification/:
  <channel>_<map>_ccf25.tif         atlas-space maps (float32)
  <channel>_<labels>_regions.csv    one row per region x hemisphere
  atlas_cache/                      label volumes resampled to the output grid
  quantification.json               settings, orientation matches, timings

Usage (from the repo's code/ folder):
  uv run python quantify_registered.py "<sample folder>" ^
      --allen-dir "<folder with average_template_25.nrrd, annotation_25.nrrd>" ^
      --labels kim="<Kim_to_allen25_ann_fwdtransforms.tif>" ^
      --labels-info kim="<atlas_info_KimRef_FPbasedLabel_v4.0.csv>"
"""

import argparse
import csv
import functools
import itertools
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("quantify")

TRANSFORMS = (
    "ls_to_template_SyN_0GenericAffine.mat",
    "ls_to_template_SyN_1Warp.nii.gz",
)
CCF_ANNOTATION_MAP = Path(__file__).resolve().parent / "aind_ccf_reg" / "ccf_files" / "annotation_map.json"


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
    logger.disabled = False
    logger.setLevel(logging.INFO)


def is_network(path):
    return str(path).startswith(("\\\\", "//"))


# --------------------------------------------------------------------------
# zarr v3 levels (tensorstore), including Windows network paths
# --------------------------------------------------------------------------
class LevelReader:
    """
    Read z-slabs of one OME-Zarr v3 level. tensorstore can't open Windows
    network (UNC) paths, so for those the shard files of each slab are copied
    to a local temp folder first (one row of shards at a time).
    """

    def __init__(self, path):
        self.path = Path(path)
        meta = json.loads((self.path / "zarr.json").read_text())
        self.shape = tuple(meta["shape"])
        self.dtype = np.dtype(meta["data_type"])
        self.outer = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        enc = meta.get("chunk_key_encoding", {"name": "default"})
        sep = enc.get("configuration", {}).get("separator", "/")
        if enc.get("name", "default") != "default" or sep != "/":
            raise ValueError(f"{path}: unsupported chunk key encoding {enc}")
        self.remote = is_network(self.path)
        self._store = None if self.remote else self._open(self.path)

    @staticmethod
    def _open(path):
        import tensorstore as ts

        return ts.open({"driver": "zarr3", "kvstore": {"driver": "file",
                        "path": os.path.abspath(path).replace("\\", "/")}}, read=True).result()

    def slab_rows(self):
        """z ranges aligned to the shard (outer chunk) rows."""
        step = self.outer[0]
        return [(z, min(z + step, self.shape[0])) for z in range(0, self.shape[0], step)]

    def read(self, z0, z1):
        if not self.remote:
            return np.asarray(self._store[z0:z1].read().result())
        step = self.outer[0]
        tmp = Path(tempfile.mkdtemp(prefix="quant_zarr_"))
        try:
            shutil.copy2(self.path / "zarr.json", tmp / "zarr.json")
            for iz in range(z0 // step, (z1 - 1) // step + 1):
                src = self.path / "c" / str(iz)
                if src.is_dir():
                    shutil.copytree(src, tmp / "c" / str(iz))
            return np.asarray(self._open(tmp)[z0:z1].read().result())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def read_all(self):
        return np.concatenate([self.read(a, b) for a, b in self.slab_rows()], axis=0)


def ome_levels(zarr_path):
    """[(path, (z, y, x) voxel size in um)] from OME-Zarr 0.5 metadata."""
    meta = json.loads((Path(zarr_path) / "zarr.json").read_text())
    attrs = meta.get("attributes", {})
    ms = attrs.get("ome", attrs)["multiscales"][0]
    out = []
    for ds in ms["datasets"]:
        sc = next(t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale")
        out.append((ds["path"], tuple(float(s) for s in sc[-3:])))
    return out


# --------------------------------------------------------------------------
# channel maps on the registration grid
# --------------------------------------------------------------------------
def block_factors(fine_um, coarse_um):
    f = [c / s for c, s in zip(coarse_um, fine_um)]
    fi = [int(round(x)) for x in f]
    if any(abs(a - b) > 1e-3 or b < 1 for a, b in zip(f, fi)):
        raise ValueError(f"voxel {coarse_um} is not an integer multiple of {fine_um}")
    return fi


def block_mean(arr, f, out_shape):
    """Mean over f-blocks (edge blocks: mean over the voxels that exist)."""
    pad = [(0, o * k - n) for n, k, o in zip(arr.shape, f, out_shape)]
    if any(p[1] < 0 for p in pad):
        raise ValueError(f"array {arr.shape} larger than {out_shape} x {f}")
    a = np.pad(arr.astype(np.float32), pad)
    shp = [out_shape[0], f[0], out_shape[1], f[1], out_shape[2], f[2]]
    s = a.reshape(shp).sum(axis=(1, 3, 5))
    # voxels per block: product of per-axis counts (only edge blocks are short)
    cz, cy, cx = (np.minimum(k, n - np.arange(o) * k).clip(0).astype(np.float32)
                  for n, k, o in zip(arr.shape, f, out_shape))
    cnt = cz[:, None, None] * cy[None, :, None] * cx[None, None, :]
    return np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)


def find_level(levels, target_um, finest=False):
    if finest:
        return levels[0]
    for p, sp in levels:
        if all(abs(a - b) < 1e-3 * max(1.0, b) for a, b in zip(sp, target_um)):
            return p, sp
    return None


def seg_maps(zarr_path, grid_shape, grid_um, threshold, level=None):
    """
    Density (fraction >= threshold) and mean probability of a TRAILMAP
    segmentation, from a fine level averaged into the registration grid.
    """
    levels = ome_levels(zarr_path)
    if level is None:
        p, sp = levels[0]
    else:
        p, sp = next((x for x in levels if str(x[0]) == str(level)), levels[0])
    f = block_factors(sp, grid_um)
    rd = LevelReader(Path(zarr_path) / str(p))
    scale = 255.0 if rd.dtype == np.uint8 else 1.0
    thr = threshold * scale
    logger.info(f"{Path(zarr_path).name}: level {p} {rd.shape} @ {sp} um, {rd.dtype}; "
                f"threshold {threshold} (value >= {thr:g}); blocks {f} -> grid {grid_shape}")
    dens = np.zeros(grid_shape, np.float32)
    prob = np.zeros(grid_shape, np.float32)
    # slabs: whole shard rows, grouped so that each slab covers whole z-blocks
    rows = rd.slab_rows()
    step = int(np.lcm(rd.outer[0], f[0]))
    groups = {}
    for a, b in rows:
        groups.setdefault(a // step, []).append((a, b))
    t0 = time.time()
    for gi, rs in sorted(groups.items()):
        z0, z1 = rs[0][0], rs[-1][1]
        slab = rd.read(z0, z1)
        oz0 = z0 // f[0]
        oz1 = min(-(-z1 // f[0]), grid_shape[0])
        sub_shape = (oz1 - oz0, grid_shape[1], grid_shape[2])
        dens[oz0:oz1] = block_mean(slab >= thr, f, sub_shape)
        prob[oz0:oz1] = block_mean(slab, f, sub_shape) / scale
        if gi % 5 == 0:
            logger.info(f"  planes {z1}/{rd.shape[0]} ({time.time() - t0:.0f} s)")
    return {"density": dens, "probability": prob}


def intensity_map(zarr_path, grid_shape, grid_um):
    """Mean intensity on the registration grid (same level, or averaged from a finer one)."""
    levels = ome_levels(zarr_path)
    hit = find_level(levels, grid_um)
    if hit:
        arr = LevelReader(Path(zarr_path) / str(hit[0])).read_all().astype(np.float32)
        if arr.shape != tuple(grid_shape):
            raise ValueError(f"{zarr_path} level {hit[0]} is {arr.shape}, registration grid {grid_shape}")
        return {"intensity": arr}
    finer = [x for x in levels if all(s <= g + 1e-6 for s, g in zip(x[1], grid_um))]
    if not finer:
        raise ValueError(f"{zarr_path}: no level at or finer than {grid_um} um")
    p, sp = finer[-1]
    arr = LevelReader(Path(zarr_path) / str(p)).read_all()
    return {"intensity": block_mean(arr, block_factors(sp, grid_um), grid_shape)}


# --------------------------------------------------------------------------
# warp to the atlas
# --------------------------------------------------------------------------
def warp_to_atlas(arr, reg, reg_dir):
    """Registration-grid array -> atlas reference grid (AIND: CCF), linear."""
    import ants
    from aind_ccf_reg.utils import check_orientation

    ap = reg["ants_params"]
    tmpl = ants.image_read(reg["template_path"])
    ccf = ants.image_read(reg["ccf_reference_path"])
    img_out, in_mat, _ = check_orientation(arr.astype(np.float64), reg["input_orientation"],
                                           ap["template_orientations"])
    order = np.where(in_mat)[1]
    spacing = tuple(ap["spacing"][s] for s in order)
    img = ants.from_numpy(img_out.astype(np.float32), spacing=spacing)
    img.set_direction(tmpl.direction)
    img.set_origin(tmpl.origin)
    moved = ants.apply_transforms(fixed=tmpl, moving=img, interpolator="linear",
                                  transformlist=[str(reg_dir / TRANSFORMS[1]), str(reg_dir / TRANSFORMS[0])])
    moved = ants.apply_transforms(fixed=ccf, moving=moved, interpolator="linear",
                                  transformlist=list(reg["template_to_ccf_transform_path"]))
    return moved.numpy()


# --------------------------------------------------------------------------
# grids: AIND's CCF NIfTI vs the Allen .nrrd / Kim tif layout
# --------------------------------------------------------------------------
def orientations():
    """All 48 axis permutations x flips, identity first."""
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product([False, True], repeat=3):
            yield perm, flips


def apply_orient(a, perm, flips):
    b = np.transpose(a, perm)
    for ax, fl in enumerate(flips):
        if fl:
            b = np.flip(b, ax)
    return b


def invert_orient(b, perm, flips):
    for ax, fl in enumerate(flips):
        if fl:
            b = np.flip(b, ax)
    return np.transpose(b, np.argsort(perm))


def match_orientation(moving, ref, score, step=3):
    """
    Best (perm, flips) with apply_orient(moving) ~ ref, scored on every
    step-th voxel. Ties (within 1e-4) go to the orientation with fewest flips
    and permuted axes.
    """
    r = ref[::step, ::step, ::step]
    results = []
    for perm, flips in orientations():
        if tuple(moving.shape[i] for i in perm) != ref.shape:
            continue
        m = apply_orient(moving, perm, flips)[::step, ::step, ::step]
        results.append((score(m, r), perm, flips))
    if not results:
        raise ValueError(f"no axis order of {moving.shape} matches {ref.shape}")
    simple = {(p, f): sum(f) + sum(a != b for a, b in zip(p, range(3))) for _, p, f in results}
    results.sort(key=lambda x: (-round(x[0], 4), simple[(x[1], x[2])]))
    return results


def corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float((a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def label_nmi(a, b):
    """Normalised mutual information of two label volumes (regions coincide -> high)."""
    _, ia = np.unique(a.ravel(), return_inverse=True)
    _, ib = np.unique(b.ravel(), return_inverse=True)
    nb = ib.max() + 1
    _, joint = np.unique(ia.astype(np.int64) * nb + ib, return_counts=True)
    n = ia.size

    def h(counts):
        p = counts / n
        return float(-(p * np.log(p)).sum())

    ha, hb, hab = h(np.bincount(ia)), h(np.bincount(ib)), h(joint)
    return 2 * (ha + hb - hab) / (ha + hb + 1e-12)


def dice_mask(a, b):
    a, b = a > 0, b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def read_any(path, labels=False):
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        return np.asarray(tifffile.imread(path))
    import ants

    return ants.image_read(path, pixeltype="unsigned int" if labels else "float").numpy()


def small_int(a):
    """Labels as uint32 when they fit (Allen IDs do), to halve memory."""
    if np.issubdtype(a.dtype, np.floating):
        a = np.rint(a)
    return a.astype(np.uint32) if a.max() < 2**32 else a.astype(np.uint64)


def read_names(path):
    """{id: (acronym, name)} from a CSV (id/acronym/name columns) or JSON {id: acronym}."""
    if not path:
        return {}
    if str(path).lower().endswith(".json"):
        return {str(k): (str(v), "") for k, v in json.loads(Path(path).read_text()).items()}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    low = {c.lower().strip(): c for c in rows[0]}
    idc = next((low[c] for c in ("id", "label", "label_id", "structure_id", "index") if c in low), None)
    acc = next((low[c] for c in ("acronym", "abbreviation", "abbr", "short_name") if c in low), None)
    nmc = next((low[c] for c in ("name", "full_name", "structure_name", "region") if c in low), None)
    if idc is None:
        raise SystemExit(f"{path}: no id column in {list(rows[0])}")
    out = {}
    for r in rows:
        try:
            k = str(int(float(r[idc])))
        except (TypeError, ValueError):
            continue
        out[k] = (str(r.get(acc, "") or "").strip() if acc else "", str(r.get(nmc, "") or "").strip() if nmc else "")
    return out


def _same(a):
    return a


def _reorient(a, perm, flips):
    return np.ascontiguousarray(apply_orient(a, perm, flips))


def prepare_atlases(ccf_ref_path, allen_dir, extra_labels, cache_dir):
    """
    Output grid = Allen annotation_25.nrrd layout (AP, DV, LR) when allen_dir
    is given, else the registration's reference grid. Returns
    (to_out(array_in_ref_grid) -> array_in_out_grid, {name: label array}, info).
    """
    import ants

    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_img = ants.image_read(str(ccf_ref_path))
    ref = ref_img.numpy()
    info = {"reference": str(ccf_ref_path), "reference_shape": list(ref.shape),
            "voxel_mm": [float(x) for x in ref_img.spacing]}
    labels = {}
    to_out = _same
    if allen_dir:
        allen_dir = Path(allen_dir)
        avg = read_any(allen_dir / "average_template_25.nrrd")
        res = match_orientation(ref, avg, corr)
        best, second = res[0], res[1] if len(res) > 1 else (float("nan"),)
        perm, flips = best[1], best[2]
        logger.info(f"AIND CCF grid -> Allen nrrd layout: axes {perm}, flips {flips}, "
                    f"correlation {best[0]:.4f} (next best {second[0]:.4f})")
        if best[0] < 0.95:
            raise SystemExit(f"{ccf_ref_path} does not match {allen_dir / 'average_template_25.nrrd'} "
                             f"(best correlation {best[0]:.3f})")
        info["ref_to_allen"] = {"perm": list(perm), "flips": list(flips), "corr": best[0], "next": second[0]}

        to_out = functools.partial(_reorient, perm=perm, flips=flips)

        allen = small_int(read_any(allen_dir / "annotation_25.nrrd", labels=True))
        labels["allen"] = allen
        out_shape = avg.shape
    else:
        out_shape = ref.shape
    for name, path in extra_labels.items():
        arr = small_int(read_any(path, labels=True))
        if "allen" in labels:
            res = match_orientation(arr, labels["allen"], label_nmi)
            how = "region overlap with Allen labels (NMI)"
        else:
            res = match_orientation(arr, np.ones(out_shape), dice_mask)
            how = "brain-mask Dice"
        best = res[0]
        second = res[1] if len(res) > 1 else (float("nan"),)
        perm, flips = best[1], best[2]
        logger.info(f"{name} labels {arr.shape} -> output grid: axes {perm}, flips {flips}, "
                    f"{how} {best[0]:.4f} (next best {second[0]:.4f})")
        if best[0] - second[0] < 0.002:
            logger.warning(f"{name}: next-best orientation (axes {second[1]}, flips {second[2]}) scores "
                           "almost the same, typically the left/right mirror of a symmetric atlas")
        labels[name] = np.ascontiguousarray(apply_orient(arr, perm, flips))
        info[f"{name}_to_output"] = {"perm": list(perm), "flips": list(flips), "score": best[0],
                                     "score_type": how, "next": second[0]}
    return to_out, labels, out_shape, info


# --------------------------------------------------------------------------
# region statistics
# --------------------------------------------------------------------------
def region_table(labels, maps, names, voxel_mm3, lr_axis=2):
    """
    Rows per region x hemisphere (left / right split at the midline of
    lr_axis, assuming that axis runs left -> right) and both together.
    lr_axis=None: no hemisphere split.
    """
    ids, inv = np.unique(labels.ravel(), return_inverse=True)
    n_ids = len(ids)
    side = np.zeros(labels.shape, np.int8)
    if lr_axis is not None:
        sl = [slice(None)] * 3
        sl[lr_axis] = slice(labels.shape[lr_axis] // 2, None)
        side[tuple(sl)] = 1
    key = inv * 2 + side.ravel()
    count = np.bincount(key, minlength=2 * n_ids)
    sums = {m: np.bincount(key, weights=v.ravel().astype(np.float64), minlength=2 * n_ids) for m, v in maps.items()}
    rows = []
    for i, rid in enumerate(ids):
        if rid == 0:
            continue
        acr, nm = names.get(str(int(rid)), ("", ""))
        sides = (("left", [2 * i]), ("right", [2 * i + 1])) if lr_axis is not None else ()
        for hemi, sel in sides + (("both", [2 * i, 2 * i + 1]),):
            n = int(count[sel].sum())
            if n == 0:
                continue
            row = {"region_id": int(rid), "acronym": acr, "name": nm, "hemisphere": hemi,
                   "voxels": n, "volume_mm3": round(n * voxel_mm3, 6)}
            for m, s in sums.items():
                row[f"mean_{m}"] = float(s[sel].sum() / n)
            rows.append(row)
    return rows


def write_csv(path, rows, extra=None):
    if not rows:
        return
    cols = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=(list(extra) if extra else []) + cols)
        w.writeheader()
        for r in rows:
            w.writerow({**(extra or {}), **r})


# --------------------------------------------------------------------------
def default_channels(sample):
    """All seg-* zarrs (and nothing else) in the sample folder."""
    return sorted(e.name for e in Path(sample).iterdir()
                  if e.is_dir() and e.name.lower().startswith("seg-") and e.name.endswith(".zarr"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", help="sample folder (*_Destripe_DONE)")
    ap.add_argument("--registration", default="registration_ccf25_aind/Ex_488_Ch0_stitched",
                    help="registration folder inside the sample (default: %(default)s)")
    ap.add_argument("--channels", nargs="+",
                    help="zarr folder names to quantify (default: all seg-*.ome.zarr); e.g. "
                         "seg-Ex_639_Ch2_stitched.ome.zarr Ex_639_Ch2_stitched.ome.zarr")
    ap.add_argument("--threshold", type=float, default=0.5, help="TRAILMAP probability threshold (0-1)")
    ap.add_argument("--seg-level", help="seg zarr level used for thresholding (default: finest)")
    ap.add_argument("--allen-dir", help="folder with average_template_25.nrrd and annotation_25.nrrd: "
                                        "output in that layout, Allen regions quantified")
    ap.add_argument("--allen-info", default=str(CCF_ANNOTATION_MAP),
                    help="names for Allen IDs (CSV id/acronym/name or JSON; default: repo's CCF map)")
    ap.add_argument("--labels", nargs="*", default=[], metavar="NAME=PATH",
                    help="more label volumes in Allen 25 um space, any axis order, e.g. kim=Kim_to_allen25.tif")
    ap.add_argument("--labels-info", nargs="*", default=[], metavar="NAME=CSV",
                    help="region names for --labels, e.g. kim=atlas_info_KimRef_FPbasedLabel_v4.0.csv")
    ap.add_argument("--out", help="output folder (default: <registration>/quantification)")
    args = ap.parse_args(argv)

    sample = Path(args.sample.rstrip("/\\"))
    reg_dir = sample / Path(args.registration)
    if not (reg_dir / "run_config.json").is_file():
        raise SystemExit(f"No registration in {reg_dir} (run run_custom_atlas.py first)")
    missing = [t for t in TRANSFORMS if not (reg_dir / t).is_file()]
    if missing:
        raise SystemExit(f"{reg_dir} is missing {missing}")
    out = Path(args.out) if args.out else reg_dir / "quantification"
    final_out = out
    if is_network(out):  # tifffile / ANTs write fine, but keep heavy I/O local
        out = Path(tempfile.mkdtemp(prefix="quant_")) / "quantification"
    out.mkdir(parents=True, exist_ok=True)
    import aind_ccf_reg.utils  # noqa: F401  (its import reconfigures logging)

    setup_logging(out / "quantification.log")
    t_start = time.time()

    reg = json.loads((reg_dir / "run_config.json").read_text())
    grid_path = Path(reg["input_image_path"])
    grid_meta = json.loads((grid_path / "zarr.json").read_text())
    grid_shape = tuple(grid_meta["shape"])
    grid_um = tuple(s * 1000 for s in reg["ants_params"]["spacing"])
    logger.info(f"Sample {sample.name}; registration {reg_dir.name} (grid {grid_shape} @ {grid_um} um)")

    extra = dict(x.split("=", 1) for x in args.labels)
    extra_info = dict(x.split("=", 1) for x in args.labels_info)
    to_out, label_sets, out_shape, orient_info = prepare_atlases(
        reg["ccf_reference_path"], args.allen_dir, extra, out / "atlas_cache")
    names = {"allen": read_names(args.allen_info)}
    for k in extra:
        names[k] = read_names(extra_info.get(k))
    import tifffile

    for k, lab in label_sets.items():
        p = out / "atlas_cache" / f"{k}_labels_ccf25.tif"
        if not p.exists():
            tifffile.imwrite(p, lab, compression="zlib")

    channels = args.channels or default_channels(sample)
    if not channels:
        raise SystemExit(f"No seg-*.zarr in {sample}; name channels with --channels")
    voxel_mm3 = float(np.prod(orient_info["voxel_mm"]))
    summary = {"sample": str(sample), "registration": str(reg_dir), "threshold": args.threshold,
               "output_layout": "Allen annotation_25.nrrd (AP, DV, LR)" if args.allen_dir else "reference grid",
               "orientation": orient_info, "channels": {}}
    for ch in channels:
        zp = sample / ch
        if not zp.is_dir():
            logger.warning(f"skipping {ch}: not found")
            continue
        t0 = time.time()
        label = ch.replace(".ome.zarr", "").replace(".zarr", "")
        if ch.lower().startswith("seg-"):
            maps = seg_maps(zp, grid_shape, grid_um, args.threshold, args.seg_level)
        else:
            maps = intensity_map(zp, grid_shape, grid_um)
        atlas_maps = {}
        for m, arr in maps.items():
            warped = to_out(warp_to_atlas(arr, reg, reg_dir)).astype(np.float32)
            if warped.shape != tuple(out_shape):
                raise SystemExit(f"warped {m} is {warped.shape}, expected {out_shape}")
            atlas_maps[m] = warped
            tifffile.imwrite(out / f"{label}_{m}_ccf25.tif", warped, compression="zlib")
        for k, lab in label_sets.items():
            rows = region_table(lab, atlas_maps, names.get(k, {}), voxel_mm3,
                                lr_axis=2 if args.allen_dir else None)
            write_csv(out / f"{label}_{k}_regions.csv", rows,
                      extra={"sample": sample.name, "channel": label})
            logger.info(f"{label}: {len(rows)} {k} region rows")
        summary["channels"][ch] = {"maps": list(atlas_maps), "seconds": round(time.time() - t0, 1)}
    summary["seconds"] = round(time.time() - t_start, 1)
    (out / "quantification.json").write_text(json.dumps(summary, indent=2, default=str))
    if out != final_out:
        shutil.copytree(out, final_out, dirs_exist_ok=True)
        shutil.rmtree(out.parent, ignore_errors=True)
    logger.info(f"Done in {summary['seconds']:.0f} s. Results in {final_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
