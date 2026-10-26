"""
Register one SmartSPIM OME-Zarr channel to a custom atlas (e.g. the Kim lab
unified atlas) with the AIND template-based pipeline, without Code Ocean.

The AIND pipeline is two-stage (brain -> SPIM template -> CCF). Here the
atlas's own template is the registration target and stage 2 is an identity
transform, so: brain -> atlas template (rigid + affine + SyN), then the atlas
labels are warped back into the brain's own voxel grid.

Nothing on disk is reoriented. The sample's and the atlas's axis directions
are described with --sample-axes / --atlas-axes, and the pipeline rotates an
in-memory copy (utils.check_orientation), then rotates the labels back.

Usage (from the repo's code/ folder, inside the uv env):

  # 1) check orientation only: writes orientation_check.png, no registration
  uv run python run_custom_atlas.py "Z:/.../Ex_488_Ch0_stitched.ome.zarr" \
      --atlas-template ".../average_template_25.nrrd" \
      --atlas-labels   ".../annotation_25.nrrd" --atlas-res 25 --check

  # other atlases: add --atlas-info <CSV id,acronym,name>; anisotropic tif atlases
  # take three sizes, e.g. --atlas-res 50 20 20 (tif z y x)

  # 2) the real run: same command without --check

Outputs (default: <sample folder>/registration_<atlas>/<channel>/):
  ls_to_template_SyN_*            transforms (brain <-> atlas)
  moved_ls_to_ccf.nii.gz          brain warped into atlas space
  OMEZarr/image.zarr              same, as OME-Zarr
  annotation_precomputed/         atlas labels in the brain's voxel grid
                                  (Neuroglancer precomputed segmentation,
                                  region names on hover). Segment IDs are
                                  consecutive indices: label_map.csv maps
                                  them to the atlas's own IDs.
  metadata/registration_metadata/ QC images (reg_*.png, prep_*.png, ...)
  run_config.json                 everything this run used
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

AXIS_WORDS = {"anterior", "posterior", "superior", "inferior", "left", "right"}
OPPOSITE = {
    "anterior": "posterior",
    "posterior": "anterior",
    "superior": "inferior",
    "inferior": "superior",
    "left": "right",
    "right": "left",
}

# SmartSPIM as acquired in the Stuber lab: z = ventral -> dorsal,
# y = anterior at the top of each plane, x = left -> right (verify with --check)
DEFAULT_SAMPLE_AXES = "inferior_to_superior,anterior_to_posterior,left_to_right"
# Coronal atlas stack read as (plane, row, column)
DEFAULT_ATLAS_AXES = "anterior_to_posterior,superior_to_inferior,left_to_right"
# (the Allen CCF .nrrd files are stored the same way: AP, DV, LR)
CCF_ANNOTATION_MAP = Path(__file__).resolve().parent / "aind_ccf_reg" / "ccf_files" / "annotation_map.json"

logger = logging.getLogger("run_custom_atlas")


def setup_logging(out):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "registration.log", encoding="utf-8"),
        ],
        force=True,
    )
    # a dictConfig() inside the package's imports disables existing loggers
    logger.disabled = False
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# axis descriptions
# --------------------------------------------------------------------------
def parse_axes(text):
    """'a_to_b,c_to_d,e_to_f' -> ['a_to_b', ...] (validated, lower case)."""
    dirs = [d.strip().lower() for d in text.split(",")]
    if len(dirs) != 3:
        raise SystemExit(f"Need three axis directions, got: {text!r}")
    seen = set()
    for d in dirs:
        parts = d.split("_to_")
        if len(parts) != 2 or parts[0] not in AXIS_WORDS or OPPOSITE[parts[0]] != parts[1]:
            raise SystemExit(
                f"Bad axis direction {d!r}. Use e.g. anterior_to_posterior, "
                "inferior_to_superior, left_to_right"
            )
        pair = frozenset(parts)
        if pair in seen:
            raise SystemExit(f"Axis {d!r} appears twice in {text!r}")
        seen.add(pair)
    return dirs


def flip_direction(d):
    a, b = d.split("_to_")
    return f"{b}_to_{a}"


def sample_orientation(sample_dirs):
    """The acquisition 'axes' list the pipeline expects (dimension = array axis)."""
    names = ["Z", "Y", "X"]
    return [
        {"name": names[i], "dimension": i, "direction": d.capitalize()}
        for i, d in enumerate(sample_dirs)
    ]


def template_orientations(atlas_dirs, sample_dirs):
    """
    {direction: atlas array axis}. check_orientation looks up each sample
    direction (or its reverse) in this dict, so keys use the atlas's words.
    """
    out = {d: i for i, d in enumerate(atlas_dirs)}
    for s in sample_dirs:
        if s not in out and flip_direction(s) not in out:
            raise SystemExit(f"Sample direction {s} has no matching atlas axis")
    return out


# --------------------------------------------------------------------------
# OME-Zarr metadata
# --------------------------------------------------------------------------
UNIT_TO_MM = {
    "micrometer": 1e-3,
    "micron": 1e-3,
    "um": 1e-3,
    "millimeter": 1.0,
    "mm": 1.0,
    "nanometer": 1e-6,
    "nm": 1e-6,
}


def read_multiscales(zarr_path):
    """Return (multiscales dict, zarr format) for OME-Zarr 0.4 or 0.5."""
    p = Path(zarr_path)
    if (p / "zarr.json").is_file():
        meta = json.loads((p / "zarr.json").read_text())
        attrs = meta.get("attributes", {})
        ms = attrs.get("ome", attrs).get("multiscales")
        fmt = 3
    elif (p / ".zattrs").is_file():
        ms = json.loads((p / ".zattrs").read_text()).get("multiscales")
        fmt = 2
    else:
        raise SystemExit(f"{zarr_path} is not a zarr group (no zarr.json / .zattrs)")
    if not ms:
        raise SystemExit(f"{zarr_path} has no OME multiscales metadata")
    return ms[0], fmt


def level_spacings_mm(ms):
    """[(path, (z, y, x) voxel size in mm)] for each level, last three axes."""
    axes = ms.get("axes", [])
    units = [UNIT_TO_MM.get(str(a.get("unit", "micrometer")).lower(), 1e-3) for a in axes][-3:]
    if len(units) < 3:
        units = [1e-3] * 3
    out = []
    for ds in ms["datasets"]:
        scale = None
        for t in ds.get("coordinateTransformations", []):
            if t.get("type") == "scale":
                scale = t["scale"]
        if scale is None:
            raise SystemExit(f"Level {ds['path']} has no scale transform")
        out.append((ds["path"], tuple(float(s) * u for s, u in zip(scale[-3:], units))))
    return out


def pick_level(levels, atlas_res_mm, forced=None):
    """
    Coarsest level that is not coarser than the atlas's finest axis
    (largest sample voxel <= smallest atlas voxel).
    """
    if forced is not None:
        for path, sp in levels:
            if str(path) == str(forced):
                return path, sp
        raise SystemExit(f"Level {forced} not in {[p for p, _ in levels]}")
    ok = [(p, sp) for p, sp in levels if max(sp) <= atlas_res_mm * 1.0001]
    if not ok:
        return levels[0]
    return max(ok, key=lambda x: max(x[1]))


# --------------------------------------------------------------------------
# atlas files
# --------------------------------------------------------------------------
def tif_voxel_um(path):
    """
    Voxel size (axis 0, 1, 2) in um stored in an ImageJ tif: 'spacing' for the
    plane axis, XResolution/YResolution for rows/columns. None if incomplete.
    """
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tf:
            ij = tf.imagej_metadata or {}
            unit = str(ij.get("unit", "")).lower().replace("\\u00b5", "u").replace("µ", "u")
            factor = {"um": 1, "micron": 1, "microns": 1, "mm": 1000, "nm": 1e-3}.get(unit)
            tags = tf.pages[0].tags
            if not (factor and ij.get("spacing") and "XResolution" in tags):
                return None
            xn, xd = tags["XResolution"].value
            yn, yd = tags["YResolution"].value if "YResolution" in tags else (xn, xd)
            if xn and yn:
                return (float(ij["spacing"]) * factor, yd / yn * factor, xd / xn * factor)
    except Exception:
        pass
    return None


def read_volume(path, labels=False):
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        return np.asarray(tifffile.imread(path)), None
    import ants

    # labels: read as integers. Allen CCF IDs go up to 614454277, beyond what
    # float32 (ANTs' default pixel type) holds exactly.
    img = ants.image_read(path, pixeltype="unsigned int" if labels else "float")
    return img.numpy(), img


def atlas_to_nifti(src, dst, spacing_mm, labels=False):
    """
    Write the atlas volume as NIfTI with identity direction and zero origin,
    keeping the array axis order exactly as stored (numpy axis i = ANTs axis i).
    spacing_mm: voxel size of array axes 0, 1, 2 (anisotropic is fine).
    """
    import ants

    arr, img = read_volume(src, labels=labels)
    if arr.ndim != 3:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise SystemExit(f"{src}: expected a 3D volume, got shape {arr.shape}")
    if labels:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.rint(arr)
        arr = arr.astype(np.uint64)
        # ANTs warps labels as float32, exact only up to 2**24. Large atlas IDs
        # (Allen CCF) are replaced by consecutive indices for the warp; the
        # returned table maps them back (0 stays 0 = outside the brain).
        ids, compact = np.unique(arr, return_inverse=True)
        if ids[0] != 0:
            ids = np.concatenate([[0], ids])
            compact = compact + 1
        compact = compact.reshape(arr.shape).astype(np.float32)
        out = ants.from_numpy(compact, spacing=tuple(spacing_mm))
        ants.image_write(out, str(dst))
        return arr.shape, {i: int(a) for i, a in enumerate(ids)}
    else:
        # Same 2-98 percentile scaling the pipeline applies to the brain, so the
        # template and the brain share an intensity range (QC plots assume 0-1.5).
        arr = arr.astype(np.float32)
        nz = arr[arr > 0]
        if nz.size:
            p0, p1 = np.percentile(nz, [2, 98])
            if p1 > p0:
                arr = (np.maximum(arr, p0) - p0) / (p1 - p0)
        out = ants.from_numpy(arr, spacing=tuple(spacing_mm))
    ants.image_write(out, str(dst))
    return arr.shape, arr.dtype


def read_regions(csv_path, id_col=None, name_col=None, desc_col=None):
    """
    {label id (str): acronym}, {label id: full name} from the atlas info CSV,
    or from a JSON {id: acronym} such as the repo's CCF annotation_map.json.
    """
    if str(csv_path).lower().endswith(".json"):
        acr = {str(k): str(v) for k, v in json.loads(Path(csv_path).read_text()).items()}
        logger.info(f"Regions: {len(acr)} from {csv_path}")
        return acr, {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(f, dialect=dialect))
    if not rows:
        raise SystemExit(f"{csv_path} has no rows")
    cols = list(rows[0].keys())
    low = {c.lower().strip(): c for c in cols}

    def find(explicit, candidates):
        if explicit:
            if explicit not in cols:
                raise SystemExit(f"Column {explicit!r} not in {cols}")
            return explicit
        for c in candidates:
            if c in low:
                return low[c]
        return None

    id_c = find(id_col, ["id", "label", "label_id", "structure_id", "index", "value"])
    name_c = find(name_col, ["acronym", "abbreviation", "abbr", "short_name", "name"])
    desc_c = find(desc_col, ["name", "full_name", "structure_name", "region"])
    if id_c is None or name_c is None:
        raise SystemExit(
            f"Could not guess the id / name columns in {csv_path}: {cols}. "
            "Use --id-col and --name-col."
        )
    if desc_c == name_c:
        desc_c = None
    acr, desc = {}, {}
    for r in rows:
        raw = str(r.get(id_c, "")).strip()
        if not raw:
            continue
        try:
            k = str(int(float(raw)))
        except ValueError:
            continue
        acr[k] = str(r.get(name_c, "")).strip()
        if desc_c:
            desc[k] = str(r.get(desc_c, "")).strip()
    logger.info(f"Regions: {len(acr)} from {csv_path} (id={id_c}, label={name_c}, desc={desc_c})")
    return acr, desc


def write_identity(path):
    import ants

    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    ants.write_transform(tx, str(path))


# --------------------------------------------------------------------------
# orientation check (no registration)
# --------------------------------------------------------------------------
def orientation_check(img_array, spacing_mm, sample_axes, tmpl_orient, atlas_nii, png):
    import ants
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aind_ccf_reg.utils import check_orientation

    img_out, in_mat, _ = check_orientation(img_array.astype(np.float32), sample_axes, tmpl_orient)
    order = np.where(in_mat)[1]
    sp = [0.0, 0.0, 0.0]
    for src_axis, dst_axis in enumerate(order):
        sp[dst_axis] = spacing_mm[src_axis]
    atlas = ants.image_read(str(atlas_nii))
    a = atlas.numpy()
    asp = atlas.spacing

    inv = {v: k for k, v in tmpl_orient.items()}
    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    for ax_i in range(3):
        others = [k for k in range(3) if k != ax_i]
        for col, (vol, vsp, title) in enumerate(
            [(img_out, sp, "sample (rotated in memory)"), (a, asp, "atlas template")]
        ):
            sl = np.take(vol, vol.shape[ax_i] // 2, axis=ax_i)
            lo, hi = np.percentile(sl, [1, 99.5])
            ext = [0, sl.shape[1] * vsp[others[1]], sl.shape[0] * vsp[others[0]], 0]
            axs[ax_i, col].imshow(sl, cmap="gray", vmin=lo, vmax=hi, extent=ext)
            axs[ax_i, col].set_title(f"{title}\nmid-slice of axis {ax_i}", fontsize=9)
            axs[ax_i, col].set_ylabel(f"axis {others[0]}: {inv[others[0]]}", fontsize=8)
            axs[ax_i, col].set_xlabel(f"axis {others[1]}: {inv[others[1]]} (mm)", fontsize=8)
    fig.suptitle("Both columns should show the brain the same way round", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def improve_segment_properties(precomputed_dir, acr, desc):
    """Add the full region name (and atlas ID) as a description next to the label."""
    if not desc:
        return
    p = Path(precomputed_dir) / "segment_properties" / "info"
    if not p.is_file():
        return
    info = json.loads(p.read_text())
    ids = info["inline"]["ids"]
    info["inline"]["properties"].append(
        {"id": "description", "type": "description", "values": [desc.get(i, "") for i in ids]}
    )
    p.write_text(json.dumps(info, indent=2))


TRANSFORMS = (
    "ls_to_template_SyN_0GenericAffine.mat",
    "ls_to_template_SyN_1Warp.nii.gz",
    "ls_to_template_SyN_1InverseWarp.nii.gz",
)


def patch_cloudfiles():
    """
    Windows' monotonic clock ticks every ~16 ms, so a fast local file read in
    cloudfiles' transfer monitor can start and end on the same tick, and its
    IntervalTree refuses the zero-length interval. Give it 1 us instead.
    """
    try:
        import time

        from cloudfiles import monitoring
    except ImportError:
        return

    def end_io(self, flight_id, num_bytes):
        end_us = int(time.monotonic() * 1e6)
        with self._lock:
            start_us = int(self._in_flight.pop(flight_id) * 1e6)
            self._in_flight_bytes -= num_bytes
            self._intervaltree.addi(start_us, max(end_us, start_us + 1), [flight_id, num_bytes])
            self._total_bytes_landed += num_bytes

    monitoring.TransmissionMonitor.end_io = end_io


def resumable(register_cls):
    """Register subclass that reuses finished transforms instead of recomputing them."""
    import ants

    class ResumableRegister(register_cls):
        reuse = True

        def register_to_template(self, ants_fixed, ants_moving, moving_mask=None):
            rf = Path(self.args["results_folder"])
            if self.reuse and all((rf / t).is_file() for t in TRANSFORMS):
                logger.info(f"Reusing the transforms in {rf} (use --redo to recompute)")
                moved = ants.apply_transforms(
                    fixed=ants_fixed,
                    moving=ants_moving,
                    transformlist=[str(rf / TRANSFORMS[1]), str(rf / TRANSFORMS[0])],
                )
                self._qc_reg(
                    ants_moving,
                    ants_fixed,
                    moved,
                    moved_path=self.args["ants_params"].get("moved_to_template_path"),
                    figpath_name="reg_to_template",
                )
                return moved
            return super().register_to_template(ants_fixed, ants_moving, moving_mask)

    return ResumableRegister


def deliver(work, final, on_network, cleanup=True):
    """Copy a locally produced output folder to its network destination."""
    if not on_network or not work.is_dir():
        return
    logger.info(f"Copying results to {final}")
    shutil.copytree(work, final, dirs_exist_ok=True)
    if cleanup:
        shutil.rmtree(work.parent, ignore_errors=True)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr", help="channel OME-Zarr, e.g. ...\\Ex_488_Ch0_stitched.ome.zarr")
    ap.add_argument("--atlas-template", required=True, help="atlas template volume (tif or nii)")
    ap.add_argument("--atlas-labels", required=True, help="atlas label volume, same grid")
    ap.add_argument("--atlas-info", default=str(CCF_ANNOTATION_MAP),
                    help="CSV (id, acronym, name) or JSON {id: acronym} for the label IDs "
                         "(default: the repo's Allen CCF annotation_map.json)")
    ap.add_argument("--atlas-res", type=float, nargs="+", metavar="UM",
                    help="atlas voxel size in um: one value (isotropic) or three for "
                         "array axes 0 1 2, i.e. the tif's z y x (e.g. 50 20 20 for 50 um "
                         "coronal sections at 20 um in-plane)")
    ap.add_argument("--atlas-axes", default=DEFAULT_ATLAS_AXES,
                    help=f"directions of atlas array axes 0,1,2 (default {DEFAULT_ATLAS_AXES})")
    ap.add_argument("--sample-axes", default=DEFAULT_SAMPLE_AXES,
                    help=f"directions of zarr axes z,y,x (default {DEFAULT_SAMPLE_AXES})")
    ap.add_argument("--level", help="zarr level to register (default: coarsest not coarser than atlas)")
    ap.add_argument("--out", help="output folder (default: <sample>/registration_<atlas>/<channel>)")
    ap.add_argument("--id-col"), ap.add_argument("--name-col"), ap.add_argument("--desc-col")
    ap.add_argument("--syn-iterations", help="e.g. 200,100,25,3")
    ap.add_argument("--redo", action="store_true",
                    help="recompute the registration even if transforms from an earlier run exist")
    ap.add_argument("--check", action="store_true",
                    help="only write orientation_check.png and the plan; no registration")
    args = ap.parse_args(argv)

    zarr_path = Path(args.zarr.rstrip("/\\"))
    if not zarr_path.is_dir():
        raise SystemExit(f"Not found: {zarr_path}")
    channel = zarr_path.name
    for suf in (".ome.zarr", ".zarr"):
        if channel.endswith(suf):
            channel = channel[: -len(suf)]
            break
    atlas_name = Path(args.atlas_template).stem.split(".")[0]

    out = Path(args.out) if args.out else zarr_path.parent / f"registration_{atlas_name}" / channel
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas").mkdir(exist_ok=True)

    setup_logging(out)

    # tensorstore and CloudVolume can't write to Windows network (UNC) paths,
    # so a network output folder is produced locally and copied over at the end.
    final = out
    on_network = str(final).startswith(("\\\\", "//"))
    if on_network:
        import tempfile

        out = Path(tempfile.mkdtemp(prefix="ccfreg_")) / final.name
        (out / "atlas").mkdir(parents=True)
        logger.info(f"Output is on a network path: working in {out}, copying to {final} at the end")

    sample_dirs = parse_axes(args.sample_axes)
    atlas_dirs = parse_axes(args.atlas_axes)
    tmpl_orient = template_orientations(atlas_dirs, sample_dirs)
    sample_axes = sample_orientation(sample_dirs)

    # ---- atlas -> NIfTI (once per output folder) ----
    tmpl_src = Path(args.atlas_template)
    if args.atlas_res is None:
        if tmpl_src.suffix.lower() in (".tif", ".tiff"):
            um = tif_voxel_um(tmpl_src)
            if um is None:
                raise SystemExit(
                    f"{tmpl_src} does not store its voxel size; pass --atlas-res "
                    "(um, z y x of the tif, the values your ClearMap atlas config uses)"
                )
            logger.info(f"Atlas voxel size from tif metadata: {um} um")
            atlas_sp_mm = tuple(u / 1000.0 for u in um)
        else:
            import ants

            sp = [float(s) for s in ants.image_header_info(str(tmpl_src))["spacing"]]
            # Allen .nrrd headers give micrometres (25), NIfTI usually mm (0.025)
            if min(sp) > 1:
                logger.info(f"Header voxel size {sp} read as um")
                sp = [x / 1000.0 for x in sp]
            atlas_sp_mm = tuple(sp)
    elif len(args.atlas_res) == 1:
        atlas_sp_mm = (args.atlas_res[0] / 1000.0,) * 3
    elif len(args.atlas_res) == 3:
        atlas_sp_mm = tuple(r / 1000.0 for r in args.atlas_res)
    else:
        raise SystemExit("--atlas-res takes one value or three (z y x of the atlas tif)")
    atlas_res_mm = min(atlas_sp_mm)  # finest axis: decides which zarr level to use
    tmpl_nii = out / "atlas" / "template.nii.gz"
    lab_nii = out / "atlas" / "labels.nii.gz"
    tshape, _ = atlas_to_nifti(tmpl_src, tmpl_nii, atlas_sp_mm)
    lshape, id_map = atlas_to_nifti(args.atlas_labels, lab_nii, atlas_sp_mm, labels=True)
    if tshape != lshape:
        raise SystemExit(f"Template {tshape} and labels {lshape} are not the same grid")
    identity = out / "atlas" / "identity.mat"
    write_identity(identity)
    names, full = read_regions(args.atlas_info, args.id_col, args.name_col, args.desc_col)
    # Region tables keyed by the compact index used in the warped label volume
    acr, desc = {}, {}
    missing = 0
    with open(out / "label_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "atlas_id", "acronym", "name"])
        for c_id, a_id in id_map.items():
            key = str(a_id)
            if a_id and key not in names:
                missing += 1
            acr[str(c_id)] = names.get(key, "outside" if a_id == 0 else key)
            desc[str(c_id)] = f"{full.get(key, acr[str(c_id)])} (atlas id {a_id})"
            w.writerow([c_id, a_id, acr[str(c_id)], full.get(key, "")])
    logger.info(f"{len(id_map) - 1} labels in the atlas volume; segment IDs -> atlas IDs in "
                f"label_map.csv" + (f"; {missing} IDs have no name in {args.atlas_info}" if missing else ""))

    # ---- sample level ----
    ms, fmt = read_multiscales(zarr_path)
    levels = level_spacings_mm(ms)
    level, spacing = pick_level(levels, atlas_res_mm, args.level)
    level_path = zarr_path / str(level)
    logger.info(f"Sample {zarr_path} (zarr v{fmt}); levels: "
                + ", ".join(f"{p}: {tuple(round(s * 1000, 2) for s in sp)} um" for p, sp in levels))
    logger.info(f"Registering level {level}, voxel {tuple(round(s * 1000, 2) for s in spacing)} um "
                f"(z, y, x) to atlas at {tuple(round(s * 1000, 2) for s in atlas_sp_mm)} um "
                f"(axes 0, 1, 2), template {tshape}")

    # reference_res only picks the SyN iteration schedule (finest atlas axis)
    reference_res = int(round(atlas_res_mm * 1000))
    reg_folder = out / "metadata" / "registration_metadata"
    reg_folder.mkdir(parents=True, exist_ok=True)
    precompute_path = out / "annotation_precomputed"
    ng_path = precompute_path
    (ng_path / "segment_properties").mkdir(parents=True, exist_ok=True)

    cfg = {
        "input_data": str(zarr_path.parent),
        "input_channel": channel,
        "input_image_path": str(level_path),
        "additional_channels": [],
        "input_scale": int(level) if str(level).isdigit() else 0,
        "input_orientation": sample_axes,
        "bucket_path": "local",
        "template_path": str(tmpl_nii),
        "ccf_reference_path": str(tmpl_nii),
        "template_to_ccf_transform_path": [str(identity)],
        "ccf_to_template_transform_path": [str(identity)],
        "ccf_annotation_to_template_moved_path": str(lab_nii),
        "reference_res": reference_res,
        "output_data": str(out / "OMEZarr"),
        "metadata_folder": str(out / "metadata"),
        "code_url": "https://github.com/AllenNeuralDynamics/aind-ccf-registration",
        "results_folder": str(out),
        "reg_folder": str(reg_folder),
        "prep_params": {
            k: str(reg_folder / v)
            for k, v in {
                "rawdata_figpath": "prep_zarr_img.jpg",
                "rawdata_path": "prep_zarr_img.nii.gz",
                "resample_figpath": "prep_resampled_zarr_img.jpg",
                "resample_path": "prep_resampled_zarr_img.nii.gz",
                "mask_figpath": "prep_mask.jpg",
                "mask_path": "prep_mask.nii.gz",
                "n4bias_figpath": "prep_n4bias.jpg",
                "n4bias_path": "prep_n4bias.nii.gz",
                "percNorm_figpath": "prep_percNorm.jpg",
                "percNorm_path": "prep_percNorm.nii.gz",
            }.items()
        },
        "ants_params": {
            "spacing": tuple(spacing),
            "unit": "millimetre",
            "template_orientations": tmpl_orient,
            "rigid_path": str(reg_folder / "moved_rigid.nii.gz"),
            "affine_path": str(reg_folder / "moved_affine.nii.gz"),
            "moved_to_template_path": str(reg_folder / "moved_ls_to_template.nii.gz"),
            "moved_to_ccf_path": str(out / "moved_ls_to_ccf.nii.gz"),
            "ccf_anno_to_brain_path": str(reg_folder / "moved_ccf_anno_to_ls.nii.gz"),
            # voxel size (um) of OMEZarr/image.zarr (the brain in atlas space).
            # register.run writes the atlas-grid array with its axes reversed,
            # so the sizes go in reverse atlas-axis order.
            "output_voxel_size": tuple(s * 1000 for s in atlas_sp_mm[::-1]),
        },
        "OMEZarr_params": {
            "clevel": 1,
            "compressor": "zstd",
            "chunks": (64, 64, 64),
            "n_lvls": 4,
            "shard_size": (512, 512, 512),
            "chunk_size": (128, 128, 128),
            "scale_factor": (2, 2, 2),
        },
        "ng_params": {
            # forward slashes: CloudVolume builds a file:// URL from this
            "save_path": ng_path.as_posix(),
            "regions": acr,
            "scale_params": {
                "encoding": "compresso",
                "compressed_block": [16, 16, 16],
                "chunk_size": [32, 32, 32],
                "factors": [2, 2, 2],
                "num_scales": 3,
            },
        },
    }
    if args.syn_iterations:
        cfg["ants_params"]["syn_iterations"] = [int(x) for x in args.syn_iterations.split(",")]

    record = {k: v for k, v in cfg.items() if k != "ng_params"}
    record["ng_params"] = {**cfg["ng_params"], "regions": f"{len(acr)} regions from {args.atlas_info}"}
    record["sources"] = {
        "zarr": str(zarr_path),
        "level": str(level),
        "atlas_template": str(args.atlas_template),
        "atlas_labels": str(args.atlas_labels),
        "atlas_info": str(args.atlas_info),
        "atlas_voxel_um_axes_012": [s * 1000 for s in atlas_sp_mm],
        "sample_axes": sample_dirs,
        "atlas_axes": atlas_dirs,
    }
    (out / "run_config.json").write_text(json.dumps(record, indent=2, default=str))

    from aind_ccf_reg import register  # heavy imports after argument checks

    setup_logging(final)  # the package's imports reset logging handlers

    img_array = register.Register._Register__read_zarr_image(None, level_path) if args.check else None
    if args.check:
        png = out / "orientation_check.png"
        orientation_check(img_array, spacing, sample_axes, tmpl_orient, tmpl_nii, png)
        logger.info(f"Wrote {png}. Check it, then run again without --check.")
        deliver(out, final, on_network)
        return 0

    # args=[] stops argschema from parsing this script's own command line
    patch_cloudfiles()
    if on_network and not args.redo:
        # transforms from an earlier (possibly failed) run in the final folder
        for t in TRANSFORMS:
            if (final / t).is_file() and not (out / t).is_file():
                shutil.copy2(final / t, out / t)
    reg = resumable(register.Register)(input_data=cfg, args=[])
    reg.reuse = not args.redo
    try:
        reg.run()
    except BaseException:
        # keep whatever was produced (transforms, QC) so a rerun can reuse it
        deliver(out, final, on_network, cleanup=False)
        raise
    improve_segment_properties(ng_path, acr, desc)
    deliver(out, final, on_network)
    logger.info(f"Done. Results in {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
