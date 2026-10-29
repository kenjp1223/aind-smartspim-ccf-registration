#!/usr/bin/env python
r"""
Write the sample-space atlas labels (kim_labels.ome.zarr) as a Neuroglancer
*precomputed* segmentation, the same way the Allen CCFv3 regions layer is made,
so the catalog serves it as one clean precomputed:// source (with region names
on hover) instead of a zarr volume + external segment_properties, which blanks
the viewer.

For each sample it writes, next to the label zarr:
  <sample>/<registration>/sample_space_<labels>/<labels>_annotation_precomputed/
    info, segment_properties/info, and the compresso label pyramid.

Reuses aind_ccf_reg.utils.create_precomputed (the pipeline's own writer), so run
it in the same environment you use for quantify_sample_space.

Usage (from the repo's code/ folder, on the station):
  uv run python write_kim_precomputed.py "<sample>" ["<sample>" ...] ^
      --labels-info "<...\atlas_info_KimRef_FPbasedLabel_v4.0.csv>"

  # all the kim samples at once:
  uv run python write_kim_precomputed.py "\\...\LINCS_series\*_Destripe_DONE" --labels-info "<csv>"
"""
import argparse
import glob
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _patch_intervaltree_monitor():
    """cloudfiles' bandwidth monitor makes a zero-width interval when an I/O finishes in
    under a microsecond (fast local disk); newer intervaltree rejects those and crashes
    CloudVolume. Skip zero-width adds — they carry no meaningful stats."""
    try:
        import intervaltree
        _orig = intervaltree.IntervalTree.addi

        def addi(self, begin, end, data=None):
            if begin == end:
                return None
            return _orig(self, begin, end, data)

        intervaltree.IntervalTree.addi = addi
    except Exception:  # noqa: BLE001  (best effort; harmless if intervaltree isn't present)
        pass


def load_label_level0(zpath):
    """kim_labels.ome.zarr level 0 -> (uint array in z,y,x), voxel size (um, z,y,x)."""
    import tensorstore as ts

    zpath = Path(zpath)
    meta = json.loads((zpath / "zarr.json").read_text(encoding="utf-8"))
    ms = (meta.get("attributes") or meta)["ome"]["multiscales"][0]
    ds0 = ms["datasets"][0]
    scale = next(t["scale"] for t in ds0["coordinateTransformations"] if t["type"] == "scale")
    level0 = zpath / ds0["path"]

    # tensorstore's file driver can't open a Windows UNC path (\\server\share); copy the
    # (highly compressible) label level to a local temp and read that instead.
    tmp = None
    if str(zpath).startswith(("\\\\", "//")):
        tmp = Path(tempfile.mkdtemp(prefix="kimlbl_"))
        shutil.copytree(level0, tmp / "0")
        open_path = (tmp / "0").as_posix()
    else:
        open_path = str(level0).replace("\\", "/")
    try:
        arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": open_path}}).result()
        data = np.asarray(arr.read().result())
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return data, [float(s) for s in scale]


def write_one(sample, args, names):
    from aind_ccf_reg.utils import create_precomputed

    sp_dir = Path(sample) / args.registration / args.space
    zpath = sp_dir / f"{args.labels_name}.ome.zarr"
    if not (zpath / "zarr.json").is_file():
        print(f"  SKIP {Path(sample).name}: no {zpath}")
        return False

    img, vox_um = load_label_level0(zpath)             # z, y, x  (um)
    present = sorted(int(i) for i in np.unique(img) if i)
    regions = {i: (names.get(str(i), (str(i), ""))[0] or str(i)) for i in present}
    descs = {i: (names.get(str(i), ("", ""))[1] or "") for i in present}
    print(f"  {Path(sample).name}: {img.shape} labels, {len(present)} regions, voxel {vox_um} um")

    # precomputed is x,y,z; resolution in nm, same axis order as the volume
    img_xyz = np.ascontiguousarray(np.swapaxes(img, 0, 2)).astype("uint32")
    res_nm = [float(v) * 1000.0 for v in vox_um[::-1]]

    # CloudVolume can't write to a UNC/network path, so build locally then copy
    final = sp_dir / args.out_name
    local = Path(tempfile.mkdtemp(prefix="kimprecomp_")) / args.out_name if str(final).startswith(("\\\\", "//")) else final
    (local / "segment_properties").mkdir(parents=True, exist_ok=True)

    ng_params = {
        "regions": regions,
        "save_path": local.as_posix(),
        "scale_params": {"encoding": "compresso", "compressed_block": [16, 16, 16],
                         "chunk_size": [32, 32, 32], "factors": [2, 2, 2], "num_scales": args.num_scales,
                         "res": res_nm, "dims": list(img_xyz.shape)},
    }
    seg = create_precomputed(ng_params)
    seg.create_segmentation_info()
    seg.build_precomputed_info()
    seg.create_segment_precomputed(img_xyz)

    # add a full-name "description" property alongside the acronym "label"
    info_p = local / "segment_properties" / "info"
    info = json.loads(info_p.read_text())
    if any(descs.values()):
        info["inline"]["properties"].append(
            {"id": "description", "type": "description",
             "values": [descs[i] for i in present]})
        info_p.write_text(json.dumps(info, indent=2))

    if local != final:
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        shutil.copytree(local, final, dirs_exist_ok=True)
        shutil.rmtree(local.parent, ignore_errors=True)
    print(f"    wrote {final}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="sample folders (globs ok)")
    ap.add_argument("--labels-info", required=True, help="CSV with region id/acronym/name (the Kim atlas info)")
    ap.add_argument("--registration", default="registration_ccf25_aind/Ex_488_Ch0_stitched")
    ap.add_argument("--space", default="sample_space_kim", help="sample-space folder name")
    ap.add_argument("--labels-name", default="kim_labels", help="label zarr basename (writes <name>.ome.zarr)")
    ap.add_argument("--out-name", default="kim_annotation_precomputed")
    ap.add_argument("--num-scales", type=int, default=3)
    args = ap.parse_args()

    _patch_intervaltree_monitor()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from quantify_registered import read_names
    except ImportError:
        sys.exit("run this from the repo's code/ folder (needs quantify_registered.read_names)")
    names = read_names(args.labels_info)   # {id: (acronym, name)}
    if not names:
        sys.exit(f"no regions read from {args.labels_info}")
    print(f"{len(names)} regions in {Path(args.labels_info).name}")

    samples = []
    for s in args.samples:
        samples += [g for g in glob.glob(s)] or [s]
    samples = [s for s in samples if Path(s).is_dir()]
    if not samples:
        sys.exit("no sample folders matched")

    ok = sum(write_one(s, args, names) for s in sorted(samples))
    print(f"\n{ok}/{len(samples)} samples written. Rescan the catalog to pick up the new layer.")


if __name__ == "__main__":
    main()
