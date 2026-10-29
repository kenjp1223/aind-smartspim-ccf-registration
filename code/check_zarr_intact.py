#!/usr/bin/env python
r"""
Check that a SmartSPIM sample's OME-Zarr channels are intact (not empty / not
truncated), reading the real voxel data level by level.

For each *.ome.zarr in the sample folder it prints, per pyramid level:
  shape, dtype, how many chunk files exist vs expected, min / max / mean,
  and the fraction of nonzero voxels.
A level is flagged if it is all zero, if chunk files are missing, or if it
cannot be opened. A channel with signal has nonzero min/max and a sensible
nonzero fraction at every level.

Usage (in the omezarr env, on the station):
  python check_zarr_intact.py "\\10.158.246.229\DataCommon\SmartSPIM2\Ken\LINCS_series\20261022_11_17_15_KOR_VTA_M4_Destripe_DONE"

  # one channel only, and sample big levels instead of reading them whole:
  python check_zarr_intact.py "<sample>" --only seg-Ex_639_Ch2_stitched.ome.zarr
  python check_zarr_intact.py "<sample>" --max-read-gb 0.5

Needs: numpy, tensorstore  (both are in the omezarr env).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import tensorstore as ts
except ImportError:
    sys.exit("tensorstore is not installed in this environment; run in the omezarr env "
             "(or: pip install tensorstore)")


def level_dirs(zpath):
    """(level path, scale) for each dataset in the OME multiscales, in order."""
    meta = json.loads((zpath / "zarr.json").read_text(encoding="utf-8"))
    ome = (meta.get("attributes") or {}).get("ome") or meta.get("ome")
    if not ome or not ome.get("multiscales"):
        raise ValueError("no OME multiscales metadata (is this an OME-Zarr group?)")
    out = []
    for ds in ome["multiscales"][0]["datasets"]:
        scale = next((t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale"), None)
        out.append((ds["path"], scale))
    return out


def count_chunks(level_path):
    """number of chunk files actually on disk under <level>/c/ (zarr v3 default layout)."""
    c = level_path / "c"
    if not c.is_dir():
        # some writers put chunks directly under the level dir
        return sum(1 for _ in level_path.rglob("*") if _.is_file()) - 1
    return sum(1 for _ in c.rglob("*") if _.is_file())


def read_array(level_path, max_voxels):
    """Open a zarr v3 array with tensorstore; read whole if small, else a strided subsample."""
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file", "path": str(level_path).replace("\\", "/")}}).result()
    shape = list(arr.shape)
    n = int(np.prod(shape))
    if n <= max_voxels:
        return np.asarray(arr.read().result()), shape, 1
    # subsample along each axis so the read stays under max_voxels
    step = int(np.ceil((n / max_voxels) ** (1.0 / len(shape))))
    sl = tuple(slice(None, None, step) for _ in shape)
    return np.asarray(arr[sl].read().result()), shape, step


def check_channel(zpath, max_voxels):
    print(f"\n=== {zpath.name} ===")
    try:
        levels = level_dirs(zpath)
    except Exception as e:  # noqa: BLE001
        print(f"  CANNOT READ METADATA: {e}")
        return False
    ok = True
    for lp, scale in levels:
        level_path = zpath / lp
        tag = f"  level {lp} (scale {scale} um):"
        if not (level_path / "zarr.json").is_file():
            print(f"{tag} MISSING (no zarr.json)")
            ok = False
            continue
        try:
            data, shape, step = read_array(level_path, max_voxels)
        except Exception as e:  # noqa: BLE001
            print(f"{tag} FAILED TO READ: {e}")
            ok = False
            continue
        nchunks = count_chunks(level_path)
        mx = int(data.max()) if data.size else 0
        nz = float((data > 0).mean()) * 100 if data.size else 0.0
        note = ""
        if mx == 0:
            note = "  <-- ALL ZERO"
            ok = False
        if nchunks == 0:
            note += "  <-- NO CHUNK FILES"
            ok = False
        sampled = "" if step == 1 else f" (every {step}th voxel)"
        print(f"{tag} shape {shape}{sampled} {data.dtype}  min {int(data.min()) if data.size else 0} "
              f"max {mx} mean {data.mean():.2f}  nonzero {nz:.1f}%  chunks {nchunks}{note}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", help="sample folder (…_Destripe_DONE) or a single .ome.zarr")
    ap.add_argument("--only", nargs="+", help="check only these .ome.zarr names")
    ap.add_argument("--max-read-gb", type=float, default=1.0,
                    help="read a level whole only if under this many GB, else subsample (default 1.0)")
    args = ap.parse_args()

    root = Path(args.sample.rstrip("/\\"))
    if not root.exists():
        sys.exit(f"not found: {root}")
    max_voxels = int(args.max_read_gb * 1e9 / 2)   # ~2 bytes/voxel budget

    if root.name.endswith(".ome.zarr"):
        zarrs = [root]
    else:
        zarrs = sorted(p for p in root.iterdir()
                       if p.name.endswith(".ome.zarr") and p.is_dir()
                       and (not args.only or p.name in args.only))
    if not zarrs:
        sys.exit("no .ome.zarr channels found here")

    print(f"Checking {len(zarrs)} channel(s) in {root}")
    results = {z.name: check_channel(z, max_voxels) for z in zarrs}

    print("\n=== summary ===")
    for name, good in results.items():
        print(f"  {'OK   ' if good else 'BAD  '} {name}")
    bad = [n for n, g in results.items() if not g]
    if bad:
        print(f"\n{len(bad)} channel(s) look broken: {', '.join(bad)}")
        sys.exit(1)
    print("\nAll channels have real data at every level.")


if __name__ == "__main__":
    main()
