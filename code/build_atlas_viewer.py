"""
Build a Neuroglancer viewer that compares brains in atlas space.

Collects the maps written by warp_signal_to_atlas.py (all on the same 25 um
reference grid, e.g. the Kim-to-Allen annotation) into one folder of
OME-Zarr volumes plus ready-made Neuroglancer views:

  <out>/
    template.ome.zarr               CCF average template (grey background)
    regions.ome.zarr                reference labels as a segmentation, with
                                    region names (hover) from --labels-info
    samples/<sample>.ome.zarr       one map per brain (default: seg fraction)
    groups/<group>_mean.ome.zarr    mean over each --group
    groups/<A>_minus_<B>.ome.zarr   difference of the first two groups
    views.json                      Neuroglancer states (overlay, side by side)
    index.html                      links that open the views
  Serve it with serve_atlas_viewer.py (any station on the network can open it).

Every volume has the same 25 um grid and origin, so they overlay exactly.

Left/right: injections are often counterbalanced between hemispheres. By
default every brain whose signal is mostly on the other side is mirrored
(about the atlas midline, found from the reference labels) so all brains have
their strong side on the same side before group means are taken:
  --align-side auto     the side most brains already have (default)
  --align-side low|high the low- or high-index side of the left/right axis
  --align-side none     no automatic mirroring
  --flip TEXT / --keep TEXT   force mirroring / no mirroring of the brains
                              whose name contains TEXT (repeatable)
The index page lists what was mirrored and has a "mirror" switch per brain
for the per-brain layers (group means keep the build-time choice; rebuild with
--flip/--keep to change those).

Usage (from the repo's code/ folder):
  uv run python build_atlas_viewer.py <sample> <sample> ... ^
      --reference "<Kim_to_allen25_ann_fwdtransforms.tif>" ^
      --labels-info "<atlas_info_KimRef_FPbasedLabel_v4.0.csv>" ^
      --group Calcr=Calcr_NAc --group Pdyn=Pdyn_F ^
      --out "\\\\server\\...\\LINCS_series\\atlas_viewer"
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("atlas_viewer")
GROUP_COLORS = ["#ff00ff", "#00ff00", "#00c8ff", "#ffa000", "#ff3030", "#ffff00"]


# --------------------------------------------------------------------------
# OME-Zarr writing (Zarr v3, the layout the lab's Neuroglancer reads)
# --------------------------------------------------------------------------
def _write_array(path, arr, dtype):
    import tensorstore as ts

    ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(Path(path).resolve()).replace("\\", "/")},
        "metadata": {
            "shape": list(arr.shape), "data_type": dtype,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64, 64]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                       {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "shuffle",
                                                           "typesize": np.dtype(dtype).itemsize, "blocksize": 0}}],
            "dimension_names": ["z", "y", "x"], "fill_value": 0,
        },
        "create": True, "delete_existing": True,
    }).result().write(arr).result()


def _halve_mean(a):
    """2x mean downsampling, edge-padded for odd sizes."""
    pad = [(0, s % 2) for s in a.shape]
    if any(p[1] for p in pad):
        a = np.pad(a, pad, mode="edge")
    s = a.shape
    return a.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5), dtype=np.float64).astype(a.dtype)


def write_ome_zarr(path, arr, voxel_um, labels=False, name=None):
    """
    3D array -> OME-Zarr 0.5 multiscale (z, y, x in micrometers). Images get a
    mean pyramid (translation shifted by half a voxel per level, the OME
    voxel-centre convention); labels get every 2nd voxel (never averaged).
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    dtype = "uint32" if labels else "float32"
    a = np.ascontiguousarray(arr.astype(np.uint32 if labels else np.float32))
    sc, tr, datasets, lv = [float(v) for v in voxel_um], [0.0, 0.0, 0.0], [], 0
    while True:
        _write_array(path / str(lv), a, dtype)
        datasets.append({"path": str(lv), "coordinateTransformations": [
            {"type": "scale", "scale": [round(v, 6) for v in sc]},
            {"type": "translation", "translation": [round(v, 6) for v in tr]}]})
        if max(a.shape) <= 64:
            break
        if labels:
            a = np.ascontiguousarray(a[::2, ::2, ::2])
        else:
            tr = [t + 0.5 * s for t, s in zip(tr, sc)]
            a = _halve_mean(a)
        sc, lv = [v * 2 for v in sc], lv + 1
    (path / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {
        "version": "0.5", "multiscales": [{
            "name": name or path.name.replace(".ome.zarr", ""),
            "axes": [{"name": ax, "type": "space", "unit": "micrometer"} for ax in "zyx"],
            "datasets": datasets, "type": "nearest" if labels else "mean"}]}}}, indent=1))


def write_segment_properties(zarr_path, ids, names):
    present = [int(i) for i in ids if i]
    info = {"@type": "neuroglancer_segment_properties", "inline": {
        "ids": [str(i) for i in present],
        "properties": [
            {"id": "label", "type": "label",
             "values": [names.get(str(i), (str(i), ""))[0] or str(i) for i in present]},
            {"id": "description", "type": "description",
             "values": [f"{names.get(str(i), ('', ''))[1] or names.get(str(i), (str(i), ''))[0]} (id {i})"
                        for i in present]}]}}
    d = Path(zarr_path) / "segment_properties"
    d.mkdir(exist_ok=True)
    (d / "info").write_text(json.dumps(info))


# --------------------------------------------------------------------------
# Left/right alignment
# --------------------------------------------------------------------------
def find_midline(ref, max_shift=8):
    """
    (axis, centre) of the atlas's mirror plane, found from the brain mask:
    mirroring index i -> centre - i along that axis maps the mask best onto
    itself. centre = n - 1 means a plain np.flip.
    """
    def search(mask, axes, shifts):
        best = None
        for ax in axes:
            n = mask.shape[ax]
            for shift in shifts:
                m = mirror(mask, ax, n - 1 + shift)
                union = np.logical_or(m, mask).sum()
                iou = np.logical_and(m, mask).sum() / union if union else 0.0
                if best is None or iou > best[0]:
                    best = (iou, ax, n - 1 + shift)
        return best

    mask = ref > 0
    # which axis: on every 2nd voxel; then the exact plane on that axis at full resolution
    _, ax, _ = search(np.ascontiguousarray(mask[::2, ::2, ::2]), range(3), range(-max_shift // 2, max_shift // 2 + 1))
    iou, ax, centre = search(mask, [ax], range(-max_shift, max_shift + 1))
    logger.info(f"mirror plane: axis {'zyx'[ax]} (#{ax}), index i <-> {centre} - i, brain-mask overlap {iou:.3f}")
    if iou < 0.8:
        logger.warning("the atlas is not clearly left/right symmetric; check the flips in Neuroglancer")
    return ax, centre


def mirror(a, ax, centre):
    """out[..., i, ...] = a[..., centre - i, ...] along axis ax (zeros where that falls outside)."""
    n = a.shape[ax]
    src = centre - np.arange(n)
    ok = (src >= 0) & (src < n)
    out = np.zeros_like(a)
    dst_idx = [slice(None)] * a.ndim
    src_idx = [slice(None)] * a.ndim
    dst_idx[ax] = np.nonzero(ok)[0]
    src_idx[ax] = src[ok]
    out[tuple(dst_idx)] = a[tuple(src_idx)]
    return out


def side_totals(a, ax, centre):
    """(signal on the low-index side, signal on the high-index side) of the mirror plane."""
    i = np.arange(a.shape[ax])
    shape = [1, 1, 1]
    shape[ax] = -1
    i = i.reshape(shape)
    a = np.nan_to_num(a)
    return float((a * (2 * i < centre)).sum()), float((a * (2 * i > centre)).sum())


# --------------------------------------------------------------------------
# Neuroglancer views
# --------------------------------------------------------------------------
def image_shader(color, rng):
    return ("#uicontrol invlerp normalized(range=[%g, %g])\n"
            "#uicontrol vec3 color color(default=\"%s\")\n"
            "void main() {\n  emitRGB(color * normalized());\n}\n" % (rng[0], rng[1], color))


DIFF_SHADER = ("#uicontrol float scale slider(min=0, max=%g, default=%g)\n"
               "void main() {\n  float v = clamp(getDataValue() / scale, -1.0, 1.0);\n"
               "  emitRGB(v > 0.0 ? vec3(v, 0.0, v) : vec3(0.0, -v, 0.0));\n}\n")


def views(cfg):
    """Neuroglancer states with data URLs relative to the server ({BASE} is filled in by index.html)."""
    def src(rel):
        return "zarr://{BASE}/" + rel

    base_layers = [{"type": "image", "name": "CCF template", "source": src("template.ome.zarr"),
                    "shader": image_shader("#ffffff", cfg["template_range"]), "opacity": 0.6}]
    regions = {"type": "segmentation", "name": "regions", "source":
               [src("regions.ome.zarr"), "precomputed://{BASE}/regions.ome.zarr/segment_properties"],
               "selectedAlpha": 0.08, "notSelectedAlpha": 0.0, "saturation": 0.6, "hideSegmentZero": True}
    common = {"dimensions": {a: [cfg["voxel_um"][i] * 1e-6, "m"] for i, a in enumerate("zyx")},
              "crossSectionScale": max(cfg["shape"]) / 380.0, "projectionScale": 1.6 * max(cfg["shape"]),
              "position": [s / 2 for s in cfg["shape"]], "layout": "4panel"}

    out = {}
    groups = cfg["groups"]
    layers = list(base_layers)
    for gi, (g, members) in enumerate(groups.items()):
        layers.append({"type": "image", "name": f"{g} mean (n={len(members)})", "source": src(f"groups/{g}_mean.ome.zarr"),
                       "shader": image_shader(GROUP_COLORS[gi % len(GROUP_COLORS)], cfg["range"]),
                       "blend": "additive"})
    if cfg.get("diff"):
        a, b = cfg["diff"]
        layers.append({"type": "image", "name": f"{a} - {b}", "source": src(f"groups/{a}_minus_{b}.ome.zarr"),
                       "shader": DIFF_SHADER % (cfg["range"][1] * 2, cfg["range"][1]), "visible": False})
    for s in cfg["samples"]:
        gi = next((i for i, (g, m) in enumerate(groups.items()) if s in m), None)
        color = GROUP_COLORS[gi % len(GROUP_COLORS)] if gi is not None else "#ffffff"
        layers.append({"type": "image", "name": s, "source": src(f"samples/{s}.ome.zarr"),
                       "shader": image_shader(color, cfg["range"]), "blend": "additive", "visible": False})
    layers.append(regions)
    out["overlay"] = {**common, "layers": layers,
                      "selectedLayer": {"layer": "regions", "visible": False}}

    # side by side: one panel per group (its mean + its brains), navigation linked
    if groups:
        layers2, panels = list(base_layers) + [regions], []
        for gi, (g, members) in enumerate(groups.items()):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            names = [f"{g} mean"]
            layers2.append({"type": "image", "name": f"{g} mean", "source": src(f"groups/{g}_mean.ome.zarr"),
                            "shader": image_shader(color, cfg["range"]), "blend": "additive"})
            for s in members:
                layers2.append({"type": "image", "name": f"{g}: {s}", "source": src(f"samples/{s}.ome.zarr"),
                                "shader": image_shader(color, cfg["range"]), "blend": "additive", "visible": False})
                names.append(f"{g}: {s}")
            panels.append({"type": "viewer", "layers": ["CCF template", "regions"] + names, "layout": "xy"})
        out["side_by_side"] = {**common, "layers": layers2,
                               "layout": {"type": "row", "children": panels}}
    # each brain in its own panel
    if cfg["samples"]:
        layers3 = list(base_layers) + [regions]
        rows, row = [], []
        for s in cfg["samples"]:
            gi = next((i for i, (g, m) in enumerate(groups.items()) if s in m), None)
            color = GROUP_COLORS[gi % len(GROUP_COLORS)] if gi is not None else "#ffffff"
            layers3.append({"type": "image", "name": s, "source": src(f"samples/{s}.ome.zarr"),
                            "shader": image_shader(color, cfg["range"]), "blend": "additive"})
            row.append({"type": "viewer", "layers": ["CCF template", "regions", s], "layout": "xy"})
            if len(row) == 4:
                rows.append({"type": "row", "children": row})
                row = []
        if row:
            rows.append({"type": "row", "children": row})
        out["each_brain"] = {**common, "layers": layers3,
                             "layout": rows[0] if len(rows) == 1 else {"type": "column", "children": rows}}
    return out


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Atlas viewer</title>
<style>body{font:15px system-ui,sans-serif;margin:2em;max-width:64em;background:#111;color:#ddd}
a{color:#7cf}li{margin:.4em 0}code{color:#fc9}table{border-collapse:collapse;margin:.5em 0}
td,th{padding:.25em .8em;border-bottom:1px solid #333;text-align:left}th{color:#aaa;font-weight:normal}
.note{color:#999;font-size:13px}</style></head><body>
<h2>Atlas-space comparison</h2>
<p id=info></p>
<ul id=views></ul>
<p>All layers share the 25&nbsp;um atlas grid. Hover over a voxel to read every layer's value and the region name.
Toggle layers by clicking their tab; change contrast in the layer's panel (double-click the tab).</p>
<h3>Left / right</h3>
<p id=hinfo class=note></p>
<p id=stale style="color:#fc9"></p>
<table id=brains><tr><th>brain</th><th>signal low / high side</th><th>mirrored at build</th><th>mirror now</th></tr></table>
<p class=note>"Mirror now" flips that brain's own layers in the links above (tick, then open a view).
Group means and the difference keep the build-time choice; to change them, rebuild with
<code>--flip NAME</code> or <code>--keep NAME</code>.</p>
<script>
const BASE = location.origin + location.pathname.replace(/\\/[^/]*$/, "");
let V = null;
const extra = new Set();

function mirrored(src, h) {
  // the same volume seen through a left/right mirror: i -> centre - i along the left/right axis
  const M = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
  M[h.axis_index][h.axis_index] = -1;
  M[h.axis_index][3] = h.centre;
  return {url: src, transform: {outputDimensions: h.dimensions, matrix: M}};
}

const sampleOf = src => { const m = typeof src === "string" && src.match(/\\/samples\\/([^/]+)\\.ome\\.zarr$/); return m && m[1]; };
const groupOf = src => { const m = typeof src === "string" && src.match(/\\/groups\\/(.+)_mean\\.ome\\.zarr$/); return m && m[1]; };

function groupMembers() {
  // group -> brains, read from the side-by-side view's "<group>: <brain>" layers
  const g = {};
  for (const l of (V.views.side_by_side || {layers: []}).layers) {
    const s = sampleOf(l.source), i = l.name.indexOf(": ");
    if (s && i > 0) (g[l.name.slice(0, i)] = g[l.name.slice(0, i)] || []).push(s);
  }
  return g;
}

function render() {
  const ul = document.getElementById("views");
  ul.innerHTML = "";
  // groups with a manually mirrored brain: their build-time mean is out of date
  const members = groupMembers();
  const stale = new Set(Object.keys(members).filter(g => members[g].some(s => extra.has(s))));
  const note = document.getElementById("stale");
  note.textContent = stale.size ? `Mean of ${[...stale].join(", ")} hidden (built without this mirroring); ` +
    "their brains are shown one by one instead. The difference map is hidden too." : "";
  for (const [name, st0] of Object.entries(V.views)) {
    const st = JSON.parse(JSON.stringify(st0).split("{BASE}").join(BASE));
    for (const l of st.layers) {
      const s = sampleOf(l.source), g = groupOf(l.source);
      if (s && extra.has(s)) l.source = mirrored(l.source, V.hemispheres);
      if (!stale.size) continue;
      if (g && stale.has(g)) l.visible = false;
      if (typeof l.source === "string" && l.source.includes("_minus_")) l.visible = false;
      if (s && [...stale].some(k => members[k].includes(s))) l.visible = true;
    }
    const li = document.createElement("li"), a = document.createElement("a");
    a.href = location.origin + "/ng/#!" + encodeURIComponent(JSON.stringify(st));
    a.textContent = V.titles[name] || name;
    li.appendChild(a); ul.appendChild(li);
  }
}

fetch("views.json").then(r => r.json()).then(v => {
  V = v;
  document.getElementById("info").textContent = v.description;
  const h = v.hemispheres;
  if (h) {
    document.getElementById("hinfo").textContent =
      `Left/right axis: ${h.axis} (mirror plane at index ${h.centre / 2}). ` +
      (h.target === "none" ? "No automatic mirroring." :
       `Brains were mirrored so that their stronger side is the ${h.target}-index side.`);
    const tb = document.getElementById("brains");
    for (const [b, s] of Object.entries(h.brains)) {
      const tr = document.createElement("tr");
      const fmt = x => x.toPrecision(3);
      tr.innerHTML = `<td>${b}</td><td>${fmt(s.low)} / ${fmt(s.high)} (x${s.ratio})</td>` +
                     `<td>${s.mirrored ? "yes" : "no"}</td><td><input type=checkbox></td>`;
      tr.querySelector("input").onchange = e => { e.target.checked ? extra.add(b) : extra.delete(b); render(); };
      tb.appendChild(tr);
    }
  }
  render();
});
</script></body></html>
"""


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="sample folders with atlas_space maps")
    ap.add_argument("--reference", required=True, help="the reference label tif used by warp_signal_to_atlas.py")
    ap.add_argument("--labels-info", help="CSV with region names for the reference labels")
    ap.add_argument("--registration", default="registration_ccf25_aind/Ex_488_Ch0_stitched")
    ap.add_argument("--atlas-space", default="atlas_space", help="folder name used by warp_signal_to_atlas.py")
    ap.add_argument("--map", default="seg-Ex_639_Ch2_stitched_fraction_ccf25.tif",
                    help="which map to show per brain (default: %(default)s)")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=TEXT",
                    help="a group = samples whose folder name contains TEXT (repeat; first two are compared)")
    ap.add_argument("--align-side", default="auto", choices=["auto", "low", "high", "none"],
                    help="mirror brains so their strong side is on this side (default: %(default)s)")
    ap.add_argument("--flip", action="append", default=[], metavar="TEXT",
                    help="always mirror brains whose name contains TEXT")
    ap.add_argument("--keep", action="append", default=[], metavar="TEXT",
                    help="never mirror brains whose name contains TEXT")
    ap.add_argument("--voxel-um", type=float, default=25.0)
    ap.add_argument("--out", required=True, help="viewer folder to create")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

    import ants
    import tifffile
    from quantify_registered import invert_orient, read_any, read_names

    out = Path(args.out)
    work = Path(tempfile.mkdtemp(prefix="atlasview_")) / out.name if str(out).startswith(("\\\\", "//")) else out
    (work / "samples").mkdir(parents=True, exist_ok=True)
    (work / "groups").mkdir(exist_ok=True)
    vox = [args.voxel_um] * 3

    # brains
    ref = read_any(args.reference, labels=True)
    lr_axis, centre = find_midline(ref)
    maps, sides, ref_orient, ccf_path = {}, {}, None, None
    for s in args.samples:
        sd = Path(s.rstrip("/\\"))
        a_dir = sd / args.registration / args.atlas_space
        f = a_dir / args.map
        if not f.is_file():
            logger.warning(f"{sd.name}: no {args.map} in {a_dir}, skipped")
            continue
        name = sd.name.replace("_Destripe_DONE", "")
        name = "_".join(name.split("_")[4:]) or name     # drop the acquisition date/time prefix
        maps[name] = tifffile.imread(f).astype(np.float32)
        # side decision on the sum-preserving map (count / sum) when there is one
        side_f = a_dir / args.map.replace("_fraction_", "_count_").replace("_mean_", "_sum_")
        lo, hi = side_totals(tifffile.imread(side_f) if side_f.is_file() and side_f != f else maps[name],
                             lr_axis, centre)
        sides[name] = {"low": lo, "high": hi, "strong": "high" if hi > lo else "low",
                       "ratio": round(max(lo, hi) / max(min(lo, hi), 1e-12), 2),
                       "decided_on": side_f.name if side_f.is_file() else f.name}
        if ref_orient is None:
            info = json.loads((a_dir / "atlas_space.json").read_text())
            ref_orient = info["reference_orientation_to_ccf"]
            ccf_path = json.loads((sd / args.registration / "run_config.json").read_text())["ccf_reference_path"]
        logger.info(f"{name}: {f.name} {maps[name].shape}")
    if not maps:
        raise SystemExit("no maps found; run warp_signal_to_atlas.py first")
    shape = next(iter(maps.values())).shape
    for n, m in maps.items():
        if m.shape != shape:
            raise SystemExit(f"{n}: map {m.shape} differs from {shape}")
    if ref.shape != shape:
        raise SystemExit(f"reference {ref.shape} differs from the maps {shape}")

    # left/right alignment
    if args.align_side == "auto":
        votes = [s["strong"] for s in sides.values()]
        target = "high" if votes.count("high") > votes.count("low") else "low"
    else:
        target = args.align_side
    flipped = {}
    for n in maps:
        forced = ("flip" if any(t.lower() in n.lower() for t in args.flip)
                  else "keep" if any(t.lower() in n.lower() for t in args.keep) else None)
        do = forced == "flip" or (forced is None and target != "none" and sides[n]["strong"] != target)
        flipped[n] = do
        why = f"forced ({forced})" if forced else "automatic" if target != "none" else "off"
        logger.info(f"{n}: signal low/high side {sides[n]['low']:.4g} / {sides[n]['high']:.4g} "
                    f"(x{sides[n]['ratio']}), mirrored: {'yes' if do else 'no'} [{why}]")
        if forced is None and target != "none" and sides[n]["ratio"] < 1.3:
            logger.warning(f"{n}: the two sides differ by less than 30%; check this brain's mirroring")
        if do:
            maps[n] = mirror(maps[n], lr_axis, centre)
    for n, m in maps.items():
        write_ome_zarr(work / "samples" / f"{n}.ome.zarr", m, vox, name=n)

    # groups
    groups = {}
    for g in args.group:
        gname, text = g.split("=", 1)
        members = [n for n in maps if text.lower() in n.lower()]
        groups[gname] = members
        if members:
            write_ome_zarr(work / "groups" / f"{gname}_mean.ome.zarr",
                           np.mean([maps[n] for n in members], axis=0), vox, name=f"{gname} mean")
        logger.info(f"group {gname} ({text}): {members}")
    diff = None
    full = [g for g, m in groups.items() if m]
    if len(full) >= 2:
        a, b = full[0], full[1]
        d = np.mean([maps[n] for n in groups[a]], axis=0) - np.mean([maps[n] for n in groups[b]], axis=0)
        write_ome_zarr(work / "groups" / f"{a}_minus_{b}.ome.zarr", d, vox, name=f"{a} - {b}")
        diff = (a, b)

    # background and regions, in the reference tif's own axis order
    tmpl = invert_orient(ants.image_read(ccf_path).numpy(), ref_orient["perm"], ref_orient["flips"])
    write_ome_zarr(work / "template.ome.zarr", np.ascontiguousarray(tmpl), vox, name="CCF template")
    write_ome_zarr(work / "regions.ome.zarr", ref, vox, labels=True, name="regions")
    write_segment_properties(work / "regions.ome.zarr", np.unique(ref), read_names(args.labels_info))

    allv = np.concatenate([m[m > 0].ravel() for m in maps.values()]) if any((m > 0).any() for m in maps.values()) else np.array([1.0])
    rng = [0.0, float(np.percentile(allv, 99.5))]
    tv = tmpl[tmpl > 0]
    cfg = {"samples": list(maps), "groups": {g: m for g, m in groups.items() if m}, "diff": diff,
           "range": rng, "template_range": [float(np.percentile(tv, 1)), float(np.percentile(tv, 99.5))],
           "voxel_um": vox, "shape": list(shape)}
    titles = {"overlay": "Overlay: group means (+ every brain, hidden), regions",
              "side_by_side": "Side by side: one panel per group, linked navigation",
              "each_brain": "Each brain in its own panel"}
    desc = (f"{len(maps)} brains, map {args.map}, groups: "
            + ", ".join(f"{g} (n={len(m)})" for g, m in cfg["groups"].items())
            + f"; colour range 0 to {rng[1]:.3g}")
    hemis = {"axis": "zyx"[lr_axis], "axis_index": lr_axis, "centre": centre, "target": target,
             "dimensions": {a: [vox[i] * 1e-6, "m"] for i, a in enumerate("zyx")},
             "brains": {n: {**sides[n], "mirrored": flipped[n]} for n in maps}}
    (work / "views.json").write_text(json.dumps({"description": desc, "titles": titles, "views": views(cfg),
                                                 "hemispheres": hemis}, indent=1))
    (work / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    if work != out:
        shutil.copytree(work, out, dirs_exist_ok=True)
        shutil.rmtree(work.parent, ignore_errors=True)
    logger.info(f"Viewer written to {out}. Serve it with: uv run python serve_atlas_viewer.py \"{out}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
