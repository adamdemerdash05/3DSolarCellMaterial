from __future__ import annotations

import time
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

# Matplotlib embedding
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# VTK for 3D rendering
import vtk

# ----- Optional dependencies & feature flags -----
try:
    import SimpleITK as sitk
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False

try:
    from skimage.morphology import skeletonize_3d as _sk_skeletonize_3d
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import tifffile as _tif
    _HAS_TIFFILE = True
except Exception:
    _HAS_TIFFILE = False

try:
    import imageio.v3 as _iio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    from scipy.ndimage import gaussian_filter as _gauss
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------------------------------------------------------
# Binarization helper
# -------------------------------------------------------------------------
def _binarize_for_skeleton(
    vol: np.ndarray,
    method: str = "otsu",
    threshold: int | None = None,
) -> np.ndarray:
    """
    Return a binary volume where foreground == 1.
    method: "otsu" | "manual".
    If manual, `threshold` must be provided (0..255 for uint8).
    """
    if vol.dtype != np.uint8:
        v = vol.astype(np.float32)
        v = (255 * (v - v.min()) / (v.ptp() + 1e-8)).astype(np.uint8)
    else:
        v = vol

    if method == "manual":
        if threshold is None:
            raise ValueError("Manual threshold selected but `threshold` is None.")
        return (v >= threshold).astype(np.uint8)

    # Otsu via SimpleITK if available
    if _HAS_SITK:
        img = sitk.GetImageFromArray(v)
        otsu = sitk.OtsuThreshold(img, 0, 1)  # out: {0,1}
        return sitk.GetArrayFromImage(otsu).astype(np.uint8)

    # Manual Otsu implementation (fallback)
    hist, _ = np.histogram(v, bins=256, range=(0, 256))
    p = hist.astype(np.float64) / (v.size + 1e-8)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    t = int(np.nanargmax(sigma_b2))
    return (v >= t).astype(np.uint8)


# -------------------------------------------------------------------------
# Skeletonization backends
# -------------------------------------------------------------------------
def _skeletonize_sitk(bin_vol: np.ndarray) -> np.ndarray:
    """Skeletonize with SimpleITK's BinaryThinning (3D-capable)."""
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK not available")
    img = sitk.GetImageFromArray(bin_vol.astype(np.uint8))
    skel = sitk.BinaryThinning(img)
    out = sitk.GetArrayFromImage(skel).astype(np.uint8)
    return out


def _skeletonize_skimage(bin_vol: np.ndarray) -> np.ndarray:
    """Fallback 3D skeletonization using scikit-image."""
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image not available")
    out = _sk_skeletonize_3d(bin_vol.astype(bool)).astype(np.uint8)
    return out


def compute_skeleton(bin_vol: np.ndarray, backend: str = "sitk") -> np.ndarray:
    """
    Compute a 3D skeleton from a binary volume using preferred backend.
    backend: "sitk" (SimpleITK) | "skimage" (fallback)
    Returns uint8 {0,1} volume.
    """
    backend = backend.lower()
    if backend == "sitk":
        try:
            return _skeletonize_sitk(bin_vol)
        except Exception:
            if _HAS_SKIMAGE:
                return _skeletonize_skimage(bin_vol)
            raise
    elif backend == "skimage":
        return _skeletonize_skimage(bin_vol)
    else:
        raise ValueError("backend must be 'sitk' or 'skimage'")


# -------------------------------------------------------------------------
# VTK helpers
# -------------------------------------------------------------------------
def _vtk_actor_from_points(
    points_xyz: np.ndarray,
    sphere_radius: float = 10.0,
    color=(1.0, 0.0, 0.0),
) -> vtk.vtkActor:
    """
    Create a VTK glyph actor (spheres) at each XYZ point.
    Used only for endpoints / branchpoints, not for main skeleton.
    """
    vtk_pts = vtk.vtkPoints()
    for x, y, z in points_xyz:
        vtk_pts.InsertNextPoint(float(x), float(y), float(z))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(float(sphere_radius))
    sphere.SetPhiResolution(8)
    sphere.SetThetaResolution(8)

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(poly)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetScaleModeToDataScalingOff()
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetScalarVisibility(False)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(0.9)
    return actor
def _vtk_tube_actor_from_skeleton_points(zyx: np.ndarray,
                                         tube_radius: float = 1.0,
                                         color=(0.2, 1.0, 0.2)):
    """
    Create a VTK tube-actor from skeleton voxel coordinates.
    This builds a line-set by connecting each voxel to its 26-neighbour
    voxel if that neighbour also exists in the skeleton.

    zyx : Nx3 array (Z,Y,X voxel coordinates)
    """

    if zyx.size == 0:
        return None

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Map each voxel -> VTK point ID
    point_ids = {}
    for i, (z, y, x) in enumerate(zyx):
        pid = pts.InsertNextPoint(float(x), float(y), float(z))
        point_ids[(z, y, x)] = pid

    # 26-connected neighbourhood offsets (excluding 0,0,0)
    neighbours = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    voxels_set = set(map(tuple, zyx.tolist()))

    # Build connections
    for (z, y, x) in voxels_set:
        pid1 = point_ids[(z, y, x)]
        for dz, dy, dx in neighbours:
            nb = (z + dz, y + dy, x + dx)
            if nb in voxels_set:
                pid2 = point_ids[nb]

                # Create a line connecting pid1 → pid2
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pid1)
                line.GetPointIds().SetId(1, pid2)
                lines.InsertNextCell(line)

    # PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)

    # Tube filter (gives a smooth branch-like appearance)
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetRadius(float(tube_radius))
    tube.SetNumberOfSides(12)
    tube.CappingOn()
    tube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor

def _downsample3d(vol: np.ndarray, factor: int) -> np.ndarray:
    """Simple stride-based 3D downsampling."""
    if factor <= 1:
        return vol
    return vol[::factor, ::factor, ::factor]


# -------------------------------------------------------------------------
# Line skeleton actor from voxel coordinates (no tubes)
# -------------------------------------------------------------------------
def _vtk_line_actor_from_skeleton_points(
    zyx: np.ndarray,
    line_width: float = 1.0,
    color=(0.2, 1.0, 0.2),
) -> tuple[vtk.vtkActor | None, int]:
    """
    Build a VTK actor that renders the skeleton as THIN LINES (no tubes).
    `zyx` is an (N, 3) array of integer voxel coordinates (Z, Y, X).

    Logic:
      - Treat each skeleton voxel as a node.
      - Connect nodes that are 26-connected neighbors (3x3x3 neighbourhood
        excluding self) to form a network of short line segments.
      - Render as polydata with lines and adjustable line width.

    Returns (actor, num_segments).
    """
    if zyx.size == 0:
        return None, 0

    # Map voxel coordinate -> VTK point id
    points = vtk.vtkPoints()
    point_ids: dict[tuple[int, int, int], int] = {}
    for (z, y, x) in zyx:
        key = (int(z), int(y), int(x))
        pid = points.InsertNextPoint(float(x), float(y), float(z))  # XYZ
        point_ids[key] = pid

    vox_set = set(point_ids.keys())

    # 26-connected neighbourhood (excluding self)
    neighbour_offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    lines = vtk.vtkCellArray()
    added_pairs: set[tuple[int, int]] = set()

    for (z, y, x) in vox_set:
        pid0 = point_ids[(z, y, x)]
        for dz, dy, dx in neighbour_offsets:
            nb = (z + dz, y + dy, x + dx)
            if nb in vox_set:
                pid1 = point_ids[nb]
                key = tuple(sorted((pid0, pid1)))
                if key in added_pairs:
                    continue
                added_pairs.add(key)

                # Create a 2-point line cell
                lines.InsertNextCell(2)
                lines.InsertCellPoint(pid0)
                lines.InsertCellPoint(pid1)

    n_segments = lines.GetNumberOfCells()
    if n_segments == 0:
        return None, 0

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarVisibility(False)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetLineWidth(float(line_width))
    return actor, n_segments


# -------------------------------------------------------------------------
# Endpoint & branchpoint detection (6-connected)
# -------------------------------------------------------------------------
def _find_endpoints_and_branchpoints(
    skel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect endpoints and branch-points in a 3D skeleton volume.
    skel: uint8 {0,1} array.
    Returns (endpoints_zyx, branchpoints_zyx) as integer arrays with shape (N, 3).
    """
    sk_bool = skel.astype(bool)
    coords = np.argwhere(sk_bool)
    if coords.size == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 3), dtype=int)

    voxels = set(map(tuple, coords.tolist()))

    neighbour_offsets = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    endpoints = []
    branchpoints = []

    for (z, y, x) in coords:
        degree = 0
        for dz, dy, dx in neighbour_offsets:
            if (int(z + dz), int(y + dy), int(x + dx)) in voxels:
                degree += 1
        if degree == 1:
            endpoints.append((int(z), int(y), int(x)))
        elif degree > 2:
            branchpoints.append((int(z), int(y), int(x)))

    endpoints = np.array(endpoints, dtype=int) if endpoints else np.empty((0, 3), dtype=int)
    branchpoints = np.array(branchpoints, dtype=int) if branchpoints else np.empty((0, 3), dtype=int)
    return endpoints, branchpoints

def _prune_skeleton_volume(skel: np.ndarray, min_component_size: int) -> np.ndarray:
    """
    Remove very small connected components from a skeleton volume.
    A 'component' is defined by 26-connectivity in Z,Y,X.
    """
    if min_component_size <= 1:
        return skel

    sk_bool = skel.astype(bool)
    coords = np.argwhere(sk_bool)
    if coords.size == 0:
        return skel

    voxels = set(map(tuple, coords.tolist()))

    # 26-connected neighbourhood (excluding self)
    neighbour_offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    visited = set()
    keep = []

    for start in voxels:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []

        while stack:
            z, y, x = stack.pop()
            component.append((z, y, x))
            for dz, dy, dx in neighbour_offsets:
                nb = (z + dz, y + dy, x + dx)
                if nb in voxels and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if len(component) >= min_component_size:
            keep.extend(component)

    if not keep:
        return np.zeros_like(skel, dtype=skel.dtype)

    keep_arr = np.array(keep, dtype=int)
    pruned = np.zeros_like(skel, dtype=skel.dtype)
    pruned[keep_arr[:, 0], keep_arr[:, 1], keep_arr[:, 2]] = np.array(1, dtype=skel.dtype)
    return pruned
# -------------------------------------------------------------------------
# GUI integration
# -------------------------------------------------------------------------
def add_skeleton_tab(
    root: tk.Tk,
    notebook: ttk.Notebook,
    renderer3d: vtk.vtkRenderer,
    get_active_volume,
    sphere_radius: float = 1,
):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Skeletonization")

    # --- State ---
    state = {
        "skel_actor": None,      # main skeleton actor (points or tubes)
        "end_actor": None,       # endpoints (red spheres)
        "branch_actor": None,    # branch points (blue spheres)
        "skel_vol": None,        # uint8 {0,1} (already pruned)
        "last_counts": None,     # per-Z counts used by the graph
        "skeleton_only": False,  # whether we are in "show only skeleton" mode
        "hidden_actors": [],     # actors we hid when entering skeleton-only mode
    }

    # --- Layout ---
    left = ttk.Frame(tab)
    right = ttk.Frame(tab)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

    # Controls
    ttk.Label(left, text="Binarization").pack(anchor=tk.W)
    method_var = tk.StringVar(value="otsu")
    cmb_method = ttk.Combobox(
        left,
        textvariable=method_var,
        values=["otsu", "manual"],
        state="readonly",
    )
    cmb_method.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Manual threshold (0–255)").pack(anchor=tk.W)
    thr_var = tk.IntVar(value=128)
    spn_thr = ttk.Spinbox(
        left,
        from_=0,
        to=255,
        increment=1,
        textvariable=thr_var,
    )
    spn_thr.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Skeleton backend").pack(anchor=tk.W, pady=(8, 0))
    backend_var = tk.StringVar(value="sitk" if _HAS_SITK else "skimage")
    cmb_backend = ttk.Combobox(
        left,
        textvariable=backend_var,
        values=["sitk", "skimage"],
        state="readonly",
    )
    cmb_backend.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Downsample factor").pack(anchor=tk.W, pady=(8, 0))
    ds_var = tk.IntVar(value=1)
    cmb_ds = ttk.Spinbox(
        left,
        from_=1,
        to=8,
        increment=1,
        textvariable=ds_var,
    )
    cmb_ds.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Gaussian sigma (pre-smooth)").pack(anchor=tk.W, pady=(8, 0))
    sigma_var = tk.DoubleVar(value=0.0)  # 0 = no smoothing
    spn_sigma = ttk.Spinbox(
        left,
        from_=0.0,
        to=5.0,
        increment=0.5,
        textvariable=sigma_var,
    )
    spn_sigma.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Max rendered points (cap)").pack(anchor=tk.W, pady=(8, 0))
    maxpts_var = tk.IntVar(value=200_000)
    spn_maxpts = ttk.Spinbox(
        left,
        from_=1_000,
        to=5_000_000,
        increment=10_000,
        textvariable=maxpts_var,
    )
    spn_maxpts.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Prune components smaller than (voxels)").pack(anchor=tk.W, pady=(8, 0))
    prune_var = tk.IntVar(value=0)
    spn_prune = ttk.Spinbox(
        left,
        from_=0,
        to=100_000,
        increment=10,
        textvariable=prune_var,
    )
    spn_prune.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Render mode").pack(anchor=tk.W, pady=(8, 0))
    render_mode_var = tk.StringVar(value="tubes")  # default to tubes / branches
    cmb_render = ttk.Combobox(
        left,
        textvariable=render_mode_var,
        values=["points", "tubes"],
        state="readonly",
    )
    cmb_render.pack(fill=tk.X, pady=2)

    ttk.Label(left, text="Node / tube radius").pack(anchor=tk.W)
    radius_var = tk.DoubleVar(value=float(sphere_radius))
    spn_radius = ttk.Spinbox(
        left,
        from_=0.1,
        to=10.0,
        increment=0.1,
        textvariable=radius_var,
    )
    spn_radius.pack(fill=tk.X, pady=2)

    btn_run = ttk.Button(left, text="Compute skeleton", command=lambda: _on_compute())
    btn_run.pack(fill=tk.X, pady=(10, 2))

    btn_toggle = ttk.Button(
        left,
        text="Toggle skeleton visibility",
        command=lambda: _on_toggle(),
    )
    btn_toggle.pack(fill=tk.X, pady=2)

    btn_skel_only = ttk.Button(
        left,
        text="Show only skeleton (hide volume)",
        command=lambda: _on_toggle_skeleton_only(),
    )
    btn_skel_only.pack(fill=tk.X, pady=2)

    btn_clear = ttk.Button(left, text="Clear skeleton", command=lambda: _on_clear())
    btn_clear.pack(fill=tk.X, pady=2)

    # Export buttons
    btn_exp_tiff = ttk.Button(
        left,
        text="Export TIFF stack (.tif)",
        command=lambda: _on_export_tiff(),
    )
    btn_exp_tiff.pack(fill=tk.X, pady=(12, 2))

    btn_exp_raw = ttk.Button(
        left,
        text="Export RAW volume (.raw)",
        command=lambda: _on_export_raw(),
    )
    btn_exp_raw.pack(fill=tk.X, pady=2)

    status_var = tk.StringVar(value="Idle")
    ttk.Label(left, textvariable=status_var, foreground="#2d6cdf").pack(
        anchor=tk.W, pady=(8, 0)
    )

    # Figure (graph): skeleton voxel count per Z
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Skeleton voxels per Z-slice")
    ax.set_xlabel("Z index")
    ax.set_ylabel("# skeleton voxels")

    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # --- Callbacks ---------------------------------------------------------
    def _on_compute():
        """Compute skeleton, apply pruning, and update VTK + graph."""
        t0 = time.time()

        vol = get_active_volume()
        if vol is None or vol.size == 0:
            status_var.set("No volume/ROI to process")
            return

        sigma = float(sigma_var.get())
        if sigma > 0 and _HAS_SCIPY:
            vol_proc = _gauss(vol.astype(np.float32), sigma=sigma)
        else:
            if sigma > 0 and not _HAS_SCIPY:
                status_var.set("scipy not installed: smoothing disabled")
            vol_proc = vol

        # Binarize
        m = method_var.get()
        t = thr_var.get()
        bin_t0 = time.time()
        bin_vol = _binarize_for_skeleton(
            vol_proc,
            method=m,
            threshold=(t if m == "manual" else None),
        )
        bin_t1 = time.time()

        fg = float(bin_vol.mean())  # since {0,1}, this is foreground ratio
        status_var.set(f"Binarized in {bin_t1 - bin_t0:.2f}s | FG={fg * 100:.2f}%")
        root.update_idletasks()

        if fg < 0.0005:  # Too little foreground; graph would be empty
            _on_clear()
            status_var.set(
                "Foreground ~0% after threshold. "
                "Try lower manual threshold or different ROI."
            )
            return

        # Safety: avoid hanging on huge volumes if user forgot to downsample
        ds = int(ds_var.get())
        max_voxels = 256 ** 3  # tune this if needed
        if bin_vol.size > max_voxels and ds <= 1:
            status_var.set(
                f"Volume too large for skeletonization ({bin_vol.size} voxels). "
                "Please increase downsample factor or use a smaller ROI."
            )
            return

        # Downsample (speed-up)
        if ds > 1:
            bin_vol = _downsample3d(bin_vol, ds)

        # Skeletonize
        backend = backend_var.get()
        sk_t0 = time.time()
        try:
            sk = compute_skeleton(bin_vol, backend=backend)
        except Exception as e:
            if backend != "skimage" and _HAS_SKIMAGE:
                sk = compute_skeleton(bin_vol, backend="skimage")
            else:
                status_var.set(f"Skeletonization failed: {e}")
                return
        sk_t1 = time.time()

        # Prune very small components (short spurious branches)
        min_comp = int(prune_var.get())
        if min_comp > 1:
            sk = _prune_skeleton_volume(sk, min_comp)

        state["skel_vol"] = sk

        # Update VTK actor(s)
        _update_skeleton_actor(sk, max_points=maxpts_var.get())

        # Update graph
        _update_graph(sk)

        total = int(sk.sum())
        status_var.set(
            "Done in "
            f"{time.time() - t0:.2f}s | bin: {bin_t1 - bin_t0:.2f}s | "
            f"skel: {sk_t1 - sk_t0:.2f}s | voxels: {total}"
        )

    def _on_toggle():
        """Toggle visibility of the skeleton actors only (volume untouched)."""
        any_actor = state["skel_actor"]
        if any_actor is None:
            return
        new_vis = 0 if any_actor.GetVisibility() else 1
        for key in ["skel_actor", "end_actor", "branch_actor"]:
            actor = state.get(key)
            if actor is not None:
                actor.SetVisibility(new_vis)
        renderer3d.GetRenderWindow().Render()

    def _on_toggle_skeleton_only():
        """
        Toggle 'skeleton-only' mode:
        - ON: hide all non-skeleton actors, show only branches/endpoints.
        - OFF: restore previous actors' visibility.
        """
        if state["skel_actor"] is None:
            return

        ren = renderer3d
        if not state["skeleton_only"]:
            # Enter skeleton-only mode
            hidden = []
            actors = ren.GetActors()
            actors.InitTraversal()
            for _ in range(actors.GetNumberOfItems()):
                actor = actors.GetNextActor()
                if actor is None:
                    continue
                if actor in (
                    state["skel_actor"],
                    state["end_actor"],
                    state["branch_actor"],
                ):
                    actor.SetVisibility(1)
                else:
                    if actor.GetVisibility():
                        actor.SetVisibility(0)
                        hidden.append(actor)
            state["hidden_actors"] = hidden
            state["skeleton_only"] = True
        else:
            # Leave skeleton-only mode
            for actor in state.get("hidden_actors", []):
                actor.SetVisibility(1)
            state["hidden_actors"] = []
            state["skeleton_only"] = False

        ren.GetRenderWindow().Render()

    def _on_clear():
        """Remove all skeleton actors and clear the graph."""
        for key in ["skel_actor", "end_actor", "branch_actor"]:
            actor = state.get(key)
            if actor is not None:
                renderer3d.RemoveActor(actor)
                state[key] = None

        renderer3d.GetRenderWindow().Render()

        state["skel_vol"] = None
        state["last_counts"] = None

        ax.clear()
        ax.set_title("Skeleton voxels per Z")
        ax.set_xlabel("Z index")
        ax.set_ylabel("Number of skeleton voxels")
        ax.grid(True, alpha=0.35)
        canvas.draw_idle()

    def _on_export_tiff():
        sk = state.get("skel_vol")
        if sk is None or sk.size == 0:
            status_var.set("Nothing to export: compute the skeleton first")
            return

        path = filedialog.asksaveasfilename(
            title="Save skeleton as TIFF stack",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            vol = sk.astype("uint8")  # Z,Y,X
            if _HAS_TIFFILE:
                _tif.imwrite(path, vol, imagej=True)
            elif _HAS_IMAGEIO:
                _iio.imwrite(path, vol)
            else:
                raise RuntimeError("Install tifffile or imageio to export TIFF")

            status_var.set(f"Saved TIFF: {path}")
        except Exception as e:
            status_var.set(f"Export TIFF failed: {e}")

    def _on_export_raw():
        sk = state.get("skel_vol")
        if sk is None or sk.size == 0:
            status_var.set("Nothing to export: compute the skeleton first")
            return

        path = filedialog.asksaveasfilename(
            title="Save skeleton as RAW",
            defaultextension=".raw",
            filetypes=[("RAW", "*.raw"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            vol = sk.astype("uint8")
            with open(path, "wb") as f:
                vol.tofile(f)

            # Write a tiny sidecar with shape & dtype for easy reload
            meta_path = path + ".meta.txt"
            with open(meta_path, "w", encoding="utf-8") as m:
                m.write(f"shape={tuple(vol.shape)}\n")
                m.write(f"dtype={vol.dtype}\n")

            status_var.set(f"Saved RAW: {path}")
        except Exception as e:
            status_var.set(f"Export RAW failed: {e}")

    def _update_graph(skel: np.ndarray):
        """Update the per-Z voxel count graph."""
        sk_bool = skel.astype(bool)
        counts = sk_bool.sum(axis=(1, 2))  # sum over Y,X for each Z
        state["last_counts"] = counts

        ax.clear()
        z = np.arange(counts.shape[0])
        ax.plot(z, counts, marker="o", markersize=2, linewidth=1)
        ax.set_title("Skeleton voxels per Z")
        ax.set_xlabel("Z index")
        ax.set_ylabel("Number skeleton voxels")
        ax.grid(True, alpha=0.35)
        canvas.draw_idle()

    def _update_skeleton_actor(skel: np.ndarray, max_points: int = 200_000):
        """
        Create / update the VTK actors for:
          - main skeleton (points or tubes)
          - endpoints (red spheres)
          - branch-points (blue spheres)
        """
        # Remove old actors (we'll recreate them)
        for key in ["skel_actor", "end_actor", "branch_actor"]:
            actor_old = state.get(key)
            if actor_old is not None:
                renderer3d.RemoveActor(actor_old)
                state[key] = None

        sk_bool = skel.astype(bool)
        zyx_all = np.argwhere(sk_bool)
        if zyx_all.size == 0:
            renderer3d.GetRenderWindow().Render()
            return

        # Limit points for performance (for both tubes & endpoints visualisation)
        n = zyx_all.shape[0]
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            zyx = zyx_all[idx]
        else:
            zyx = zyx_all

        render_mode = render_mode_var.get()
        radius = float(radius_var.get())

        # --- Main skeleton actor: points or tubes ---
        if render_mode == "tubes":
            actor_main = _vtk_tube_actor_from_skeleton_points(
                zyx,
                tube_radius=radius,
                color=(0.2, 1.0, 0.2),
            )
        else:  # points
            xyz = zyx[:, [2, 1, 0]].astype(np.float32)
            actor_main = _vtk_actor_from_points(
                xyz,
                sphere_radius=radius,
                color=(0.2, 1.0, 0.2),
            )

        if actor_main is not None:
            state["skel_actor"] = actor_main
            renderer3d.AddActor(actor_main)

        # --- Endpoints & branch points on the (pruned) skeleton ---
        endpoints_zyx, branch_zyx = _find_endpoints_and_branchpoints(skel)

        if endpoints_zyx.size > 0:
            endpoints_xyz = endpoints_zyx[:, [2, 1, 0]].astype(np.float32)
            end_actor = _vtk_actor_from_points(
                endpoints_xyz,
                sphere_radius=radius * 2.0,
                color=(1.0, 0.1, 0.1),  # red
            )
            state["end_actor"] = end_actor
            renderer3d.AddActor(end_actor)

        if branch_zyx.size > 0:
            branch_xyz = branch_zyx[:, [2, 1, 0]].astype(np.float32)
            br_actor = _vtk_actor_from_points(
                branch_xyz,
                sphere_radius=radius * 1.5,
                color=(0.1, 0.3, 1.0),  # blue
            )
            state["branch_actor"] = br_actor
            renderer3d.AddActor(br_actor)

        renderer3d.GetRenderWindow().Render()

    # Expose some internals for advanced users
    tab._state = state
    tab._update_graph = _update_graph
    tab._update_skeleton_actor = _update_skeleton_actor

    return tab
# End of add_skeleton_tab