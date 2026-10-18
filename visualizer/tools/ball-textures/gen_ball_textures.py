"""Generate logo-free equirectangular albedo + normal maps for the
Mikasa V200W and Molten V5M5000 (Flistatec) volleyballs.

Output: <out>/<ball>_albedo.png (WxH), <ball>_normal.png.
Pattern model: swirl bands in twisted-longitude space
(psi = phi + swirl*t + curl*t^3), micro-texture from a 3D trig lattice
(no polar pinch), seam grooves at panel edges. Panel edges are hard
(1px smoothing) and the map is rendered 2x supersampled then downscaled,
matching the crisp molded seams of the real balls. Normal map derived
from a height field by finite differences.
"""
import numpy as np
from PIL import Image
import json, sys, os

W, H = 2048, 1024
SS = 2  # supersample factor; edges stay crisp but antialiased
OUT = sys.argv[1] if len(sys.argv) > 1 else "textures"
PARAMS_FILE = sys.argv[2] if len(sys.argv) > 2 else "ball_params.json"

os.makedirs(OUT, exist_ok=True)
P = json.load(open(PARAMS_FILE))


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def grids():
    w, h = W * SS, H * SS
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    phi = (uu - 0.5) * 2 * np.pi          # longitude [-pi, pi]
    theta = vv * np.pi                    # colatitude [0, pi]
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = np.cos(theta)                     # pole axis
    z = st * np.sin(phi)
    t = np.cos(theta)                     # [-1(south) .. 1(north)]
    return phi, theta, t, x, y, z


def save(img_arr, path):
    im = Image.fromarray(img_arr)
    if SS > 1:
        im = im.resize((W, H), Image.LANCZOS)
    im.save(path)


def twisted_cell(phi, t, n, swirl, curl, phase):
    """Position in [0,1) within a band cell of twisted longitude."""
    psi = phi + swirl * t + curl * t ** 3 + phase
    return (psi * n / (2 * np.pi)) % 1.0


def normal_from_height(h, strength):
    gy, gx = np.gradient(h)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / l, ny / l, nz / l], axis=-1)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def lattice3d(x, y, z, freq):
    """Quasi-regular bump lattice on the sphere, no polar pinch."""
    return (np.cos(x * freq) * np.cos(y * freq) * np.cos(z * freq))


def cellular3d(x, y, z, freq):
    """Ridged lattice from skewed 3D directions ~ reads as cellular/hex
    embossing without axis-aligned ring artifacts on the sphere."""
    dirs = [
        (0.9, 0.31, 0.27),
        (0.21, 0.93, 0.30),
        (0.30, 0.24, 0.92),
        (-0.55, 0.62, 0.56),
    ]
    walls = None
    for dx, dy, dz in dirs:
        w = np.abs(np.sin((x * dx + y * dy + z * dz) * freq))
        walls = w if walls is None else np.minimum(walls, w)
    return walls


def gen_mikasa(p):
    phi, theta, t, x, y, z = grids()
    yellow = hex_to_rgb(p["yellow"])
    blue = hex_to_rgb(p["blue"])

    cell = twisted_cell(phi, t, p["bands"], p["swirl"], p.get("curl", 0.0), p.get("phase", 0.0))
    # blue band width varies with latitude: bands narrow toward the poles and
    # converge there, forming the pinwheel between yellow lobes (no soft cap).
    w = p["width_eq"] + (p["width_pole"] - p["width_eq"]) * np.abs(t) ** p["width_pow"]
    edge = p["edge_soft"]
    d = np.minimum(cell, 1.0 - cell)  # 0 at band center, 0.5 between
    blue_mask = 1.0 - smoothstep(w / 2 - edge, w / 2 + edge, d)

    col = yellow[None, None, :] * (1 - blue_mask[..., None]) + blue[None, None, :] * blue_mask[..., None]

    # dimple micro-texture (albedo modulation)
    dim = lattice3d(x, y, z, p["dimple_freq"])
    dimple_dark = smoothstep(0.3, 0.9, dim) * p["dimple_albedo"]
    col *= (1.0 - dimple_dark[..., None])

    # seam grooves at band edges
    seam_d = np.abs(d - w / 2)
    seam = 1.0 - smoothstep(0.0, p["seam_width"], seam_d)
    col *= (1.0 - seam[..., None] * p["seam_dark"])

    hgt = -smoothstep(0.3, 0.9, dim) * p["dimple_depth"] - seam * p["seam_depth"]
    nrm = normal_from_height(hgt, p["normal_strength"])

    save(np.clip(col, 0, 255).astype(np.uint8), f"{OUT}/mikasa_albedo.png")
    save(nrm, f"{OUT}/mikasa_normal.png")


def gen_molten(p):
    phi, theta, t, x, y, z = grids()
    white = hex_to_rgb(p["white"])
    red = hex_to_rgb(p["red"])
    blue = hex_to_rgb(p["blue"])

    # red+blue arms come as adjacent PAIRS sharing a seam, separated by white
    # channels; arms stay wide through the mid-latitudes, then taper fast to
    # curled points near the poles (curl term hooks the tips).
    cell = twisted_cell(phi, t, p["pairs"], p["swirl"], p.get("curl", 0.0), p.get("phase", 0.0))
    t_abs = np.abs(t)
    taper = 1.0 - smoothstep(p["flat_end"], p["tip_end"], t_abs)
    w = p["width_eq"] * taper
    edge = p["edge_soft"]
    d = np.abs(cell - 0.5)  # 0 at pair center
    color_mask = 1.0 - smoothstep(w / 2 - edge, w / 2 + edge, d)
    # red|blue seam sits red_frac of the way across the pair; the red arm
    # ends earlier toward the poles (staggered tips), the blue curls on
    rf = p["red_frac"] * (1.0 - smoothstep(p["flat_end"], p["red_tip_end"], t_abs))
    split = 0.5 - w / 2 + w * rf
    is_red = (cell < split).astype(np.float32)

    col = white[None, None, :] * np.ones_like(np.stack([t, t, t], axis=-1))
    swoosh = red[None, None, :] * is_red[..., None] + blue[None, None, :] * (1 - is_red[..., None])
    col = col * (1 - color_mask[..., None]) + swoosh * color_mask[..., None]

    # hex-ish embossing
    hexf = cellular3d(x, y, z, p["hex_freq"])
    ridge = 1.0 - smoothstep(0.0, p["hex_line"], hexf)
    col *= (1.0 - ridge[..., None] * p["hex_albedo"])

    # seams at arm edges and at the red/blue join inside each pair
    seam_d = np.abs(d - w / 2)
    join = (1.0 - smoothstep(0.0, p["seam_width"], np.abs(cell - split))) * color_mask
    seam = np.maximum((1.0 - smoothstep(0.0, p["seam_width"], seam_d)) * (w > 0.02), join)
    col *= (1.0 - seam[..., None] * p["seam_dark"])

    hgt = ridge * p["hex_height"] - seam * p["seam_depth"]
    nrm = normal_from_height(hgt, p["normal_strength"])

    save(np.clip(col, 0, 255).astype(np.uint8), f"{OUT}/molten_albedo.png")
    save(nrm, f"{OUT}/molten_normal.png")


gen_mikasa(P["mikasa"])
gen_molten(P["molten"])
print("wrote", OUT)
