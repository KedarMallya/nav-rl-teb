"""Occupancy-grid map (standard ROS map_server YAML + PGM/PNG) -> sim / planner / RViz.

One map feeds three consumers, so they can never disagree:
  * RViz      : nav_msgs/OccupancyGrid on /map   (built in ros_node.py from MapData)
  * MuJoCo    : exact box geoms, one per rectangle (grid_to_rects)
  * TEB solver: ellipses circumscribing each rectangle (rects_to_ellipses),
                rows [x, y, a, b, theta, cos, sin]

Map origin yaw is ignored (assumed 0).
Run `python3 map_utils.py [path/to/demo.yaml]` to (re)generate the demo map.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image


@dataclass
class MapData:
    occ: np.ndarray    # (H, W) bool. Row 0 = bottom (y = origin_y), col 0 = left (x = origin_x)
    resolution: float  # metres per cell
    origin: tuple      # (x, y, yaw) of the lower-left corner of cell (0, 0)

    @property
    def height(self):
        return self.occ.shape[0]

    @property
    def width(self):
        return self.occ.shape[1]

    @property
    def extent(self):
        return self.width * self.resolution, self.height * self.resolution

    @property
    def center(self):
        w, h = self.extent
        return self.origin[0] + 0.5 * w, self.origin[1] + 0.5 * h


def load_map(yaml_path):
    """Load a map_server style YAML (image, resolution, origin, negate, occupied_thresh)."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
    img = np.asarray(Image.open(img_path).convert("L"), dtype=float)
    p = img / 255.0 if meta.get("negate", 0) else (255.0 - img) / 255.0
    occ = p > float(meta.get("occupied_thresh", 0.65))  # unknown/grey counts as free
    occ = occ[::-1].copy()                               # image row 0 is the top; flip so row 0 = bottom
    ox, oy, oyaw = (list(meta.get("origin", [0.0, 0.0, 0.0])) + [0.0, 0.0, 0.0])[:3]
    return MapData(occ, float(meta["resolution"]), (float(ox), float(oy), float(oyaw)))


def grid_to_rects(m, max_len=1.0):
    """Greedy split of occupied cells into rectangles of at most max_len metres per side.

    Returns (k, 4) array of [cx, cy, hx, hy] (centre and half sizes) in the map frame.
    """
    res = m.resolution
    cap = max(1, int(round(max_len / res)))
    rem = m.occ.copy()
    H, W = rem.shape
    rects = []
    for r in range(H):
        c = 0
        while c < W:
            if not rem[r, c]:
                c += 1
                continue
            w = 1
            while c + w < W and w < cap and rem[r, c + w]:
                w += 1
            h = 1
            while r + h < H and h < cap and rem[r + h, c:c + w].all():
                h += 1
            rem[r:r + h, c:c + w] = False
            x0, y0 = m.origin[0] + c * res, m.origin[1] + r * res
            x1, y1 = x0 + w * res, y0 + h * res
            rects.append([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2])
            c += w
    return np.array(rects, float).reshape(-1, 4)


def rects_to_ellipses(rects, scale=np.sqrt(2.0)):
    """Axis-aligned ellipse through the 4 corners of each rectangle -> (m, 7) solver rows."""
    rows = [[cx, cy, hx * scale, hy * scale, 0.0, 1.0, 0.0] for cx, cy, hx, hy in rects]
    return np.array(rows, float).reshape(-1, 7)


def distance_to_rects(xy, rects):
    """Distance from point xy to the nearest rectangle (0 if inside one)."""
    if len(rects) == 0:
        return float("inf")
    dx = np.maximum(np.abs(xy[0] - rects[:, 0]) - rects[:, 2], 0.0)
    dy = np.maximum(np.abs(xy[1] - rects[:, 1]) - rects[:, 3], 0.0)
    return float(np.min(np.hypot(dx, dy)))


def write_demo_map(yaml_path, size=12.0, res=0.05):
    """12 x 12 m arena centred on (0, 0): border, a half wall, a wall with a gap, 4 pillars."""
    n = int(round(size / res))
    o = -size / 2.0
    img = np.full((n, n), 254, np.uint8)  # 254 = free, 0 = occupied (map_server convention)

    def fill(x0, y0, x1, y1):  # metres, map frame
        c0, c1 = int(round((x0 - o) / res)), int(round((x1 - o) / res))
        r0, r1 = int(round(n - (y1 - o) / res)), int(round(n - (y0 - o) / res))
        img[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = 0

    t, h = 0.2, size / 2.0
    fill(-h, -h, h, -h + t)       # border
    fill(-h, h - t, h, h)
    fill(-h, -h, -h + t, h)
    fill(h - t, -h, h, h)
    fill(-2.6, -h, -2.4, -1.0)    # vertical half wall (open at the top)
    fill(0.5, 2.9, 3.5, 3.1)      # horizontal wall (open on the right)
    fill(1.7, 0.0, 2.3, 0.6)      # pillars
    fill(3.0, -1.5, 4.0, -0.9)
    fill(3.8, 1.2, 4.4, 1.8)
    fill(-2.0, 1.6, -1.0, 2.0)

    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    pgm = os.path.splitext(yaml_path)[0] + ".pgm"
    Image.fromarray(img).save(pgm)
    with open(yaml_path, "w") as f:
        f.write(f"image: {os.path.basename(pgm)}\nmode: trinary\nresolution: {res}\n"
                f"origin: [{o}, {o}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return yaml_path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "demo.yaml")
    write_demo_map(target)
    m = load_map(target)
    rects = grid_to_rects(m)
    print(f"wrote {target}: {m.width}x{m.height} cells, {len(rects)} obstacle rectangles")
