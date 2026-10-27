import sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from map_utils import load_map
from global_planner import GlobalPlanner

m = load_map(os.path.join(ROOT, "maps", "demo.yaml"))
t = time.time(); gp = GlobalPlanner(m); print(f"init {time.time()-t:.2f}s  grid {m.occ.shape}")

# brute-force reference clearance: min distance from every path point to any occupied cell
rr, cc = np.nonzero(m.occ)
occ_xy = np.stack([m.origin[0]+(cc+.5)*m.resolution, m.origin[1]+(rr+.5)*m.resolution], 1)
def clearance(path):
    return min(np.min(np.hypot(*(occ_xy - p).T)) for p in path[::3])

cases = [((0,0),(5,0)), ((0,0),(5,3)), ((0,0),(-2,3.5)), ((0,0),(-4,-4)), ((-4,-4),(4,4.5)), ((0,0),(2,4.5)), ((0,0),(-1.5,2.6))]
for s, g in cases:
    gp.set_goal(g); t = time.time(); ok = gp.update(s); dt = time.time()-t
    if ok:
        L = gp.s[-1]; straight = np.hypot(g[0]-s[0], g[1]-s[1])
        print(f"{s}->{g}: ok len={L:.2f} (straight {straight:.2f}) clearance={clearance(gp.path):.2f}m {dt*1000:.0f}ms")
    else: print(f"{s}->{g}: FAILED {gp.status}")


gp.set_goal((-6.2,0)); print("outside map ->", gp.update((0,0)), gp.status)

# circle plugging the wall gap: must detour around the LEFT end of the wall at x=0.5
gp.set_goal((4.5,4.5)); gp.update((0,0)); L0=gp.s[-1]; v0=gp.version
circ = np.array([[4.6, 3.0, 0.9]])
ok = gp.update((0,0), circ)
d = np.hypot(gp.path[:,0]-4.6, gp.path[:,1]-3.0) - 0.9
print(f"gap blocked: ok={ok} len {L0:.2f}->{gp.s[-1]:.2f} version {v0}->{gp.version} min clearance to circle {d.min():.2f}")
# fully sealed goal
gp.set_goal((4.5,4.5)); sealed=np.array([[4.5,4.5,0.2],[4.5,4.5,0.2]])
print("goal under circle: ok=", gp.update((0,0), np.array([[4.5,4.5,0.5]])), "effective goal", np.round(gp.effective_goal,2), "(true goal 4.5,4.5)")
# irrelevant circle
gp.set_goal((5,0)); gp.update((0,0),np.empty((0,3))); v=gp.version; gp.update((0,0), np.array([[-4,-4,0.3]])); print("irrelevant circle replans:", gp.version!=v)
# robot drifts off path -> replan
gp.update((0,0.0)); gp.update((0,-1.5)); print("off-path replans:", gp.version!=v, gp.status)
