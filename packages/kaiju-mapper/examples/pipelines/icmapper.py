"""
IC-Mapper
---------

IC-Mapper, introduced in Chalapathi, et al. [#icmapper]_, is an adaptive cover
scheme that optimizes the skeletonization of data obtained by embedding the
mapper graph via node centroids. The cover iteratively splits intervals when
doing so improves an information criterion (BIC or AIC) computed from the
clustering induced by nearest-node-centroid assignment from a partial
mapper complex.
"""

# %%
# Imports
# =======
# Imports to create the animation plots as well as apply mapper similar to the
# paper's implementation.

from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from kaiju_mapper import ICMapperCoverScheme, mapper
from kaiju_mapper.adapters import sk_learn, to_networkx
from kaiju_mapper.datasets import sphere
from kaiju_mapper.icmapper import ICInterval
from sklearn.cluster import DBSCAN
from zen_mapper import precomputed_cover

# %%
# Data
# ====
# We build a circles dataset similar to figure 4 from the paper.
# Interestingly, the source code for this is not available in their repo.

rng = np.random.default_rng(seed=0xDEADBEEF)
data_size = 2_000
data = np.vstack(
    (
        sphere(dim=1, radius=0.5, num_samples=int(2000 / 3), seed=rng),
        sphere(dim=1, radius=1, num_samples=int(2000 * 2 / 3), seed=rng),
    )
)
data += 0.04 * rng.normal(0, 1, data.shape)


plt.scatter(data[:, 0], data[:, 1], s=1)
plt.gca().axis("equal")
plt.show()

# %%
# Lens function
# =============
# We take the same lens function: (x,y) -> x + y

projection = data[:, 0] + data[:, 1]

plt.scatter(data[:, 0], data[:, 1], c=projection, s=1)
plt.gca().axis("equal")
plt.show()

# %%
# Mapper
# ============
# Define how we want IC-Mapper to create the Mapper graph hard-clustering.
# Note that we initialize the number of intervals to be 2 (similar to the original
# paper). Since the source code is not available, we do a split as if
# a single interval padded by `eps` meets the splitting criteria.
# We are also only using BIC and not AIC.

clusterer = sk_learn(DBSCAN(eps=0.1, min_samples=5))
lens_min = float(np.min(projection))
lens_max = float(np.max(projection))
lens_len = float(lens_max - lens_min)
eps = 1e-10
overlap = 0.1
interval_max = 20


# effectively split once
init_cover_template = [
    ICInterval(lens_min - eps, lens_min + lens_len / (2 - overlap)),
    ICInterval(lens_max - lens_len / (2 - overlap), lens_max + eps),
]

SCOPES = [("local", True), ("global", False)]
METHODS = ["best", "bfs", "dfs", "random"]
PANELS = [(s_name, s_local, m) for (s_name, s_local) in SCOPES for m in METHODS]

MAX_INTERVAL_RANGE = range(2, interval_max + 1)
RANDOM_SEED = 137  # for the "random" method, kept fixed so frames are reproducible


def compute_graph(max_intervals, use_local, method):
    """Build IC-Mapper for the given settings; return drawing data or None."""
    # fresh ICIntervals each call so members/state aren't shared between runs
    init_cover = [
        ICInterval(iv.lower_bound, iv.upper_bound) for iv in init_cover_template
    ]

    ic_cover_scheme = ICMapperCoverScheme(
        data=data,
        lens=projection,
        clusterer=clusterer,
        delta=0.0,
        iterations=50,
        max_intervals=max_intervals,
        init_cover=init_cover,
        use_local=use_local,
        use_aic=False,
        overlap=overlap,
        method=method,
        seed=RANDOM_SEED,
    )
    ic_membership = ic_cover_scheme()
    cover = precomputed_cover(ic_membership)

    result = mapper(
        data=data,
        projection=projection,
        cover_scheme=cover,
        clusterer=clusterer,
        dim=1,
    )

    graph = to_networkx(result.nerve)
    node_members = {n: result.nodes[n] for n in graph.nodes}
    pos = {n: data[members].mean(axis=0) for n, members in node_members.items()}
    colors = np.array(
        [
            (projection[node_members[n]].mean() - lens_min) / lens_len
            for n in graph.nodes
        ]
    )
    sizes = np.array([8 * np.sqrt(len(node_members[n])) for n in graph.nodes])
    edges = list(graph.edges)
    return pos, colors, sizes, edges


# %%
# Pre-Compute Panel Data
# ======================
# all_frames[i] is a
# :code:`dict: {(scope_name, method): (pos, colors, sizes, edges) or None}`


all_frames = []  # list of (max_intervals, dict_of_panel_data)

for max_intervals in MAX_INTERVAL_RANGE:
    panel_data = {}
    for scope_name, use_local, method in PANELS:
        try:
            panel_data[(scope_name, method)] = compute_graph(
                max_intervals, use_local, method
            )
        except Exception as e:
            print(
                f"[{scope_name}/{method}] max_intervals={max_intervals} failed: "
                f"{e}; skipping panel."
            )
            panel_data[(scope_name, method)] = None
    all_frames.append((max_intervals, panel_data))


# %%
# Animating IC-Mapper as ``max_intervals`` grows across all variants
# ==================================================================
# For each value of ``max_intervals`` from 1 to 20 we rebuild the IC-Mapper
# cover for every (scope, method) combination, run mapper, record the graph,
# and animate them side by side.


# 2x4 figure
n_rows = len(SCOPES)
n_cols = len(METHODS)
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True
)

# draw labels and scatter
for r, (scope_name, _use_local) in enumerate(SCOPES):
    for c, method in enumerate(METHODS):
        ax = axes[r, c]
        ax.scatter(data[:, 0], data[:, 1], c=projection, s=2, alpha=0.15, zorder=1)
        ax.set_aspect("equal")
        if r == 0:
            ax.set_title(method)
        if c == 0:
            ax.set_ylabel(scope_name, fontsize=12)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# panel cache
drawn_panels: dict[tuple[str, str], dict[str, Any]] = {
    (scope, method): {"nodes": None, "edges": None} for (scope, _, method) in PANELS
}


def _clear_panel(scope, method):
    cache = drawn_panels[(scope, method)]
    if cache["nodes"] is not None:
        cache["nodes"].remove()
        cache["nodes"] = None
    if cache["edges"] is not None:
        for coll in cache["edges"]:
            coll.remove()
        cache["edges"] = None


def _draw_panel(ax, scope, method, pos, colors, sizes, edges):
    if not pos:
        return
    g = nx.Graph()
    g.add_nodes_from(pos.keys())
    g.add_edges_from(edges)

    edge_coll = nx.draw_networkx_edges(
        g, pos=pos, ax=ax, edge_color="black", width=1.2, alpha=0.8
    )
    node_coll = nx.draw_networkx_nodes(
        g,
        pos=pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors="black",
        linewidths=0.6,
    )
    cache = drawn_panels[(scope, method)]
    cache["edges"] = edge_coll if isinstance(edge_coll, list) else [edge_coll]
    cache["nodes"] = node_coll


def update(frame):
    max_intervals, panel_data = frame
    for r, (scope_name, _use_local) in enumerate(SCOPES):
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            _clear_panel(scope_name, method)
            data_for_panel = panel_data.get((scope_name, method))
            if data_for_panel is None:
                continue
            pos, colors, sizes, edges = data_for_panel
            _draw_panel(ax, scope_name, method, pos, colors, sizes, edges)

    fig.suptitle(f"(embedded) IC-Mapper graphs @ iteration {max_intervals}")
    return ()


anim = animation.FuncAnimation(
    fig,
    update,
    frames=all_frames,
    interval=500,
    repeat=True,
    blit=False,  # could set to true to optimize size?
)

plt.show()

# %%
# References
# ==========
# .. [#gmapper] E. Alvarado, R. Belton, E. Fischer, K. J. Lee, S. Palande, S.
#        Percival, and E. Purvine,  "G-mapper: Learning a cover in the
#        mapper construction" 2025.
#
# .. [#icmapper] N. Chalapathi, Y. Zhou, and B. Wang, "Adaptive covers for
#        mapper graphs using information criteria" IEEE International
#        Conference on Big Data (IEEE BigData), 2021.
