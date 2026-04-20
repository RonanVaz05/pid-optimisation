"""
P&ID Network Optimization Using Matrix Methods
Course: Matrices and Linear Transformations, BCSAI 2026, IE University
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.csgraph import minimum_spanning_tree

# ============================================================
# 1. DEFINE THE PIPING NETWORK
# ============================================================

equipment = {
    0: "Feed Tank",
    1: "Pump P-101",
    2: "Heat Exchanger E-201",
    3: "Reactor R-301",
    4: "Cooler E-202",
    5: "Separator V-401",
    6: "Product Tank T-501",
    7: "Waste Treatment W-601",
    8: "Recycle Valve CV-101",
    9: "Recycle Mixer M-101",
}

pipes = [
    (0, 1, 2),
    (1, 9, 3),
    (9, 2, 4),
    (2, 3, 5),
    (3, 4, 4),
    (4, 5, 3),
    (5, 6, 3),
    (5, 7, 2),
    (5, 8, 2),
    (8, 9, 3),
    (3, 7, 6),
    (0, 9, 5),
    (4, 6, 7),
]

G = nx.DiGraph()
for node_id, name in equipment.items():
    G.add_node(node_id, label=name)
for u, v, w in pipes:
    G.add_edge(u, v, weight=w)

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} pipes")
print(f"Total pipe cost: {sum(w for _, _, w in pipes)}")

# ============================================================
# 2. ADJACENCY MATRIX
# ============================================================

A = nx.adjacency_matrix(G).todense()
print("\n--- ADJACENCY MATRIX ---")
print(np.array(A))

A2 = np.array(A) @ np.array(A)
print(f"\nA^2[0,9] = {A2[0,9]} (two-step paths: Feed Tank -> Mixer)")

# ============================================================
# 3. INCIDENCE MATRIX + RANK ANALYSIS
# ============================================================

B = nx.incidence_matrix(G, oriented=True).todense()
rank_B = np.linalg.matrix_rank(B)
n, m = G.number_of_nodes(), G.number_of_edges()

print(f"\n--- INCIDENCE MATRIX ---")
print(f"Shape: {B.shape} ({n} nodes x {m} pipes)")
print(f"Rank: {rank_B}")
print(f"Connected components: {n - rank_B}")
print(f"Redundant pipes: {m - rank_B}")
print(f"  (Minimum pipes for connectivity: {rank_B}, actual: {m})")

# ============================================================
# 4. GRAPH LAPLACIAN + EIGENVALUES
# ============================================================

G_undir = G.to_undirected()
L = nx.laplacian_matrix(G_undir).todense().astype(float)

eigenvalues, eigenvectors = eigh(np.array(L))

print(f"\n--- LAPLACIAN EIGENVALUES ---")
for i, ev in enumerate(eigenvalues):
    marker = ""
    if i == 0:
        marker = " (zero = connected)"
    elif i == 1:
        marker = " (Fiedler value = algebraic connectivity)"
    print(f"  lambda_{i} = {ev:.4f}{marker}")

# ============================================================
# 5. SPECTRAL PARTITIONING
# ============================================================

fiedler = np.array(eigenvectors[:, 1]).flatten()

print(f"\n--- SPECTRAL PARTITION (FIEDLER VECTOR) ---")
module_a, module_b = [], []
for i, val in enumerate(fiedler):
    module = "A" if val >= 0 else "B"
    (module_a if val >= 0 else module_b).append(i)
    print(f"  Node {i:2d} ({equipment[i]:25s}): {val:+.4f} -> Module {module}")

print(f"\nModule A: {[equipment[i] for i in module_a]}")
print(f"Module B: {[equipment[i] for i in module_b]}")

cut_edges = [(u, v) for u, v in G_undir.edges()
             if (u in module_a and v in module_b) or (u in module_b and v in module_a)]
print(f"\nInter-module connections (pipes to connect on-site): {len(cut_edges)}")
for u, v in cut_edges:
    print(f"  {equipment[u]} <-> {equipment[v]}")

# ============================================================
# 6. MINIMUM SPANNING TREE
# ============================================================

W = np.zeros((10, 10))
for u, v, w in pipes:
    W[u, v] = w
    W[v, u] = w

mst = minimum_spanning_tree(W).toarray()
mst_cost = mst[mst > 0].sum()
total_cost = sum(w for _, _, w in pipes)

print(f"\n--- MINIMUM SPANNING TREE ---")
print(f"Original network cost: {total_cost}")
print(f"MST cost:              {mst_cost:.0f}")
print(f"Savings:               {total_cost - mst_cost:.0f} ({(total_cost - mst_cost) / total_cost * 100:.1f}%)")

# ============================================================
# 7. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
pos = nx.spring_layout(G_undir, seed=42)
labels = {i: f"{i}" for i in range(10)}

# Original
ax1 = axes[0]
ax1.set_title("Original Piping Network", fontweight="bold", fontsize=13)
nx.draw(G, pos, ax=ax1, with_labels=False, node_color="#2962FF",
        node_size=500, edge_color="#888", arrows=True, arrowsize=15, width=1.5)
nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=9, font_color="white", font_weight="bold")

# Spectral partition
ax2 = axes[1]
ax2.set_title("Spectral Partition (Fiedler Vector)", fontweight="bold", fontsize=13)
colors = ["#2962FF" if i in module_a else "#FF6D00" for i in range(10)]
nx.draw(G_undir, pos, ax=ax2, with_labels=False, node_color=colors,
        node_size=500, edge_color="#888", width=1.5)
nx.draw_networkx_labels(G_undir, pos, labels, ax=ax2, font_size=9, font_color="white", font_weight="bold")
nx.draw_networkx_edges(G_undir, pos, edgelist=cut_edges, ax=ax2,
                       edge_color="red", width=3, style="dashed")

# MST
ax3 = axes[2]
ax3.set_title("Minimum Spanning Tree", fontweight="bold", fontsize=13)
mst_edges = [(i, j) for i in range(10) for j in range(10) if mst[i, j] > 0]
G_mst = nx.Graph()
G_mst.add_nodes_from(range(10))
G_mst.add_edges_from(mst_edges)
nx.draw(G_mst, pos, ax=ax3, with_labels=False, node_color="#00B878",
        node_size=500, edge_color="#00B878", width=2.5)
nx.draw_networkx_labels(G_mst, pos, labels, ax=ax3, font_size=9, font_color="white", font_weight="bold")

for ax in axes:
    ax.set_facecolor("#FAFAFA")

plt.tight_layout()
out_path = "/Users/ronanvaz/programming/linear-algebra-project/pid_network_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nVisualization saved to {out_path}")
plt.close()
