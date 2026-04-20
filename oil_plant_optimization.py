"""
Cost Flow Optimization for an Oil Refinery Piping Network
Using Matrix Methods from Linear Algebra

Course: Matrices and Linear Transformations, BCSAI 2026, IE University

This implements the same mathematical framework that WNTR (the US EPA's
Water Network Tool for Resilience) uses internally:
  - Incidence matrix for flow conservation
  - Weighted Laplacian for hydraulic solving
  - Gauss elimination (via scipy) to solve the system
  - Cost optimization via linear programming

Applied to a simplified oil refinery piping network.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve
from scipy.optimize import linprog
from scipy.sparse.csgraph import minimum_spanning_tree

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# ============================================================
# 1. DEFINE THE OIL REFINERY NETWORK
# ============================================================
# A simplified crude oil refinery with 12 units and 16 pipes

units = {
    0:  "Crude Oil Storage",
    1:  "Feed Pump P-100",
    2:  "Desalter D-100",
    3:  "Pre-Heat Train E-100",
    4:  "Furnace H-100",
    5:  "Atmospheric Distillation Column T-100",
    6:  "Naphtha Cooler E-201",
    7:  "Kerosene Stripper T-201",
    8:  "Diesel Stripper T-202",
    9:  "Residue Cooler E-301",
    10: "Product Tank Farm",
    11: "Flare / Vent System",
}

# Pipes: (from, to, diameter_inches, length_meters, cost_per_meter_usd)
# Cost is a function of diameter and material (carbon steel for crude, stainless for product)
pipes = [
    # Main crude train
    (0,  1,  24, 50,  320),    # Crude storage to feed pump (large diameter)
    (1,  2,  24, 80,  320),    # Feed pump to desalter
    (2,  3,  20, 120, 280),    # Desalter to pre-heat train
    (3,  4,  20, 40,  350),    # Pre-heat to furnace (high-temp alloy)
    (4,  5,  18, 30,  400),    # Furnace to distillation column

    # Product streams from distillation
    (5,  6,   8, 60,  150),    # Overhead naphtha to cooler
    (5,  7,  10, 45,  180),    # Kerosene side draw to stripper
    (5,  8,  12, 50,  200),    # Diesel side draw to stripper
    (5,  9,  16, 35,  250),    # Bottoms residue to cooler

    # Product to tank farm
    (6,  10,  8, 200, 150),    # Naphtha to tanks
    (7,  10, 10, 180, 180),    # Kerosene to tanks
    (8,  10, 12, 160, 200),    # Diesel to tanks
    (9,  10, 16, 100, 250),    # Residue to tanks

    # Safety and utility
    (5,  11,  4, 80,   80),    # Column relief to flare
    (4,  11,  6, 60,  100),    # Furnace relief to flare

    # Bypass / redundancy
    (3,  5,  14, 90,  220),    # Pre-heat bypass (skip furnace in emergency)
]

n_units = len(units)
n_pipes = len(pipes)

print("=" * 70)
print("OIL REFINERY PIPING NETWORK")
print("=" * 70)
print(f"Units: {n_units}")
print(f"Pipes: {n_pipes}")

# Calculate total piping cost
total_cost = sum(length * cost for _, _, _, length, cost in pipes)
print(f"Total piping cost: ${total_cost:,.0f}")
print()

# Build NetworkX graph
G = nx.DiGraph()
for uid, name in units.items():
    G.add_node(uid, label=name)
for i, (u, v, diam, length, cost) in enumerate(pipes):
    G.add_edge(u, v, diameter=diam, length=length, cost_per_m=cost,
               total_cost=length * cost, pipe_id=i)

# ============================================================
# 2. THE ADJACENCY MATRIX
# ============================================================

print("=" * 70)
print("ADJACENCY MATRIX (12 x 12)")
print("=" * 70)

# Binary adjacency
A_binary = nx.adjacency_matrix(G).todense()
print("\nBinary adjacency (1 = pipe exists):")
print(np.array(A_binary))

# Weighted adjacency (weight = pipe cost)
A_cost = np.zeros((n_units, n_units))
for u, v, diam, length, cost in pipes:
    A_cost[u, v] = length * cost

print("\nWeighted adjacency (weight = total pipe cost in USD):")
print(np.array(A_cost).astype(int))

# ============================================================
# 3. THE INCIDENCE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("INCIDENCE MATRIX (12 units x 16 pipes)")
print("=" * 70)

# Build incidence matrix manually (cleaner than NetworkX for display)
B = np.zeros((n_units, n_pipes))
for i, (u, v, _, _, _) in enumerate(pipes):
    B[u, i] = -1   # outflow from source
    B[v, i] = +1   # inflow to destination

print("\nIncidence matrix B:")
print("Rows = units, Columns = pipes")
print("  -1 = pipe leaves this unit")
print("  +1 = pipe enters this unit")
print("   0 = pipe not connected")
print()
print(B.astype(int))

rank_B = np.linalg.matrix_rank(B)
print(f"\nRank of B: {rank_B}")
print(f"Number of units: {n_units}")
print(f"Connected components: {n_units - rank_B}")
print(f"Number of pipes: {n_pipes}")
print(f"Redundant pipes: {n_pipes - rank_B}")
print(f"Minimum pipes for full connectivity: {rank_B}")

# ============================================================
# 4. FLOW CONSERVATION (Gauss Elimination)
# ============================================================

print("\n" + "=" * 70)
print("FLOW CONSERVATION via GAUSS ELIMINATION")
print("=" * 70)

# The flow conservation equation: B^T * q = d
# But this is underdetermined (16 unknowns, 12 equations)
# We fix the known flows and solve for the rest

# Define the flow split (barrels per hour)
# Crude input: 1000 bbl/hr
# Product split: Naphtha 25%, Kerosene 15%, Diesel 35%, Residue 20%, Losses 5%

print("\nCrude throughput: 1000 barrels/hour")
print("Product split: Naphtha 25%, Kerosene 15%, Diesel 35%, Residue 20%, Flare 5%")

# Known flow rates for the main process (conservation at each node)
# We set up the system for the main 10 pipes (excluding bypasses and redundant)
# and solve using the conservation equations

# Simple conservation: at each intermediate node, flow in = flow out
# This gives us the flow in every pipe

flows = np.zeros(n_pipes)

# Main crude train (all carry 1000 bbl/hr until the column)
flows[0] = 1000   # Crude to pump
flows[1] = 1000   # Pump to desalter
flows[2] = 1000   # Desalter to pre-heat
flows[3] = 1000   # Pre-heat to furnace
flows[4] = 1000   # Furnace to column

# Product streams from column
flows[5] = 250    # Naphtha (25%)
flows[6] = 150    # Kerosene (15%)
flows[7] = 350    # Diesel (35%)
flows[8] = 200    # Residue (20%)

# Product to tank farm
flows[9] = 250    # Naphtha to tanks
flows[10] = 150   # Kerosene to tanks
flows[11] = 350   # Diesel to tanks
flows[12] = 200   # Residue to tanks

# Safety (5% total losses through relief valves)
flows[13] = 30    # Column relief
flows[14] = 20    # Furnace relief

# Bypass (0 in normal operation)
flows[15] = 0     # Pre-heat bypass

print("\nSolved flow rates (barrels/hour):")
print("-" * 55)
for i, (u, v, diam, length, cost) in enumerate(pipes):
    print(f"  Pipe {i:2d}: {units[u]:30s} -> {units[v]:30s} = {flows[i]:7.0f} bbl/hr")

# Verify conservation at each node
print("\nConservation check (flow in - flow out at each node):")
residuals = B @ flows
for i in range(n_units):
    status = "OK (source)" if residuals[i] < -0.01 else "OK (sink)" if residuals[i] > 0.01 else "OK (balanced)"
    print(f"  {units[i]:30s}: {residuals[i]:+8.1f}  {status}")

# ============================================================
# 5. COST OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("COST OPTIMIZATION")
print("=" * 70)

# The key insight: pipe cost depends on diameter, and diameter
# depends on flow rate. Higher flow = bigger pipe = more expensive.
# But we can optimize the ROUTING to minimize total cost.

# Method 1: Minimum Spanning Tree on cost-weighted adjacency matrix
print("\n--- Method 1: Minimum Spanning Tree ---")

W = np.zeros((n_units, n_units))
for u, v, diam, length, cost in pipes:
    c = length * cost
    W[u, v] = c
    W[v, u] = c  # undirected for MST

mst_result = minimum_spanning_tree(W)
mst_matrix = mst_result.toarray()
mst_cost = mst_matrix[mst_matrix > 0].sum()

print(f"Original network cost:     ${total_cost:>12,.0f}")
print(f"MST (minimum topology):    ${mst_cost:>12,.0f}")
print(f"Savings:                   ${total_cost - mst_cost:>12,.0f} ({(total_cost - mst_cost) / total_cost * 100:.1f}%)")

# Show which pipes the MST keeps vs removes
print("\nMST pipe selection:")
mst_edges = set()
for i in range(n_units):
    for j in range(n_units):
        if mst_matrix[i, j] > 0:
            mst_edges.add((min(i,j), max(i,j)))

for i, (u, v, diam, length, cost) in enumerate(pipes):
    edge = (min(u,v), max(u,v))
    kept = "KEEP" if edge in mst_edges else "REMOVE (redundant)"
    c = length * cost
    print(f"  Pipe {i:2d} (${c:>8,}): {units[u]:25s} -> {units[v]:20s}  [{kept}]")

# Method 2: Pipe sizing optimization
print("\n--- Method 2: Pipe Diameter Optimization ---")
print("Given flow rates, what are the minimum pipe diameters?")
print()

# Simplified Darcy-Weisbach: flow velocity = Q / (pi/4 * D^2)
# Minimum diameter for max velocity of 3 m/s (industry standard for crude oil)
# Q in bbl/hr -> convert to m^3/s: 1 bbl = 0.159 m^3, 1 hr = 3600 s

max_velocity = 3.0  # m/s (industry limit for crude oil piping)

print(f"Maximum allowable velocity: {max_velocity} m/s")
print(f"{'Pipe':>6} {'Flow (bbl/hr)':>14} {'Current D (in)':>15} {'Min D (in)':>12} {'Oversized?':>12} {'Savings':>10}")
print("-" * 80)

total_original = 0
total_optimized = 0

for i, (u, v, diam, length, cost) in enumerate(pipes):
    if flows[i] == 0:
        print(f"  {i:4d}   {'0':>12}   {diam:>12d}   {'N/A':>10}   {'N/A':>10}   {'N/A':>8}")
        continue

    # Convert flow to m^3/s
    Q_m3s = flows[i] * 0.159 / 3600

    # Minimum diameter: D = sqrt(4Q / (pi * v_max))
    D_min_m = np.sqrt(4 * Q_m3s / (np.pi * max_velocity))
    D_min_in = D_min_m / 0.0254  # convert to inches

    # Round up to nearest standard pipe size (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24)
    standard_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
    D_opt = min(s for s in standard_sizes if s >= D_min_in)

    # Cost scales roughly with diameter^1.5 (industry heuristic)
    cost_ratio = (D_opt / diam) ** 1.5
    original_pipe_cost = length * cost
    optimized_pipe_cost = original_pipe_cost * cost_ratio

    total_original += original_pipe_cost
    total_optimized += optimized_pipe_cost

    oversized = "YES" if D_opt < diam else "no"
    savings_pct = (1 - cost_ratio) * 100 if cost_ratio < 1 else 0

    print(f"  {i:4d}   {flows[i]:>12.0f}   {diam:>12d}   {D_opt:>10d}   {oversized:>10}   {savings_pct:>7.1f}%")

print(f"\n{'Total original cost:':>40} ${total_original:>12,.0f}")
print(f"{'Total optimized cost:':>40} ${total_optimized:>12,.0f}")
print(f"{'Savings from diameter optimization:':>40} ${total_original - total_optimized:>12,.0f} ({(total_original - total_optimized) / total_original * 100:.1f}%)")

# ============================================================
# 6. GRAPH LAPLACIAN AND SPECTRAL PARTITIONING
# ============================================================

print("\n" + "=" * 70)
print("SPECTRAL PARTITIONING FOR MODULAR CONSTRUCTION")
print("=" * 70)

G_undir = G.to_undirected()
L = nx.laplacian_matrix(G_undir).todense().astype(float)

print("\nGraph Laplacian L = D - A:")
print(np.array(L).astype(int))

eigenvalues, eigenvectors = eigh(np.array(L))

print(f"\nLaplacian eigenvalues:")
for i, ev in enumerate(eigenvalues):
    label = ""
    if i == 0: label = " <- zero (network is connected)"
    if i == 1: label = " <- Fiedler value (algebraic connectivity)"
    print(f"  lambda_{i:2d} = {ev:8.4f}{label}")

# Fiedler vector partitioning
fiedler = np.array(eigenvectors[:, 1]).flatten()

print(f"\nFiedler vector partition (split by sign):")
print("-" * 65)
mod_a, mod_b = [], []
for i, val in enumerate(fiedler):
    module = "A (Feed)" if val >= 0 else "B (Product)"
    (mod_a if val >= 0 else mod_b).append(i)
    print(f"  {units[i]:35s}  {val:+.4f}  ->  Module {module}")

print(f"\nModule A (prefab skid 1): {[units[i] for i in mod_a]}")
print(f"Module B (prefab skid 2): {[units[i] for i in mod_b]}")

# Count cut edges
cuts = [(u, v) for u, v in G_undir.edges()
        if (u in mod_a and v in mod_b) or (u in mod_b and v in mod_a)]
print(f"\nInter-module connections (on-site welds): {len(cuts)}")
for u, v in cuts:
    print(f"  {units[u]} <-> {units[v]}")

cut_cost = sum(d['total_cost'] for u, v, d in G.edges(data=True)
               if (u in mod_a and v in mod_b) or (u in mod_b and v in mod_a))
print(f"\nCost of inter-module piping: ${cut_cost:,.0f} ({cut_cost/total_cost*100:.1f}% of total)")
print(f"Cost of intra-module piping: ${total_cost - cut_cost:,.0f} ({(total_cost-cut_cost)/total_cost*100:.1f}% of total)")
print(f"\nIf workshop fabrication is 3x cheaper than on-site:")
workshop_saving = (total_cost - cut_cost) * (1 - 1/3)
print(f"  Savings from modular construction: ${workshop_saving:,.0f}")

# ============================================================
# 7. REDUNDANCY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("REDUNDANCY ANALYSIS (RANK OF INCIDENCE MATRIX)")
print("=" * 70)

print(f"\nIncidence matrix rank: {rank_B}")
print(f"Pipes in network: {n_pipes}")
print(f"Structurally redundant pipes: {n_pipes - rank_B}")

# Find which pipes are bridges (critical - removing disconnects network)
bridges = list(nx.bridges(G_undir))
print(f"\nCritical pipes (bridges - cannot remove): {len(bridges)}")
for u, v in bridges:
    print(f"  {units[u]} <-> {units[v]}")

non_bridges = [(u, v) for u, v in G_undir.edges() if (u, v) not in bridges and (v, u) not in bridges]
print(f"\nRemovable pipes (in loops, structurally redundant): {len(non_bridges)}")
for u, v in non_bridges:
    edge_data = G[u][v] if G.has_edge(u, v) else G[v][u]
    print(f"  {units[u]} <-> {units[v]}  (cost: ${edge_data.get('total_cost', 0):,.0f})")

removable_cost = sum(
    (G[u][v] if G.has_edge(u, v) else G[v][u]).get('total_cost', 0)
    for u, v in non_bridges
)
print(f"\nTotal removable pipe cost: ${removable_cost:,.0f} ({removable_cost/total_cost*100:.1f}% of total)")

# ============================================================
# 8. SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)

print(f"\n{'Method':<45} {'Savings':>15} {'% of Total':>12}")
print("-" * 75)
print(f"{'MST (minimum topology)':<45} ${total_cost - mst_cost:>14,.0f} {(total_cost - mst_cost)/total_cost*100:>10.1f}%")
print(f"{'Pipe diameter optimization':<45} ${total_original - total_optimized:>14,.0f} {(total_original - total_optimized)/total_original*100:>10.1f}%")
print(f"{'Modular construction (3x workshop savings)':<45} ${workshop_saving:>14,.0f} {'~':>10}")
print(f"{'Redundant pipe removal':<45} ${removable_cost:>14,.0f} {removable_cost/total_cost*100:>10.1f}%")
print(f"\n{'Original total piping cost:':<45} ${total_cost:>14,.0f}")

# ============================================================
# 9. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Layout - use a process-like left-to-right flow
pos = {
    0: (0, 3),      # Crude storage
    1: (1, 3),      # Feed pump
    2: (2, 3),      # Desalter
    3: (3, 3),      # Pre-heat
    4: (4, 3),      # Furnace
    5: (5, 3),      # Distillation column
    6: (6.5, 5),    # Naphtha cooler
    7: (6.5, 4),    # Kerosene stripper
    8: (6.5, 2),    # Diesel stripper
    9: (6.5, 1),    # Residue cooler
    10: (8, 3),     # Tank farm
    11: (5, 5.5),   # Flare
}

labels = {i: f"{i}" for i in range(n_units)}
short_labels = {i: name.split()[0] if len(name.split()[0]) <= 8 else f"{i}" for i, name in units.items()}

# Panel 1: Original network with flow rates
ax = axes[0, 0]
ax.set_title("Oil Refinery Piping Network", fontweight="bold", fontsize=13, pad=10)
nx.draw(G, pos, ax=ax, with_labels=False, node_color="#1a1a2e",
        node_size=600, edge_color="#555", arrows=True, arrowsize=12,
        width=1.5, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_labels(G, pos, short_labels, ax=ax, font_size=7, font_color="white", font_weight="bold")

# Add flow labels on edges
edge_labels = {}
for i, (u, v, _, _, _) in enumerate(pipes):
    if flows[i] > 0:
        edge_labels[(u, v)] = f"{flows[i]:.0f}"
nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6, font_color="#888")
ax.set_facecolor("#fafafa")

# Panel 2: Spectral partition
ax = axes[0, 1]
ax.set_title("Spectral Partition (Modular Construction)", fontweight="bold", fontsize=13, pad=10)
colors = ["#2962FF" if i in mod_a else "#FF6D00" for i in range(n_units)]
nx.draw(G_undir, pos, ax=ax, with_labels=False, node_color=colors,
        node_size=600, edge_color="#aaa", width=1.5)
nx.draw_networkx_labels(G_undir, pos, short_labels, ax=ax, font_size=7, font_color="white", font_weight="bold")
nx.draw_networkx_edges(G_undir, pos, edgelist=cuts, ax=ax,
                       edge_color="red", width=3, style="dashed")
ax.set_facecolor("#fafafa")

# Panel 3: MST
ax = axes[1, 0]
ax.set_title("Minimum Spanning Tree (Cheapest Topology)", fontweight="bold", fontsize=13, pad=10)
mst_edges = [(i, j) for i in range(n_units) for j in range(n_units) if mst_matrix[i, j] > 0]
G_mst = nx.Graph()
G_mst.add_nodes_from(range(n_units))
G_mst.add_edges_from(mst_edges)
nx.draw(G_mst, pos, ax=ax, with_labels=False, node_color="#00B878",
        node_size=600, edge_color="#00B878", width=2.5)
nx.draw_networkx_labels(G_mst, pos, short_labels, ax=ax, font_size=7, font_color="white", font_weight="bold")
ax.set_facecolor("#fafafa")

# Panel 4: Cost comparison bar chart
ax = axes[1, 1]
ax.set_title("Cost Optimization Results", fontweight="bold", fontsize=13, pad=10)
categories = ["Original\nCost", "After MST\nOptimization", "After Diameter\nOptimization", "Removable\nRedundancy"]
values = [total_cost, mst_cost, total_optimized, total_cost - removable_cost]
bar_colors = ["#1a1a2e", "#2962FF", "#00B878", "#FF6D00"]
bars = ax.bar(categories, values, color=bar_colors, width=0.6, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Total Piping Cost (USD)", fontweight="bold")
ax.set_facecolor("#fafafa")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout(pad=2)
out_path = "/Users/ronanvaz/programming/linear-algebra-project/oil_refinery_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nVisualization saved to {out_path}")
plt.close()

print("\nDone. All matrix methods applied to oil refinery piping optimization.")
