# mypy: ignore-errors

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_attributed_graph(
    G,
    value_attr=None,
    edge_attr=None,
    node_scale=1,
    edge_scale=5,
    k=None,
    stretch_x=1,
    stretch_y=1,
    node_color="#1f77b4",
    edge_color="gray",
    node_alpha=0.8,
    edge_alpha=0.3,
    edge_cmap=cm.viridis,
    weight_as_color: bool = True,
    weight_as_width: bool = False,
    pos_color="green",
    neg_color="red",
    node_size=200,
    line_width=2,
    min_edge_width=1.5,
    elev=30,
    azim=45,
    figsize=(10, 5),
    fig_title=None,
    save_format="png",
):
    """
    Visualize a NetworkX graph in 3D with optional node weights (as value lines, with positive and
    negative color and orientation) and edge weights (as edge colors and/or thickness).

    Parameters:
    value_attr (str/None): Node attribute for vertical lines (None = no lines)
    edge_attr (str/None): Edge attribute for weight visualization (None = uniform)
    """

    # Validate attributes
    if value_attr is not None:
        if not all(value_attr in G.nodes[node] for node in G.nodes):
            raise ValueError(f"All nodes must have '{value_attr}' attribute")

    if edge_attr and not all(edge_attr in G.edges[edge] for edge in G.edges):
        raise ValueError(f"Edge attribute '{edge_attr}' missing from some edges")

    n_nodes = len(G.nodes)
    default_k = 1 / np.sqrt(n_nodes) if n_nodes > 0 else 0.1
    k = k if k is not None else default_k

    # Generate 3D layout with z=0 plane
    pos = nx.spring_layout(G, dim=3, seed=42, iterations=200, k=k)
    for node in pos:
        pos[node][0] *= stretch_x
        pos[node][1] *= stretch_y
        pos[node][2] = 0

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Extract coordinates
    node_xyz = np.array([pos[node] for node in G.nodes])

    # Plot nodes
    ax.scatter(*node_xyz.T, s=node_size, alpha=node_alpha, edgecolors="w", c=node_color)

    # Add edge features
    if edge_attr is None:
        edge_thickness = [min_edge_width] * len(G.edges())
        edge_colors = [edge_color] * len(G.edges())
    else:
        edge_weights = [G.edges[u, v][edge_attr] for u, v in G.edges()]
        min_weight = min(edge_weights)

        if weight_as_color:
            max_weight = max(edge_weights)
            norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
            edge_colors = [edge_cmap(norm(w)) for w in edge_weights]
        else:
            edge_colors = [mcolors.to_rgba(edge_color)] * len(G.edges())

        if weight_as_width:
            shifted_weights = [w - min_weight for w in edge_weights]
            max_shifted = max(shifted_weights)
            edge_thickness = [
                edge_scale * (w / max_shifted) + min_edge_width for w in shifted_weights
            ]
        else:
            edge_thickness = [min_edge_width] * len(G.edges())

    # Plot edges with thickness and color
    for (u, v), current_thickness, current_color in zip(
        G.edges(), edge_thickness, edge_colors, strict=False
    ):
        ax.plot(
            *np.array([pos[u], pos[v]]).T,
            color=current_color,
            alpha=edge_alpha,
            lw=current_thickness,
        )

    # Plot value lines if requested
    if value_attr is not None:
        for node in G.nodes:
            x, y, z = pos[node]
            value = G.nodes[node][value_attr]
            color = pos_color if value >= 0 else neg_color
            ax.plot(
                [x, x],
                [y, y],
                [z, z + value / node_scale],
                color=color,
                lw=line_width,
                solid_capstyle="round",
            )

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if fig_title:
        if save_format == "png":
            plt.savefig(f"{fig_title}.png", dpi=300, bbox_inches="tight")
        elif save_format in ["svg", "pdf"]:
            plt.savefig(f"{fig_title}.{save_format}", bbox_inches="tight")

    return ax
