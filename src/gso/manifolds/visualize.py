import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

def plot_manifold(points: np.ndarray, 
                equations: list[sp.Expr], 
                variables: list[sp.Symbol],
                padding: float = 1.5,
                resolution: int = 100,
                threshold: float = 1e-3):
    """Plots a continuous algebraic manifold with sampled points."""
    ambient_dim = points.shape[1]
    
    if ambient_dim == 2:
        _plot_2d_manifold(points, equations, variables, padding, resolution)
    elif ambient_dim == 3:
        _plot_3d_manifold(points, equations, variables, padding, resolution, threshold)
    else:
        raise ValueError("Visualization only supports 2D or 3D ambient spaces")

def _plot_2d_manifold(points: np.ndarray,
                    equations: list[sp.Expr],
                    variables: list[sp.Symbol],
                    padding: float,
                    resolution: int):
    """Helper function for 2D manifold visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x_min, x_max = points[:, 0].min()-padding, points[:, 0].max()+padding
    y_min, y_max = points[:, 1].min()-padding, points[:, 1].max()+padding
    x, y = np.meshgrid(np.linspace(x_min, x_max, resolution), 
                     np.linspace(y_min, y_max, resolution))

    # Lambdify equations for numerical evaluation
    eq_funcs = [sp.lambdify(variables, eq, 'numpy') for eq in equations]

    # Plot each equation's zero contour
    for i, func in enumerate(eq_funcs):
        z = func(x, y)
        ax.contour(x, y, z, levels=[0], colors=[f'C{i}'], linewidths=2)

    # Plot valid points
    ax.scatter(points[:, 0], points[:, 1], c='red', s=30, label='Sampled Points')
    
    ax.set_xlabel(str(variables[0]))
    ax.set_ylabel(str(variables[1]))
    ax.legend()
    ax.set_title("2D Algebraic Manifold")
    plt.show()

def _plot_3d_manifold(points: np.ndarray,
                    equations: list[sp.Expr],
                    variables: list[sp.Symbol],
                    padding: float,
                    resolution: int,
                    threshold: float):
    """Helper function for 3D manifold visualization."""
    fig = go.Figure()
    
    # Create grid
    x_min, x_max = points[:, 0].min()-padding, points[:, 0].max()+padding
    y_min, y_max = points[:, 1].min()-padding, points[:, 1].max()+padding
    z_min, z_max = points[:, 2].min()-padding, points[:, 2].max()+padding
    
    x, y, z = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
        np.linspace(z_min, z_max, resolution),
        indexing='ij'
    )

    # Lambdify equations for numerical evaluation
    eq_funcs = [sp.lambdify(variables, eq, 'numpy') for eq in equations]

    # Plot isosurfaces for each equation
    for i, func in enumerate(eq_funcs):
        values = func(x, y, z)
        fig.add_trace(go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            isomin=-threshold,
            isomax=threshold,
            surface_count=1,
            opacity=0.3,
            colorscale='Viridis',
            name=f'Equation {i+1}'
        ))

    # Plot sampled points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Sampled Points'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title=str(variables[0]),
            yaxis_title=str(variables[1]),
            zaxis_title=str(variables[2])
        ),
        title="3D Algebraic Manifold",
        margin=dict(t=40, b=20)
    )
    fig.show()
