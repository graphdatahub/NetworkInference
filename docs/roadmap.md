# Inferring Dependencies in Infrastructure Networks

1. Network vs. sample size: How to learn with rank deficient covariance matrices?
2. Directionality and spatial embedding of dependencies: How to infer directionality from observed data? How to incorporate spatial constraints in the inference task, and is it needed beyond adjacencies?
3. Scalar node state: What if we cannot separate node states into single scalar variables $x_i(t)$ (that can be analyzed individually). Power system nodes (buses, generators) have multiple state variables (voltage magnitude, angle, frequency, generator rotor angle/speed) that can all shape the process we are studying.
4. How to leverage structurial prior knowledge (approximate edge weights, flow estimations and/or known eigenvectors), which can help as initial guess, regularization and/or spectral constraint?
5. Data integration:
   + Sparse data: Less data than nodes (only specific nodes are monitored, e.g. sensors for water quality in WDSs). Question of uncertainty and reconstruction under partial coverage (apply mask on data model). How to optimize sensor placement for (i) contamination detection and source identification, but also (ii) for most accurate flow reconstruction and identifiable models in general.
   + Big data streams: What strategies enable scalable inference and data integration for large data over networks? How can we efficiently integrate new data (online/incremental learning)?
6. Processes underlying the data (assumed):
   + How to model linear, diffusive-like processes (e.g., propagation of pollutants) over networks?
   + How to go beyond i.i.d. time series? Instead of maximizing variance (like PCA/EOFs), can we define optimality based on a better critera? For linear systems, eigenvectors of the system's dynamical matrix (aka the graph Laplacian for diffusion or consensus processes) represent natural modes of vibration or response. But for non-linear systems, can we find other criteria that help capture relevant modes of variations (e.g. maximizing temporal persistence, or predictability)?
   + How to extend models to non-linear, advection-like, or interference processes (e.g., frequency/voltage disturbances over power grid)—hence learning hidden geometry under different kernel-level assumption—? How to discriminate direct from indirect dependencies in those non-linear cases (since in linear cases we could attribute non-linear dependencies to indirect effects, and we the goal is not node embedding but graph inference: we dont want to extract general coupling strengths, or arrival times, described as distance in an embedding space, we want to recover actual node-to-node dependencies with the goal to find a meaningful basis to the frequency domain)?
7. How to validate inferred dependencies, especially under partial observability and model misspecification?
8. How can inferred networks support real-time monitoring, risk assessment, and decision-making?
   + How can we decompose a signal in different modes?
   + How can we analyze shifts in stability region?

**EPANET Study Case**

Compare results to that of the Laplacian we get from hydraulic parameter-dervied weights:
* hydraulic resistance (R) or its inverse, hydraulic conductance (1/R)
* Effective potential difference, based on elevation

Here we consider a system without pump. Maybe we can consider flow control valves.

**Approximation for Large Finite Systems**

* A primary approach involves investigating the mean field limit of kernels. By replacing complex interactions among many variables with an average representation of the other, the problem's complexity can be significantly reduced by neglecting higher-order correlations and fluctuations. Of course, the decision to use a mean-field approximation should align with the goals of using the specific kernel, since it works under the assumptions of homogeneity, exchangeability of variables, and weak, long-range interactions.
* Sampling-based techniques: ...
* Divide-and-conquer approaches:
* Quantum computing?
