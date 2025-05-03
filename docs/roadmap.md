# Inferring Dependencies in Infrastructure Networks

1. Network vs. samnple size: How to learn with rank deficient covariance matrices?
2. Directionality and spatial embedding of dependencies: How to infer directionality from observed data? How to incorporate spatial constraints in the inference task, and is it needed beyond adjacencies?
3. How to leverage structurial prior knowledge (approximate edge weights, flow estimations and/or known eigenvectors), which can help as initial guess, regularization and/or spectral constraint?
4. Data integration:
   + Sparse data: Less data than nodes (only specific nodes are monitored, e.g. sensors for water quality in WDSs). Question of uncertainty and reconstruction under partial coverage.
   + Big data streams: What strategies enable scalable inference and data integration for large data over networks? How can we efficiently integrate new data?
6. Processes underlying the data (assumed):
   + How to model linear, diffusive-like processes (e.g., propagation of pollutants) over networks?
   + How to go beyond i.i.d. time series?
   + How to extend models to non-linear, advection-like, or interference processes (e.g., frequency/voltage disturbances over power grid)?
7. How to validate inferred dependencies, especially under partial observability and model misspecification?
8. How can inferred networks support real-time monitoring, risk assessment, and decision-making?
   + How can we decompose a signal in different modes?
   + How can we analyze shifts in stability region?
