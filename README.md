# multicompartment
 
Repository associated with the preprint ["Smoothing in linear multicompartment biological processes subject to stochastic input"](https://arxiv.org/abs/2305.09004).

 - Each figure in the article can be reproduced by running the corresponding file in the `figures` subdirectory (for example, `figures/fig1.jl`).
 - In `results/analytical.jl`, we provide code associated with all analytical results: for example, the covariances and autocorrelation functions.
 - In `results/simulations.jl`, we provide code used to simulate the SDE model, and sample from the first passage time distribution using simulation.
 - In `results/volterra.jl`, we provide code used to solve the Volterra equations of the first kind numerically.
