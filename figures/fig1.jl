#=
    Fig 1 (c and d only)
=#

include("defaults.jl")
include("../results/analytical.jl")
include("../results/simulations.jl")
include("../results/volterra.jl")

# Parameters
θ = 1.0
k = 1.0
σ = 0.5
μ = 1.0

# Number of compartments
ν = 6

# Stationary distribution
X∞ = get_stationary_distribution(ν,μ,θ,k,σ)
I∞ = marginal(X∞,1)

# Initial condition
X₀ = [rand(I∞); zeros(ν)]

# Simulate
sol = simulate(ν,μ,θ,k,σ;X₀,tspan=[0.0,40.0])

# Figure 1
fig1c = plot(sol,idxs=1,c=cols[1],ylabel="I(t)",label="",lw=1.5)
fig1d = plot(sol,idxs=2:ν+1,c=cols[2:end]',ylabel="Xν(t)",label="",lw=1.5)

fig1 = plot(fig1c,fig1d,layout=grid(2,1),
        size=(400,450),widen=true,xlabel="t")
add_plot_labels!(fig1,offset=2)

savefig(fig1,"$(@__DIR__)/fig1.svg")