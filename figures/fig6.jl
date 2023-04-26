#=
    Fig 6

    Continuum model
=#

# using DifferentialEquations
# using Plots
# using StatsPlots
# using Statistics
# using Kronecker
# using LinearAlgebra
# using StatsBase
# using Distributions
# using LaTeXStrings
# using JLD2
# using Interpolations

include("defaults.jl")
include("../results/analytical.jl")
include("../results/simulations.jl")
include("../results/volterra.jl")

# System parameters
θ = 1.0
σ = 0.5
μ = 1.0
tmax = 30.0
xmax = 10.0
k = 1.0
k̂ = 1.0 # Now a density, matches k = 1.0 for Δ = 1

# Sample noise (first compartment)
f = (u,p,t) -> -θ * (u - μ)
g = (u,p,t) -> σ
prob = SDEProblem(f,g,μ,(0.0,tmax))
X₀ = solve(prob)

# Solve PDE numerically (discrete model)
Δt = 0.1 # For plotting
function solve_discretised_model(Δx)

    T = 0:Δt:tmax
    X = 0:Δx:xmax

    function rhs!(dx,u,p,t)
        dx .= [-k̂ * (u[1] - X₀(t)); -k̂ * diff(u)] / Δx
    end
    prob = ODEProblem(rhs!,zeros(length(X[2:end])),(0.0,tmax))
    sol = solve(prob)

    # Collate solution
    U = [X₀.(T)'; hcat([sol(t) for t in T]...)]

    # Return 
    return X,T,U

end

## Map colours correctly
cmap = cgrad(cgrad(:PuRd)[2:end],100,rev=true)
get_colour = x -> x == 0 ? grey : cmap[max(1,Int(round(100x / 10.0)))]

## Fig 6a: Δx = 1 (discrete model)
Δx = 1
X1,T1,U1 = solve_discretised_model(Δx)

fig6a = plot([],[],[],label="")
for i = 1:length(X1)
    x = fill(X1[i],length(T1))
    plot!(fig6a,x,T1,U1[i,:],c=get_colour(X1[i]),lw=2.0,label="")
end
plot!(camera=(45,25),xticks=0:2:10,xlabel="ν",ylabel="t",zlabel="x(ν)")

## Fig 6b: Δx = 0.5 (discrete model)
Δx = 0.5
X2,T2,U2 = solve_discretised_model(Δx)

fig6b = plot([],[],[],label="")
for i = 1:length(X2)
    x = fill(X2[i],length(T2))
    plot!(fig6b,x,T2,U2[i,:],c=get_colour(X2[i]),lw=2.0,label="")
end
plot!(fig6b,camera=(45,25),xticks=0:2:10,xlabel="ν",ylabel="t",zlabel="x(ν)")

## Fig 6c: Δx ≪ 1 (discrete model)
Δx = 0.01
X3,T3,U3 = solve_discretised_model(Δx)
plt_idx = 1:20:length(X3)

fig6c = plot([],[],[],label="")
for i = plt_idx
    x = fill(X3[i],length(T3))
    plot!(fig6c,x,T3,U3[i,:],c=get_colour(X3[i]),lw=2.0,label="")
end
plot!(fig6c,fill(X3[end],length(T3)),T3 .+ k̂ * X3[end],U3[1,:],c=0.5*grey,lw=2.0,ylim=(0.0,30.0),label="")
plot!(fig6c,camera=(45,25),xticks=0:2:10,xlabel="ν",ylabel="t",zlabel="x(ν)")

## Figure
fig6 = plot(fig6a,fig6b,fig6c,layout=grid(1,3),size=(1000,500),widen=false,zlim=(0.0,2.0))
add_plot_labels!(fig6)
savefig("figures/fig6.svg")