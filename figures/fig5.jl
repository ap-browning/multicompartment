#=
    Fig 5: Feedback
=#

using Interpolations
using ForwardDiff
using FiniteDiff

include("defaults.jl")
include("../results/analytical.jl")
include("../results/simulations.jl")
include("../results/volterra.jl")

# Parameters
θ = 1.0
k = 1.0
σ = 0.5
μ = 1.0
ã = 1.0

ν = 6

# Function to get standard deviation of compartment 6 with feedback (n → m)
function std_with_feedback(n,m,ε)

    # Make no-feedback matrix
    Θ = Matrix(make_Θ(ν,μ,θ,k))
        
    # Add feedback
    Θ[n+1,n+1] += ε
    Θ[m+1,n+1] += -ε

    # Stationary standard deviation of final compartment
    sqrt(Σ∞(σ,Θ)[end])
end

# Function to get ACF curvature of compartment 6 with feedback (n → m)
function ρ′′_with_feedback(n,m,ε)
    # Make no-feedback matrix
    Θ = Matrix(make_Θ(ν,μ,θ,k))
        
    # Add feedback
    Θ[n+1,n+1] += ε
    Θ[m+1,n+1] += -ε

    # ACF function
    e⁻ᶿ,Σ = exp(Θ),Σ∞(σ,Θ)
    ρ = l -> real((e⁻ᶿ^abs(l[1]) * Σ)[end,end] / Σ[end,end])

    FiniteDiff.finite_difference_hessian(ρ,[0.0])[1]
end

# No-feedback standard deviation
σ₆ = std_with_feedback(1,1,0.0)

# No-feedback curvature
ρ₆ = abs(ρ′′_with_feedback(1,1,0.0))

# Standard deviations with feedback
M1 = [std_with_feedback(m,n,0.1) for m = 1:6, n = 1:6]
M2 = [std_with_feedback(m,n,1.0) for m = 1:6, n = 1:6]

# Convert to % change
M̄1 = 100(M1 .- σ₆) ./ σ₆
M̄2 = 100(M2 .- σ₆) ./ σ₆

# Standard deviations with feedback
P1 = [abs(ρ′′_with_feedback(m,n,0.1)) for m = 1:6, n = 1:6]
P2 = [abs(ρ′′_with_feedback(m,n,1.0)) for m = 1:6, n = 1:6]

# Convert to % change
M̄1 = 100(M1 .- σ₆) ./ σ₆
M̄2 = 100(M2 .- σ₆) ./ σ₆
P̄1 = 100(P1 .- ρ₆) ./ ρ₆
P̄2 = 100(P2 .- ρ₆) ./ ρ₆

# Plots
p1 = heatmap(M̄1,c=:PRGn_9,clim=(-6.0,6.0),aspect_ratio=:equal,xticks=1:6,yticks=1:6,xlim=(0.5,6.5),ylim=(0.5,6.5),xlabel="m",ylabel="n")
p2 = heatmap(M̄2,c=:PRGn_9,clim=(-35.0,35.0),aspect_ratio=:equal,xticks=1:6,yticks=1:6,xlim=(0.5,6.5),ylim=(0.5,6.5),xlabel="m",ylabel="n")
p3 = heatmap(P̄1,c=:PuOr_9,clim=(-20.0,20.0),aspect_ratio=:equal,xticks=1:6,yticks=1:6,xlim=(0.5,6.5),ylim=(0.5,6.5),xlabel="m",ylabel="n")
p4 = heatmap(P̄2,c=:PuOr_9,clim=(-50.0,50.0),aspect_ratio=:equal,xticks=1:6,yticks=1:6,xlim=(0.5,6.5),ylim=(0.5,6.5),xlabel="m",ylabel="n")
plot!(p1,[0.5,6.5],[0.5,6.5],c=:black,ls=:dash,lw=2.0,label="")
plot!(p2,[0.5,6.5],[0.5,6.5],c=:black,ls=:dash,lw=2.0,label="")
plot!(p3,[0.5,6.5],[0.5,6.5],c=:black,ls=:dash,lw=2.0,label="")
plot!(p4,[0.5,6.5],[0.5,6.5],c=:black,ls=:dash,lw=2.0,label="")

# Figure
fig5 = plot(p1,p2,p3,p4)
add_plot_labels!(fig5)
savefig("figures/fig5.svg")