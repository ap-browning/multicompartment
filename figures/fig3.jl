#=
    Fig 3
=#

using Interpolations
using ForwardDiff

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

# Compartments to investigate
ν = 0:6

# Stationary solution
X∞ = get_stationary_distribution(ν[end],μ,θ,k,σ)

# Threshold for ν[3] (for figure)
a = mean(marginal(X∞,4)) + ã * std(marginal(X∞,4))

# Numerical parameters
nsim = 1000     # For simulations
nvol = 200      # For Volterra equations
ωvol = 50.0     # Geometric mesh parameter for Volterra equations

# Grid over which to solve Volterra equations
T = [georange(tmin,80.0,nvol,ωvol) for tmin = [0.0,0.5]]
T̄ = [[mean(Tᵢ[i:i+1]) for i = 1:length(Tᵢ) - 1] for Tᵢ in T]

###############################################################
## ROW 1: Fully fixed initial condition
###############################################################

# Initial condition
X₀1 = [μ;fill(0.0,maximum(ν))]

## Sample FPT
@time τ1 = [sample_fpt(νᵢ,μ,θ,k,σ;n=nsim,ã,X₀=νᵢ == 0 ? X₀1[1] : X₀1[1:νᵢ+1]) for νᵢ in ν]

## Solve FPT using Volterra equation
f1 = Array{Any}(undef,2); F1 = copy(f1)
@time for i = 1:2
    p,K = setup_fptvolterra_fixedic(X₀1[1:ν[i]+1],ν[i],μ,θ,k,σ;ã,α=0.05)
    f1[i] = solve_volterra(T[i],p,K;α₁=0.0,α₂=0.001)
    F1[i] = midpoint_cumsum(T[i],f1[i])
end

## Produce row 1
fig3a_i,fig3a_ii,fig3b,fig3c = [plot() for i = 1:4]

    # Column 1
    for i = 1:5
        prob = make_prob(ν[end],μ,θ,k,σ;X₀=X₀1[1:ν[end]+1],tspan=(0.0,Inf))
        callback = ContinuousCallback((u,t,i) -> a - u[end],terminate!)
        sol = solve(prob;callback)
        plot!(fig3a_i,sol,idxs=1,c=palette(:PuBu_6,rev=true)[1],label="",α=0.5)
        scatter!(fig3a_i,[0.0],[μ],c=palette(:PuBu_6,rev=true)[1],label="",msw=0.0)
        plot!(fig3a_ii,sol,idxs=ν[end]+1,c=palette(:PuRd_6,rev=true)[3],label="")
        scatter!(fig3a_ii,[sol.t[end]],[sol.u[end][end]],c=palette(:PuRd_6,rev=true)[3],label="",msw=0.0)
    end
    fig3a = plot(fig3a_i,fig3a_ii,layout=grid(2,1),link=:x,xlim=(0.0,30.0))
    plot!(fig3a,subplot=1,ylabel="I(t)",ylim=(0.0,2.0))
    plot!(fig3a,subplot=2,ylabel="X3(t)",ylim=(0.0,1.5))
    hline!(fig3a,subplot=2,[a],lw=2.0,ls=:dot,c=:black,α=0.5,label="")

    # Column 2
    for (i,νᵢ) in enumerate(ν)
        # ECDF
        T0,F0 = cdf(τ1[i],xlim=(0.0,100.0),n=nvol)
        plot!(fig3b,T0,F0,c=cols[i],lw=2.0,label="",xlabel="τ",α=0.9)
        # VCDF
        if i ≤ 2
            plot!(fig3b,T[i],F1[i],c=:black,ls=:dash,label="ν = $νᵢ",lw=2.0)
        end
    end
    plot!(fig3b,xlim=(0.0,50.0),ylabel="F(τ)",legend=:none)

    # Column 3
    l,u = [quantile.(τ1,q) for q in [0.025,0.975]]
    m = mean.(τ1)
    scatter!(fig3c,ν,m,yerr=(m-l,u-m),msw=2.0,c=grey,lc=grey,msc=grey,ms=5,lw=2.0)
    m = [midpoint_cumsum(T[i],1 .- F1[i])[end] for i = 1:2]
    l = [linear_interpolation(F1[i],T[i])(0.025) for i = 1:2]
    u = [linear_interpolation(F1[i],T[i])(0.975) for i = 1:2]
    scatter!(fig3c,ν[1:2].+0.1,m,yerr=(m-l,u-m),msw=2.0,c=:black,lc=:black,msc=:black,ms=5,lw=2.0,legend=:none)

    # Add prediction based on scaled curvature
    ρ′′ = ν -> 1 / (1 - 2ν)
    mpred = [mean(τ1[2]) * sqrt(ρ′′(1) / ρ′′(νᵢ)) for νᵢ in ν[2:end]]
    plot!(fig3c,ν[2:end],mpred,ls=:dash,lw=2.0,c=:red)
    plot!(fig3c,xlim=(-0.2,6.2),ylabel="τ",xlabel="ν")

    # Row 1
    fig3r1 = plot(fig3a,fig3b,fig3c,layout=grid(1,3),widen=true)

###############################################################
## ROW 2: Partially-variable initial condition
###############################################################

# Initial condition
I₀ = marginal(X∞,1)
X₀2 = ν -> (() -> [rand(I₀); zeros(ν)])

## Sample FPT
τ2 = [sample_fpt(νᵢ,μ,θ,k,σ;n=nsim,ã,X₀=X₀2(νᵢ)) for νᵢ in ν[2:end]]

## Solve FPT using Volterra equation
f2 = Array{Any}(undef,1); F2 = copy(f1)
p,K = setup_fptvolterra_semifixedic([0.0],ν[2],μ,θ,k,σ;ã,α=0.05)
f2 = solve_volterra(T[2],p,K;α₁=0.0,α₂=0.001)
F2 = midpoint_cumsum(T[2],f2)

## Produce row 2
fig3d_i,fig3d_ii,fig3e,fig3f = [plot() for i = 1:4]

    # Column 1
    for i = 1:5
        prob = make_prob(ν[end],μ,θ,k,σ;X₀=X₀2(ν[end])(),tspan=(0.0,Inf))
        callback = ContinuousCallback((u,t,i) -> a - u[end],terminate!)
        sol = solve(prob;callback)
        plot!(fig3d_i,sol,idxs=1,c=palette(:PuBu_6,rev=true)[1],label="",α=0.5)
        scatter!(fig3d_i,[0.0],[sol(0.0)[1]],c=palette(:PuBu_6,rev=true)[1],label="",msw=0.0)
        plot!(fig3d_ii,sol,idxs=ν[end]+1,c=palette(:PuRd_6,rev=true)[3],label="")
        scatter!(fig3d_ii,[sol.t[end]],[sol.u[end][end]],c=palette(:PuRd_6,rev=true)[3],label="",msw=0.0)
    end
    fig3d = plot(fig3d_i,fig3d_ii,layout=grid(2,1),link=:x,xlim=(0.0,30.0))
    plot!(fig3d,subplot=1,ylabel="I(t)",ylim=(0.0,2.0))
    plot!(fig3d,subplot=2,ylabel="X3(t)",ylim=(0.0,1.5))
    hline!(fig3d,subplot=2,[a],lw=2.0,ls=:dot,c=:black,α=0.5,label="")

    # Column 2
    for i = 1:6
        # ECDF
        T0,F0 = cdf(τ2[i],xlim=(0.0,100.0),n=nvol)
        plot!(fig3e,T0,F0,c=cols[i+1],lw=2.0,label="",xlabel="τ",α=0.9)
        # VCDF
        if i == 2
            plot!(fig3e,T[2],F2,c=:black,ls=:dash,label="ν = 2",lw=2.0)
        end
    end
    plot!(fig3e,ylim=(0.0,1.0),xlim=(0.0,50.0),ylabel="F(τ)",legend=:none)

    # Column 3
    l,u = [quantile.(τ2,q) for q in [0.025,0.975]]
    m = mean.(τ2)
    scatter!(fig3f,ν[2:end],m,yerr=(m-l,u-m),msw=2.0,c=grey,lc=grey,msc=grey,ms=5,lw=2.0)
    m = [midpoint_cumsum(T[i],1 .- F1[i])[end] for i = 2:2]
    l = [linear_interpolation(F1[i],T[i])(0.025) for i = 2:2]
    u = [linear_interpolation(F1[i],T[i])(0.975) for i = 2:2]
    scatter!(fig3f,[ν[2]].+0.1,m,yerr=(m-l,u-m),msw=2.0,c=:black,lc=:black,msc=:black,ms=5,lw=2.0,legend=:none)

    # Add prediction based on scaled curvature
    # ρ′′ = ν -> begin
    #     acf_fun = ρ(ν,σ)
    #     ForwardDiff.hessian(x -> acf_fun(x[1]),[0.0])[1]
    # end
    ρ′′ = ν -> 1 / (1 - 2ν)
    mpred = [mean(τ2[1]) * sqrt(ρ′′(1) / ρ′′(νᵢ)) for νᵢ in ν[2:end]]
    plot!(fig3f,ν[2:end],mpred,ls=:dash,lw=2.0,c=:red)
    plot!(fig3f,xlim=(-0.2,6.2),ylabel="τ",xlabel="ν")

    # Row 1
    fig3r2 = plot(fig3d,fig3e,fig3f,layout=grid(1,3),widen=true)


###############################################################
## FIGURE 3
###############################################################

fig3 = plot(fig3r1,fig3r2,layout=grid(2,1),size=(800,450))
add_plot_labels!(fig3)
savefig(fig3,"$(@__DIR__)/fig3.svg")