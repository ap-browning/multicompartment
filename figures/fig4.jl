#=
    Fig 4
=#

include("defaults.jl")
include("../results/analytical.jl")
include("../results/simulations.jl")
include("../results/volterra.jl")

using KernelDensity

# Parameters
θ = 1.0
k = 1.0
σ = 0.5
μ = 1.0
ã = 1.0

# Compartment to investigate
ν = 6

# Stationary solution
X∞ = get_stationary_distribution(ν[end],μ,θ,k,σ)

# Threshold for ν[end]
a = mean(marginal(X∞,ν)) + ã * std(marginal(X∞,ν))

# Initial condition
X₀ = () -> [rand(marginal(X∞,1)); zeros(ν+1)]

# Functions to perform computations
function sample_fpt_amount(k,a;n=100)

    τ = zeros(n); A = zeros(n);

    Θ = make_Θ(ν,μ,θ,k);
    M = make_M(ν,μ,θ,k);
    S = make_S(ν+1,σ)

    f(u,p,t) = [-Θ * (u[1:end-1] - M); k * u[end-1]]
    g(u,p,t) = diag(S)
    prob = SDEProblem(f,g,X₀(),(0.0,Inf))
    callback = ContinuousCallback((u,t,i) -> a - u[end-1],terminate!)

    for i = 1:n
        sol = solve(prob;callback)
        τ[i] = sol.t[end]
        A[i] = sol.u[end][end]
    end

    return τ,A

end
function get_absolute_threshold(k,ã)
    # Stationary solution
    X∞ = get_stationary_distribution(ν[end],μ,θ,k,σ)
    # Threshold for ν[end]
    mean(marginal(X∞,ν)) + ã * std(marginal(X∞,ν))
end
function sample_fpt_amount_relative(k,ã;n=100)
    a = get_absolute_threshold(k,ã)
    sample_fpt_amount(k,a;n)
end

## (a) k = ã = 1, scatter plot
k = ã = 1.0
τ,A = sample_fpt_amount_relative(k,ã;n=1000)

## Construct figure 4a
p3 = scatter(τ,A,c=:black,α=0.2,label="",xlim=(-5.0,155.0),ylim=(-5.0,155.0),ms=5,widen=false,
    xlabel="τ",ylabel="A(τ)")

x = range(-50.0,200.0)
f1 = pdf(kde(τ),x); f2 = pdf(kde(A),x);
p1 = plot(f2,x,ylim=(-5.0,155.0),xticks=0.0:0.01:0.03,lw=2.0,c=:black)
p2 = plot(x,f1,xlim=(-5.0,155.0),yticks=0.0:0.01:0.03,lw=2.0,c=:black)

fig4a = plot(p2,p3,p1,layout=@layout([
    a{0.3h} _
    c d{0.3w}
]),legend=:none,size=(450,400))
savefig("figures/fig4a.svg")

## (b) Fix k = 1, vary relative threshold ã
k = 1.0
ã = 0.1:0.1:2.0
a = get_absolute_threshold.(k,ã)
τ1,A1 = [Array{Any}(undef,length(ã)) for _ = 1:2]

for i = 1:length(ã)
    display("ã = $(ã[i])")
    τ1[i],A1[i] = sample_fpt_amount_relative(k,ã[i];n=1000)
end

    ## Produce Fig 4b
    l,u = [quantile.(A1,q) for q in [0.025,0.975]]
    m = mean.(A1)

    fig4b = scatter(ã,m,yerr=(m-l,u-m),msw=2.0,c=:black,lc=:black,msc=:black,ms=5,lw=2.0,
        yscale=:log,yticks=10.0.^(0:3),ylim=(0.5,2000.0),xlim=(0.0,2.05),label="",
        widen=true,xlabel="ã",ylabel="A(τ)")

## (c) Fix relative threshold ã = 1, vary k
ã = 1.0
k = 0.1:0.1:3.0
a = get_absolute_threshold.(k,ã)
τ2,A2 = [Array{Any}(undef,length(k)) for _ = 1:2]

for i = 1:length(k)
    display("k = $(k[i])")
    τ2[i],A2[i] = sample_fpt_amount_relative(k[i],ã;n=1000)
end

    ## Produce Fig 4c
    l,u = [quantile.(A2,q) for q in [0.025,0.975]]
    m = mean.(A2)

    fig4c = scatter(k,m,yerr=(m-l,u-m),msw=2.0,c=:black,lc=:black,msc=:black,ms=5,lw=2.0,
        yscale=:log,yticks=10.0.^(0:3),ylim=(0.5,2000.0),xlim=(0.0,3.1),label="",
        widen=true,xlabel="k",ylabel="A(τ)")

## Figure 4
fig4 = plot(fig4a,fig4b,fig4c,layout=grid(1,3),size=(1200,350))
add_plot_labels!(fig4)
savefig("figures/fig4.svg")