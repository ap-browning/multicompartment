using .Threads
using Statistics
using Distributions
using LinearAlgebra

###############################################################
## MvNormal type (to avoid intermediate Cholesky decompositions)
###############################################################

# Type and constructor
struct MvNormalNoChol <: ContinuousMultivariateDistribution
    μ::Vector
    Σ::Matrix
end
MvNormalNoChol(μ::Vector,Σ::Matrix) = MvNormalNoChol(μ,Symmetric(Σ))

# Methods
Statistics.mean(d::MvNormalNoChol) = d.μ
Base.length(d::MvNormalNoChol) = length(d.μ)
Statistics.cov(d::MvNormalNoChol) = d.Σ

# Conversions to regular MvNormal
convert(::Type{MvNormalNoChol},d::MvNormal) = MvNormalNoChol(d.μ,d.Σ)
convert(::Type{MvNormal},d::MvNormalNoChol) = MvNormal(d.μ,d.Σ)

###############################################################
## SETUP VOLTERRA EQUATIONS
###############################################################

## Fixed initial condition
function setup_fptvolterra_fixedic(x₀,ν,μ,θ,k,σ;ã = 1.0,α=0.1)
 
    if ν == 0
        return setup_fptvolterra_fixedic_ou(x₀,μ,θ,k,σ;ã,α)
    end

    # Stationary distributions
    X∞ = get_stationary_distribution(ν,μ,θ,k,σ)
    Xν = marginal(X∞,ν+1)

    # Get actual barrier location
    a = mean(Xν) + std(Xν) * ã

    # Useful precomputations
    Θ = Matrix(make_Θ(ν,μ,θ,k))
    M = make_M(ν,μ,θ,k)
    S = make_S(ν,σ)
    e⁻ᶿ = exp(-Θ)
    e⁻ᶿ⁺ᶿ = exp(-(Θ ⊕ Θ))
    ΘΘ = Θ ⊕ Θ

    # Distribution of X(t) | x₀
    function X(t)
        MvNormalNoChol(
            M + e⁻ᶿ^t * (x₀ - M),
            Symmetric(reshape(ΘΘ \ ((I - e⁻ᶿ⁺ᶿ^t) * vec(S * S')),size(Θ)...))
        )
    end

    # Joint distribution of [X(s),X(t)]
    function X(s,t)
        Xs = X(s)
        Xt = X(t)
        Σ₁₂ = cov(Xs) * exp(-Θ' * abs(t - s))
        MvNormalNoChol([mean(Xs); mean(Xt)],Symmetric([cov(Xs) Σ₁₂; Σ₁₂' cov(Xt)]))
    end

    # p(t)
    function p(t)
        Xt = marginal(X(t),ν+1)
        pdf(Xt,a + α)
    end

    # K(s,t)
    function K(s,t)

        if s == t
            return 0.0
        end

        # Get distribution of [Xν-1(s),Xν(s),Xν(t)]
        d = marginal(X(s,t),[ν,ν+1,2ν+2])

        # Get [Xν-1(s),Xν(t)] conditioned on Xν(s) = a
        d = condition(marginal(d,[1,3,2]),a)

        # Calculate p(Xν(t) = a | Xν-1(s) > a)
        (1 - cdf(condition(d,a + α),a)) * pdf(marginal(d,2),a + α) / (1 - cdf(marginal(d,1),a))

    end

    return p,K

end

function setup_fptvolterra_fixedic_ou(x₀,μ,θ,k,σ;ã = 1.0,α=0.1)
 
    # Stationary distributions
    X∞ = get_stationary_distribution(0,μ,θ,k,σ)

    # Get actual barrier location
    a = mean(X∞) + std(X∞) * ã

    function X(t)
        m = x₀[1] * exp(-θ * t) + μ * (1 - exp(-θ * t))
        v = σ^2 / 2θ * (1 - exp(-2θ * t))
        Normal(m,√v)
    end

    function X(s,t)
        Xs = X(s)
        Xt = X(t)
        v = σ^2 / 2θ * (exp(-θ * abs(t - s)) - exp(-θ * (t + s)))
        m = [mean(Xs),mean(Xt)]
        Σ = [var(Xs) v; v var(Xt)]
        return MvNormal(m,Σ)
    end        

    # p(t)
    function p(t)
        pdf(X(t),a + α)
    end

    # K(s,t)
    function K(s,t)

        if s == t
            return 0.0
        end

        # Get distribution of [X(t),X(s)]
        d = marginal(X(s,t),[2,1])

        # Get Xν(t) conditioned on Xν(s) = a
        d = condition(d,a)
        pdf(d,a + α)
    end

    return p,K

end

## Partially fixed initial condition
function setup_fptvolterra_semifixedic(x₀,ν,μ,θ,k,σ;ã = 1.0,α=0.1)

    if ν == 0
        error("Not implemented for OU (is stationary problem)!")
    end

    # Stationary distributions
    X∞ = get_stationary_distribution(ν,μ,θ,k,σ)

    # Get actual barrier location
    a = mean(marginal(X∞,ν+1)) + std(marginal(X∞,ν+1)) * ã

    # Useful precomputations
    Θ = Matrix(make_Θ(ν,μ,θ,k))
    M = make_M(ν,μ,θ,k)
    S = make_S(ν,σ)
    e⁻ᶿ = exp(-Θ)
    e⁻ᶿ⁺ᶿ = exp(-(Θ ⊕ Θ))
    ΘΘ = Θ ⊕ Θ
    S₀ = zeros(ν+1,ν+1); S₀[1] = cov(X∞)[1,1]
    X̄₀ = [μ; x₀]

    # Distribution of X(t) | x₀
    function X(t)
        B = e⁻ᶿ^t
        MvNormalNoChol(
            M + B * (X̄₀ - M),
            Symmetric(reshape(ΘΘ \ ((I - e⁻ᶿ⁺ᶿ^t) * vec(S * S')),size(Θ)...)) + B * S₀ * B'
            )
    end

    # Joint distribution of [X(s),X(t)]
    function X(s,t)
        Bs,Bt = e⁻ᶿ^s,e⁻ᶿ^t
        B = [Bs zeros(ν+1,ν+1); zeros(ν+1,ν+1) Bt]
        μs = M + Bs * (X̄₀ - M)
        μt = M + Bt * (X̄₀ - M)
        Σs = Symmetric(reshape(ΘΘ \ ((I - e⁻ᶿ⁺ᶿ^s) * vec(S * S')),size(Θ)...))
        Σt = Symmetric(reshape(ΘΘ \ ((I - e⁻ᶿ⁺ᶿ^t) * vec(S * S')),size(Θ)...))
        Σ₁₂ = e⁻ᶿ^(t - s) * Σs
        MvNormalNoChol([μs; μt],[Σs Σ₁₂'; Σ₁₂ Σt] + B * [S₀ S₀; S₀ S₀] * B')
    end

    # p(t)
    function p(t)
        pdf(marginal(X(t),ν+1),a + α)
    end

    # K(s,t)
    function K(s,t)

        # Get distribution of [Xν-1(s),Xν(s),Xν(t)]
        d = marginal(X(s,t),[ν,ν+1,2ν+2])

        # Get [Xν-1(s),Xν(t)] conditioned on Xν(s) = a
        d = convert(MvNormal,condition(marginal(d,[1,3,2]),a))

        # Calculate p(Xν(t) = a + α | Xν-1(s) > a) using Bayes theorem
        (1 - cdf(condition(d,a + α),a)) * pdf(marginal(d,2),a + α) / (1 - cdf(marginal(d,1),a))

    end

    return p,K

end

###############################################################
## VOLTERRA EQUATION SOLVERS
###############################################################

## Geometric mesh generator
function georange(xmin,xmax,n,ω=1.0)
    α = ω ^ (1 / (n-1))
    Δ = α .^ (0:n-1)
    x = cumsum(Δ)
    x .-= minimum(x)
    x ./= maximum(x)
    x * (xmax - xmin) .+ xmin
end

function solve_volterra(T::Vector,p::Function,K::Function;solve=true,α₁=0.01,α₂=0.01)

    # version ∞: use midpoint rule and regularization
    T̄ = [mean(T[i:i+1]) for i = 1:length(T) - 1]
    Δ = diff(T)
    P = p.(T[2:end])

    # Setup linear system
    M = zeros(length(T̄),length(T̄))
    for n = 1:length(T̄)
        M[n,1:n] = K.(T̄[1:n],T[n+1]) .* Δ[1:n]
    end

    if !solve
        return P,M
    end

    # Regularization I: p(t) = α₁ f(t) + ∫ K(s,t) f(s) dt for α ≪ 1
    M₁ = M + α₁ * I

    # Regularization II: min|Mf - P|² + α₂|f|²
    f = (M₁' * M₁ + α₂ * I) \ (M₁' * P)

    return f
end

###############################################################
## TRAPEZOID RULE FOR COMPUTING CDF
###############################################################
function midpoint_cumsum(T,f;init=0.0)
    if length(T) == length(f)
        return midpoint_cumsum(T,[mean(f[i:i+1]) for i = 1:length(f)-1];init)
    end
    Δ = diff(T)
    F = similar(T); F[1] = init
    for i = 1:length(f)
        F[i+1] = F[i] + Δ[i] * f[i]
    end
    return F
end


###############################################################
## MARGINAL AND CONDITIONAL MULTIVARIATE NORMAL
###############################################################

"""
    condition(d,a₂)

Calculate x₁ | x₂ = a₂ where [x₁;x₂] ~ d::MvNormal

"""
function condition(d::Union{MvNormal,MvNormalNoChol},a₂::Vector;moments=false)
    # Indices of x₁,x₂
    i₁ = 1:(length(d) - length(a₂))
    i₂ = i₁[end]+1:length(d)
    # Required blocks
    μ₁,μ₂ = [mean(d)[i] for i = [i₁,i₂]]
    Σ₁₁,Σ₂₂ = [cov(d)[i,i] for i = [i₁,i₂]]
    Σ₁₂ = cov(d)[i₁,i₂]
    # Mean and variance of conditioned system
    Σ₂₂⁻¹ = inv(Σ₂₂)
    μ̄ = μ₁ + Σ₁₂ * Σ₂₂⁻¹ * (a₂ - μ₂)
    Σ̄ = Σ₁₁ - Σ₁₂ * Σ₂₂⁻¹ * Σ₁₂'
    if moments
        return μ̄,Σ̄
    end
    # Result
    i₁[end] == 1 ? Normal(μ̄[1],sqrt(max(0.0,Σ̄[1]))) : 
        isa(d,MvNormalNoChol) ? MvNormalNoChol(μ̄,Σ̄) : MvNormal(μ̄,Symmetric(Σ̄))
end
condition(d::Union{MvNormal,MvNormalNoChol},a₂::Number) = condition(d,[a₂])

"""
    marginal(d,idx)

Returns marginal distribution x[idx] where x ~ d::MvNormal
"""
function marginal(d::Union{MvNormal,MvNormalNoChol},idx::Union{Vector{Int},Int,UnitRange{Int}})
    μ̄ = mean(d)[idx]
    Σ̄ = cov(d)[idx,idx]
    length(μ̄) == 1 ? 
        Normal(μ̄[1],sqrt(max(0.0,Σ̄[1]))) :
        isa(d,MvNormalNoChol) ? MvNormalNoChol(μ̄,Σ̄) : MvNormal(μ̄,Σ̄)
end
