using LinearAlgebra
using Kronecker
using SpecialFunctions
using Distributions
using HypergeometricFunctions

# Incomplete beta function
B(z,a,b) = z^a / a * _₂F₁(a,1-b,a+1,z)

# Beta function
B(a,b) = B(1.0,a,b)

# Gamma function (shorthand)
Γ(i) = gamma(i)

# Binomial coefficient
C(n,r) = Γ(n+1) / Γ(n - r + 1) / Γ(r + 1)

#= ############################################################
    Construct Θ, M, and S such that the system
        dX = -Θ * (X - M) dt + S dW
    where X = [I,X₁,...,Xν] is equivalent to our system.
=# 
function make_Θ(ν::Int,μ,θ,k)
    Tridiagonal(-[1.0;fill(k,ν-1)],[θ;fill(k,ν)],zeros(ν))
end
function make_M(ν::Int,μ,θ,k)
    Θ = make_Θ(ν,μ,θ,k)
    return [μ; μ * Θ[2:end,2:end] \ [1.0; fill(0.0,ν-1)]]
end
function make_M(μ,Θ::Matrix)
    ν = size(Θ,2) - 1
    return [μ; μ * Θ[2:end,2:end] \ [1.0; fill(0.0,ν-1)]]
end
function make_S(ν::Int,σ)
    Diagonal([σ;zeros(ν)])
end


#= ############################################################
    COVARIANCES
=# 

# Case for θ = k
function Σ∞(σ::Number,k::Number,i::Int,j::Int)
    factor = i == j == 1 ? 1.0 : min(i,j) == 1 ? k : k^2
    σ^2 / k / factor * Γ(i + j - 1) / (2^(i + j - 1) * Γ(i) * Γ(j))
end

# General case
function Σ∞(σ::Number,θ::Number,k::Number,i::Int,j::Int)
    θ == k && return Σ∞(σ,k,i,j)
    θ̂ = θ / k
    factor = i == j == 1 ? 1.0 : min(i,j) == 1 ? k : k^2
    if min(i,j) == 1
        return σ^2 / k / factor / (2θ̂) / (1 + θ̂)^(max(i,j) - 1)
    end
    σ^2 / k / factor * sum(C(i+q-3,q-1) / (2^(i+q-1) * θ̂ * (1 + θ̂)^(j-q)) for q = 1:(j-1)) + 
    σ^2 / k / factor * sum(C(j+q-3,q-1) / (2^(j+q-1) * θ̂ * (1 + θ̂)^(i-q)) for q = 1:(i-1))
end

# Construct whole matrix
function Σ∞(σ::Number,Θ::Union{Matrix,Tridiagonal})
    e₁ = zeros(prod(size(Θ))); e₁[1] = 1.0
    reshape(σ^2 * ((Θ ⊕ Θ) \ e₁),size(Θ)...)
end
function Σ∞(ν::Int,args...)
    display(args)
    [Σ∞(args...,i,j) for i = 1:ν+1, j = 1:ν+1]
end

#= ############################################################
    VARIANCES
=# 

# Case for θ = k
function σ²(ν::Int,σ::Number,k::Number=1.0)
    factor = ν == 0 ? 1.0 : k^2
    σ^2 / k / factor * Γ(2ν+1) / (2^(2ν+1) * Γ(ν+1)^2)
end

# General case
function σ²(ν::Int,σ::Number,θ::Number,k::Number)
    ν == 0 && return σ^2 / 2θ
    θ == k && return σ²(ν,σ,k)
    θ̂ = θ / k
    σ^2 / k ^3 / θ̂ / (1 - θ̂^2)^ν * B((1 - θ̂) / 2,ν,ν) / B(ν,ν)
end

# Approximation for θ = k
function σ̃²(ν::Number,σ::Number,k::Number=1)
    factor = ν == 0 ? 1.0 : k^2
    σ^2 / k / factor / 2 / sqrt(ν * π)
end

#= ############################################################
    AUTOCORRELATION FUNCTIONS
=# 

# Case θ = k
function ρ(ν::Int,k::Number)
    l -> exp(-k * l) * _₁F₁(-ν,-2ν,2l*k)
end

# General autocorrelation function
function ρ(ν::Int,θ::Number,k::Number)
    Θ = Matrix(make_Θ(ν,0.0,θ,k))
    Σ = Σ∞(1.0,Θ)
    l -> (exp(-Θ * l) * Σ)[end,end] / Σ[end,end]
end


#= ############################################################
    STATIONARY DISTRIBUTION
=# 
function get_stationary_distribution(ν,μ,θ,k,σ)
    if ν == 0
        return get_stationary_distribution_ou(μ,θ,σ)
    end
    Θ = make_Θ(ν,μ,θ,k)
    M = make_M(ν,μ,θ,k)
    Σ = Σ∞(σ,Θ)
    return MvNormal(M,Σ)
end
function get_stationary_distribution_ou(μ,θ,σ)
    Normal(μ,sqrt(σ^2 / 2θ))
end