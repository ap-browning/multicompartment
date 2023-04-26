using DifferentialEquations
using Distributions

include("analytical.jl")

###############################################################
## SIMULATING THE SDE
###############################################################

## SDEProblem constructor
function make_prob(ν,μ,θ,k,σ;X₀=[μ;fill(0.0,ν)],tspan=[0.0,100.0])
    Θ = make_Θ(ν,μ,θ,k)
    M = make_M(ν,μ,θ,k)
    S = make_S(ν,σ)
    f(u,p,t) = -Θ * (u - M)
    g(u,p,t) = diag(S)
    SDEProblem(f,g,X₀,tspan)
end
function make_prob(Θ,M,S;X₀=M,tspan=[0.0,100.0])
    f(u,p,t) = -Θ * (u - M)
    g(u,p,t) = diag(S)
    SDEProblem(f,g,X₀,tspan)
end

## SDE simulation
function simulate(ν,μ,θ,k,σ;X₀=[μ;fill(0.0,ν)],tspan=[0.0,100.0],output=:func)
    prob = make_prob(ν,μ,θ,k,σ;X₀,tspan)
    if output == :func
        return solve(prob)
    elseif output == :final
        return solve(prob,saveat=tspan[end])(tspan[end])
    end
end
function simulate(Θ,M,S;X₀=M,tspan=[0.0,100.0],output=:func)
    prob = make_prob(Θ,M,S;X₀,tspan)
    if output == :func
        return solve(prob)
    elseif output == :final
        return solve(prob,saveat=tspan[end])(tspan[end])
    end
end


###############################################################
## SAMPLING FROM THE FPT
###############################################################

function sample_fpt_ou(μ,θ,k,σ;n=1,ã=1.0,X₀=:random)

    # Get stationary distribution
    X∞ = get_stationary_distribution(0,μ,θ,k,σ)

    # Get threshold a
    a = mean(X∞) + std(X∞) * ã

    # Create SDE problem
    f(u,p,t) = -θ * (u - μ)
    g(u,p,t) = σ
    prob = SDEProblem(f,g,0.0,(0.0,Inf))

    # Callback to stop solve after threshold breached
    callback = ContinuousCallback(
        (u,t,i) -> a - u[1],  # Condition
        terminate!              # Action (stop solve)
    )

    # Problem creator based on type of X₀
    if isa(X₀,Symbol)
        prob_func = (prob,i,repeat) -> remake(prob,u0 = rand(X∞),saveat=[])
    elseif isa(X₀,Function)
        prob_func = (prob,i,repeat) -> remake(prob,u0 = X₀(),saveat=[])
    else
        prob_func = (prob,i,repeat) -> remake(prob,u0 = X₀,saveat=[])
    end

    # Output function
    output_func = (sol,i) -> ([sol.u[1][end],sol.t[end]],false)

    # Solve ensemble problem
    ensemble_prob = EnsembleProblem(prob;prob_func,output_func)
    sols = solve(ensemble_prob;callback,trajectories=n)
    Xν₀ = collect([s[1] for s in sols])
    τ = collect([s[2] for s in sols])

    # Filter for Xν(0) > a
    τ[Xν₀ .> a] .= 0.0

    if n == 1
        τ = τ[1]
    end

    return τ
    
end

## Sample FPT
function sample_fpt(ν::Int,μ,θ,k,σ;n=1,ã=1.0,X₀=:random)

    if ν == 0
        return sample_fpt_ou(μ,θ,k,σ;n,ã,X₀)
    end

    # Get stationary distribution
    X∞ = get_stationary_distribution(ν,μ,θ,k,σ)

    # Get threshold a
    a = mean(X∞)[end] + sqrt(cov(X∞)[end]) * ã

    # Create SDE problem
    prob = make_prob(ν,μ,θ,k,σ,tspan=(0.0,Inf))

    # Callback to stop solve after threshold breached
    callback = ContinuousCallback(
        (u,t,i) -> a - u[end],  # Condition
        terminate!              # Action (stop solve)
    )

    # Problem creator based on type of X₀
    if isa(X₀,Symbol)
        prob_func = (prob,i,repeat) -> remake(prob,u0 = rand(X∞),saveat=[])
    elseif isa(X₀,Function)
        prob_func = (prob,i,repeat) -> remake(prob,u0 = X₀(),saveat=[])
    else
        prob_func = (prob,i,repeat) -> remake(prob,u0 = X₀,saveat=[])
    end

    # Output function
    output_func = (sol,i) -> ([sol.u[1][end],sol.t[end]],false)

    # Solve ensemble problem
    ensemble_prob = EnsembleProblem(prob;prob_func,output_func)
    sols = solve(ensemble_prob;callback,trajectories=n)
    Xν₀ = collect([s[1] for s in sols])
    τ = collect([s[2] for s in sols])

    # Filter for Xν(0) > a
    τ[Xν₀ .> a] .= 0.0

    if n == 1
        τ = τ[1]
    end

    return τ

end

## Get emperical CDF (i.e., from τ::Vector)
function Distributions.cdf(X::Vector;n=100,xlim=extrema(X))
    x = range(xlim...,n)
    F = [count(X .≤ xᵢ) for xᵢ in x] / length(X)
    return x,F
end