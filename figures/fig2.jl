#=
    Fig 2
=#

using ForwardDiff
using StatsBase

include("defaults.jl")
include("../results/analytical.jl")
include("../results/simulations.jl")
include("../results/volterra.jl")

# Parameters
θ = 1.0
k = 1.0
σ = 0.5
μ = 1.0

## Estimates standard deviation from simulation
@time std_estimates1 = hcat([std(hcat([simulate(10,μ,0.5,k,σ,output=:final) for _ = 1:100]...),dims=2) for _ = 1:10]...)
@time std_estimates2 = hcat([std(hcat([simulate(10,μ,1.0,k,σ,output=:final) for _ = 1:100]...),dims=2) for _ = 1:10]...)
@time std_estimates3 = hcat([std(hcat([simulate(10,μ,2.0,k,σ,output=:final) for _ = 1:100]...),dims=2) for _ = 1:10]...)

## Fig 2a: ν vs std
fig2a = plot(size=(250,200))

    # Case θ = 0.5, k = 1
    θ = 0.5; k = 1.0
    sd_fun = ν -> sqrt(σ²(ν,σ,θ,k))
    scatter!(fig2a,0:10,mean(std_estimates1,dims=2),yerr=std(std_estimates1,dims=2),c=grey,msc=grey,lw=2.0,msw=2.0,label="")
    plot!(fig2a,1:1:10,sd_fun.(1:1:10),c=grad[1],shape=:diamond,ms=5.0,lw=2.0,label="θ = $θ")

    # Case θ = k = 1
    θ = k = 1.0
    sd_fun = ν -> sqrt(σ²(ν,σ))
    scatter!(fig2a,0:10,mean(std_estimates2,dims=2),yerr=std(std_estimates2,dims=2),c=grey,msc=grey,lw=2.0,msw=2.0,label="")
    plot!(fig2a,0:1:10,sd_fun.(0:1:10),c=grad[2],shape=:diamond,ms=5.0,lw=2.0,label="θ = $θ")
    
    # Add approximation to θ = k = 1
    νgrid = range(1.0,10.0,100); vapprox = σ̃².(νgrid,σ)
    plot!(fig2a,νgrid,sqrt.(vapprox),lw=2.0,ls=:dash,c=:black,label="Approx")

    # Case θ = 2.0, k = 1 
    θ = 2.0; k = 1.0
    sd_fun = ν -> sqrt(σ²(ν,σ,θ,k))
    scatter!(fig2a,0:10,mean(std_estimates3,dims=2),yerr=std(std_estimates3,dims=2),c=grey,msc=grey,lw=2.0,msw=2.0,label="")
    plot!(fig2a,1:1:10,sd_fun.(1:1:10),c=grad[3],shape=:diamond,ms=5.0,lw=2.0,label="θ = $θ")

    plot!(fig2a,widen=true,xlim=(0.0,10.0),ylim=(0.0,0.55),xlabel="ν",ylabel="Stationary Std",xticks=0:2:10)
    
    
## Fig 2b: ACF
fig2b = plot()
    
    ν     = 3
    lags  = 0:1:20
    tspan = [0.0,3000.0]
    tcomp = 1000.0:1:3000.
    nsim  = 10

    # Case θ = 0.1, k = 1.0
    θ = 0.1; k = 1.0
    acf_fun = ρ(ν,θ,k)
    plot!(fig2b,acf_fun,c=grad[1],xlim=extrema(lags),ls=:dash,lw=2.0,label="θ = $θ")

    # Case θ = 0.5, k = 1
    θ = 0.5; k = 1.0
    acf_fun = ρ(ν,θ,k)
    acf_estimates1 = hcat([begin 
              s = simulate(ν,μ,θ,k,σ,tspan=tspan)
              autocor([s(t)[end] for t in tcomp],lags)
         end for _ = 1:nsim]...)
    scatter!(fig2b,lags,mean(acf_estimates1,dims=2),c=grey,msc=grey,lw=2.0,msw=2.0,shape=:diamond,label="θ = $θ")
    plot!(fig2b,acf_fun,c=grad[1],lw=2.0,label="θ = $θ")

    # Case θ = k = 1
    θ = k = 1.0
    acf_fun = ρ(ν,σ)
    acf_estimates2 = hcat([begin 
            s = simulate(ν,μ,θ,k,σ,tspan=tspan)
            autocor([s(t)[end] for t in tcomp],lags)
        end for _ = 1:nsim]...)
    scatter!(fig2b,lags,mean(acf_estimates2,dims=2),c=grey,msc=grey,lw=2.0,msw=2.0,shape=:star5,label="θ = $θ")
    plot!(fig2b,acf_fun,c=grad[2],lw=2.0,label="θ = $θ")

    # Case θ = 2.0, k = 1
    θ = 2.0; k = 1.0
    acf_fun = ρ(ν,σ,θ,k)
    acf_estimates3 = hcat([begin 
             s = simulate(ν,μ,θ,k,σ,tspan=tspan)
             autocor([s(t)[end] for t in tcomp],lags)
        end for _ = 1:nsim]...)
    scatter!(fig2b,lags,mean(acf_estimates3,dims=2),c=grey,msc=grey,lw=2.0,msw=2.0,label="θ = $θ")
    plot!(fig2b,acf_fun,c=grad[3],lw=2.0,label="θ = $θ")

    # Case θ = 10.0, k = 1.0
    θ = 10.0; k = 1.0
    acf_fun = ρ(ν,σ,θ,k)
    plot!(fig2b,acf_fun,c=grad[3],ls=:dash,lw=2.0,label="θ = $θ")

    plot!(fig2b,widen=true,xlim=(0.0,10.0),ylim=(0.0,1.0),xlabel="Lag",ylabel="ACF")

## Fig 2c: ACF curvature at zero
θ = k = 1
ρ′′ = ν -> begin
    acf_fun = ρ(ν,σ)
    ForwardDiff.hessian(x -> acf_fun(x[1]),[0.0])[1]
end
fig2c = scatter(1:10,ρ′′.(1:10),xlim=(0.0,10.0),lw=2.0,widen=true,xlabel="ν",ylabel="ρ''(0)",label="",c=grad[2],shape=:diamond)
plot!(fig2c,ν -> 1 / (1 - 2ν),xlim=(0.5,10.0),ylim=(-1.0,0.0),c=grad[2],lw=2.0,xticks=0:2:10,label="")
plot!(fig2c,xlim=(0.0,10.0))

## Fig 2d-f
xmax = 10.0
νmax = 6
k = 1.0

figs = [begin
    fig = plot(l -> exp(-θ * l),xlim=(0.0,xmax),c=cols[1],ls=:dash,lw=2.0,label="I(t)",xlabel="Lag",ylabel="ACF",title="θ = $θ")
    # Plot ACF
    for ν = 1:νmax
        acf_fun = θ == k == 1 ? ρ(ν,σ) : ρ(ν,σ,θ,k)
        plot!(fig,acf_fun,c=cols[ν+1],lw=2.0,label=ν==1 ? "Xν(t)" : "")
    end
    fig
end for θ = [0.5,1.0,2.0]]

## Figure 2
fig2 = plot(fig2a,fig2b,fig2c,figs...,size=(800,420))
add_plot_labels!(fig2)

savefig(fig2,"$(@__DIR__)/fig2.svg")
