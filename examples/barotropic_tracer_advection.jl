using Oceananigans, PyPlot

include("utils.jl")

function makeplot(axs, model, T)
    T_ans = CellField(T.(xnodes(model.tracers.T), ynodes(model.tracers.T),
        znodes(model.tracers.T), model.clock.time), model.grid)

    T_diff = CellField(zeros(size(model.grid)), model.grid)
    @. T_diff.data = T_ans.data - model.tracers.T.data

    sca(axs[1])
    plotxyslice(T_ans)
    colorbar()

    sca(axs[2])
    plotxyslice(model.tracers.T)
    colorbar()

    sca(axs[3])
    plotxyslice(T_diff)
    colorbar()

    return nothing
end

 N = 128
 L = 1.0
 U = 0.5
 V = 0.8
 ν = κ = 1e-12
 δ = L/15
x₀ = L/2
y₀ = L/2

Δt = 0.05 * L/N / sqrt(U^2 + V^2)

T(x, y, z, t) = exp( -((x - U*t - x₀)^2 + (y - V*t - y₀)^2) / (2*δ^2) )
u₀(x, y, z) = U
v₀(x, y, z) = V
T₀(x, y, z) = T(x, y, z, 0)

model = Model(N=(N, N, 1), L=(L, L, L), ν=ν, κ=κ)

set_ic!(model, u=u₀, v=v₀, T=T₀)
model.clock.time = 0
model.clock.iteration = 0

fig, axs = subplots(nrows=3, figsize=(4, 10))
makeplot(axs, model, T)
gcf()

time_step!(model, 200, Δt)

fig, axs = subplots(nrows=3, figsize=(4, 10))
makeplot(axs, model, T)
gcf()

@show T_relative_error(model, T)
