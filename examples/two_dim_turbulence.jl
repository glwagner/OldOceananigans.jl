using Oceananigans, PyPlot

using Oceananigans.Operators

include("utils.jl")

function makeplot(model)
    plotxyslice(model.velocities.u)
    colorbar()
    return nothing
end

 N = 128
 L = 1.0
 κ = 1e-1
 U = 0.01
Δt = 0.01 * L/N / U

model = Model(N=(N, N, 1), L=(L, L, L), ν=κ, κ=κ)
f = model.constants.f

T0 = model.tracers.T
p₀(x, y, z) = f * U * N/L * rand()
p = CellField(p₀.(xnodes(T0), ynodes(T0), znodes(T0)), T0.grid)

for j = 1:N, i = 1:N
    @inbounds model.velocities.v[i, j, 1] = (
        δx_c2f(model.grid, p.data, i, j, 1) / model.grid.Δx / model.constants.f
    )
end

for j = 1:N, i = 1:N
    @inbounds model.velocities.u[i, j, 1] = (
        -δy_c2f(model.grid, p.data, i, j, 1) / model.grid.Δy / model.constants.f
    )
end

e0 = total_kinetic_energy(model)

for i = 1:10
    time_step!(model, 10, Δt)
    @show total_kinetic_energy(model) / e0
end
    fig, axs = subplots(nrows=1, figsize=(6, 6))
    makeplot(axs, model)
    gcf()
