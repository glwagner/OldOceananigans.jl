using Oceananigans, PyPlot

include("utils.jl")

# Parameters
 N = 128
 L = 1
 κ = 1.0
Δt = 1e-6
z₀ = -L/2
t₀ = 1e-3

T(x, y, z, t) = exp( -(z - z₀)^2 / (4κ*t) ) / sqrt(4π*κ*t)
T₀(x, y, z) = T(x, y, z, t₀)

function makeplot(axs, model, numstyle="--")

    T_num = model.tracers.T

    T_ans = CellField(T.(
        xnodes(T_num), ynodes(T_num), znodes(T_num),
        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(T_ans.data[1, 1, :], T_num.grid.zC, "-", linewidth=2, alpha=0.4)
    PyPlot.plot(T_num.data[1, 1, :], T_num.grid.zC, numstyle, linewidth=1)

    sca(axs[2])
    PyPlot.plot(T_num.data[1, 1, :] .- T_ans.data[1, 1, :], T_num.grid.zC)

    return nothing
end

model = Model(N=(1, 1, N), L=(L, L, L), κ=κ, ν=κ, eos=LinearEquationOfState(βT=0.0))

set_ic!(model, T=T₀)
model.clock.time = t₀

fig, axs = subplots(ncols=2)

makeplot(axs, model)

time_step!(model, 10000, Δt)

makeplot(axs, model)

gcf()
