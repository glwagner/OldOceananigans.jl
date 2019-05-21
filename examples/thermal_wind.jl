using Oceananigans, Printf, PyPlot

include("utils.jl")

function makeplot(axs, model, T)
    T_num = model.tracers.T
    T_ans = FaceFieldX(T.(xnodes(T_num), ynodes(T_num), znodes(T_num)), model.grid)

    sca(axs[1])
    PyPlot.plot(T_ans[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(T_num[1, 1, :], "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(T_ans[1, 1, :] .- T_num[1, 1, :], "-")

    sca(axs[3])
    plotxzslice(T_num)

    sca(axs[4])
    plotxzslice(T_ans)

    return nothing
end

p₀ = 1e-1
 k = 1
 f = 1.0
 L = 2π
tf = L/100 * f / (p₀*k)
 κ = 1e-12
z₀ = -L/2
 δ = L/20

pᶻ(x, y, z) = p₀ * exp( -(z-z₀)^2 / (2*δ)^2 )
T₀(x, y, z, t=0) = -(z-z₀) / δ^2 * pᶻ(x, y, z) * sin(k*x)
v₀(x, y, z) = pᶻ(x, y, z) * cos(k*x) * k / f

 # Numerical parameters
for N in (16, 32, 64, 128)
    Δt = 0.01 * L/N * f / (p₀*k)
    @show Nt = round(Int, tf/Δt)


    # Create the model.
    model = Model(N=(N, 1, N), L=(L, L, L), ν=κ, κ=κ,
                    eos=LinearEquationOfState(βT=1.),
                    constants=PlanetaryConstants(f=f, g=1.))

    set_ic!(model, v=v₀, T=T₀)

    time_step!(model, Nt, Δt)

    fig, axs = subplots(nrows=2, ncols=2, figsize=(8, 8))
    makeplot(axs, model, T₀)
    gcf()

    @show T_relative_error(model, T₀)
end

gcf()
