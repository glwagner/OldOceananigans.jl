using Oceananigans, Printf, PyPlot

include("utils.jl")

function makeplot_u(axs, model, u)
    u_num = model.velocities.u
    u_ans = FaceFieldX(u.(xnodes(u_num), ynodes(u_num), znodes(u_num),
                        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(u_ans[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(u_num[1, 1, :], "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(u_ans[1, 1, :] .- u_num[1, 1, :], "-")

    sca(axs[3])
    plotxzslice(u_num)

    sca(axs[4])
    plotxzslice(u_ans)

    return nothing
end

function makeplot_v(axs, model, v)
    v_num = model.velocities.v
    v_ans = FaceFieldY(v.(xnodes(v_num), ynodes(v_num), znodes(v_num),
                        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(v_ans[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(v_num[1, 1, :], "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(v_ans[1, 1, :] .- v_num[1, 1, :], "-")

    sca(axs[3])
    plotxzslice(v_num)

    sca(axs[4])
    plotxzslice(v_ans)

    return nothing
end

function makeplot_w(axs, model, w)
    w_num = model.velocities.w
    w_ans = FaceFieldZ(w.(xnodes(w_num), ynodes(w_num), znodes(w_num),
                        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(w_ans[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(w_num[1, 1, :], "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(w_ans[1, 1, :] .- w_num[1, 1, :], "-")

    sca(axs[3])
    plotxzslice(w_num)

    sca(axs[4])
    plotxzslice(w_ans)

    return nothing
end

function makeplot_T(axs, model, T)
    T_num = model.tracers.T
    T_ans = CellField(T.(xnodes(T_num), ynodes(T_num), znodes(T_num),
                        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(T_ans[1, 1, :] .- dropdims(mean(T_num.data, dims=(1, 2)), dims=(1, 2)), "-", linewidth=2, alpha=0.4)
    PyPlot.plot(T_num[1, 1, :] .- dropdims(mean(T_ans.data, dims=(1, 2)), dims=(1, 2)), "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(T_ans[1, 1, :] .- T_num[1, 1, :], "-")

    sca(axs[3])
    plotxzslice(T_num)

    sca(axs[4])
    plotxzslice(T_ans)

    return nothing
end

# Internal wave parameters
a₀ = 1e-6
 m = 12
 k = 2
 f = 0.2
 ℕ = 1.0
 σ = sqrt( (ℕ^2*k^2 + f^2*m^2) / (k^2 + m^2) )

@show σ/f

cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
U = a₀ * k * σ   / (σ^2 - f^2)
V = a₀ * k * f   / (σ^2 - f^2)
W = a₀ * m * σ   / (σ^2 - ℕ^2)
Θ = a₀ * m * ℕ^2 / (σ^2 - ℕ^2) # ∂b/∂t = - w N^2 ✓

 # Numerical parameters
@show N = 512
 L = 2π
Δt = 0.05 * 1/σ
 ν = κ = 1e-6
z₀ = -L/3
 δ = L/20
Nt = 200

a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

u(x, y, z, t) =           a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
v(x, y, z, t) =           a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
w(x, y, z, t) =           a(x, y, z, t) * W * cos(k*x + m*z - σ*t)
T(x, y, z, t) = ℕ^2 * z + a(x, y, z, t) * Θ * sin(k*x + m*z - σ*t)

u₀(x, y, z) = u(x, y, z, 0)
v₀(x, y, z) = v(x, y, z, 0)
w₀(x, y, z) = w(x, y, z, 0)
T₀(x, y, z) = T(x, y, z, 0)

# Create the model.
model = Model(N=(N, 1, N), L=(L, L, L), ν=ν, κ=κ,
                eos=LinearEquationOfState(ρ₀=1., βT=1.),
                constants=PlanetaryConstants(f=f, g=1.))

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀)
@show total_energy(model, ℕ)
@show total_kinetic_energy(model)

fig, axs = subplots(nrows=4, figsize=(8, 8))

for i = 1:3
    time_step!(model, Nt, Δt)
    makeplot_u(axs, model, v)
    #makeplot_v(axs, model, v)
    #makeplot_w(axs, model, w)
    #makeplot_T(axs, model, T)
    @show total_energy(model, ℕ)
    @show total_kinetic_energy(model)
    @show w_relative_error(model, w)
    @show T_relative_error(model, T)
end

gcf()
