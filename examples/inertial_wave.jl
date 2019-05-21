using Oceananigans, Printf, PyPlot

include("utils.jl")

function makeplot(axs, model, u, v, w)
    w_ans = FaceFieldZ(w.(
        xnodes(model.velocities.w),
        ynodes(model.velocities.w),
        znodes(model.velocities.w),
        model.clock.time), model.grid)

    u_ans = FaceFieldX(u.(
        xnodes(model.velocities.u),
        ynodes(model.velocities.u),
        znodes(model.velocities.u),
        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(w_ans.data[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(model.velocities.w.data[1, 1, :], "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(model.velocities.w.data[1, 1, :] .- w_ans.data[1, 1, :])

    sca(axs[3])
    PyPlot.plot(u_ans.data[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(model.velocities.u.data[1, 1, :], "--", linewidth=1)

    sca(axs[4])
    PyPlot.plot(model.velocities.u.data[1, 1, :] .- u_ans.data[1, 1, :])

    return nothing
end

# Numerical parameters
 N = 256
 L = 2π
 f = 1.0
Δt = 0.01
ν = κ = 1e-9

# Wave parameters
z₀ = -L/2
a₀ = 1e-9
 m = 12
 k = 8
@show σ = f*m/sqrt(k^2 + m^2)
 δ = L/15

# Analytical solution for an inviscid inertial wave
cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
 U = k * σ / (σ^2 - f^2)
 V = k * f / (σ^2 - f^2)
 W = m / σ

a(x, y, z, t) = a₀ * exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )
u(x, y, z, t) = a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
v(x, y, z, t) = a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
w(x, y, z, t) = a(x, y, z, t) * W * cos(k*x + m*z - σ*t)

u₀(x, y, z) = u(x, y, z, 0)
v₀(x, y, z) = v(x, y, z, 0)
w₀(x, y, z) = w(x, y, z, 0)

# Create the model.
model = Model(N=(N, 1, N), L=(L, L, L), ν=ν, κ=κ, constants=PlanetaryConstants(f=f))

set_ic!(model, u=u₀, v=v₀, w=w₀)
time_step!(model, 1, 1e-16)
e₀ = total_kinetic_energy(model)

fig, axs = subplots(nrows=4, figsize=(6, 8))

for i = 1:3
    time_step!(model, 1000, Δt)
    makeplot(axs, model, u, v, w)
    @show w_relative_error(model, w)
    @show u_relative_error(model, u)
    @show total_kinetic_energy(model) / e₀
end

gcf()
