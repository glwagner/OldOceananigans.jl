using Oceananigans, Printf, PyPlot

xnodes(ϕ) = repeat(reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ) = repeat(reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ) = repeat(reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

plotxzslice(ϕ, slice=1, args...; kwargs...) = pcolormesh(
    view(xnodes(ϕ), :, slice, :),
    view(znodes(ϕ), :, slice, :),
    view(ϕ.data, :, slice, :), args...; kwargs...)

Nx, Ny, Nz = 256, 1, 256
L = 100
ν = κ = 1e-4
f = 1e-4
Fu = -1e-6
dTdz = 0.0001

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(L, L, L), ν=ν, κ=κ, constants=Earth(f=1e-4))

model.boundary_conditions.u.z.left = BoundaryCondition(Flux, Fu)
model.boundary_conditions.T.z.right = BoundaryCondition(Gradient, dTdz)

T₀(x, y, z) = 20 + dTdz * z + 0.00001*rand()
u₀(x, y, z) = 1e-6 * rand()
w₀(x, y, z) = 1e-6 * rand()

model.tracers.T.data .= T₀.(
    reshape(model.grid.xC, Nx, 1, 1),
    reshape(model.grid.yC, 1, Ny, 1),
    reshape(model.grid.zC, 1, 1, Nz))

model.velocities.u.data .= u₀.(
    reshape(model.grid.xC, Nx, 1, 1),
    reshape(model.grid.yC, 1, Ny, 1),
    reshape(model.grid.zC, 1, 1, Nz))

model.velocities.w.data .= w₀.(
    reshape(model.grid.xC, Nx, 1, 1),
    reshape(model.grid.yC, 1, Ny, 1),
    reshape(model.grid.zC, 1, 1, Nz))

fig, axs = subplots(nrows=2)

    time_step!(model, 10000, 1)

    clf()
    plotxzslice(model.velocities.w)
    colorbar()
    gcf()
