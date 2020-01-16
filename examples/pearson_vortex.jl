using Oceananigans, PyPlot

nx = 32
ny = 32
Lx = 2π
Ly = 2π
Lz = 1
 ν = 1

Δt = 0.1 * (Lx/nx)^2

u(x, y, z, t) = - sin(y) * exp(-t)
v(x, y, z, t) =   sin(x) * exp(-t)

model = Model(
           grid = RegularCartesianGrid(size=(nx, ny, 1), length=(Lx, Ly, Lz)),
        closure = ConstantIsotropicDiffusivity(ν=1),
        tracers = nothing,
       buoyancy = nothing,
    timestepper = :AdamsBashforth
)

pressenter() = println("Press enter to continue"); readline()

u₀(x, y, z) = u(x, y, z, 0)
v₀(x, y, z) = v(x, y, z, 0)
set!(model, u=u₀, v=v₀)

meshgrid(x, y) = repeat(x, 1, length(y)), repeat(reshape(y, 1, length(y)), length(x), 1)

XU, YU = meshgrid(model.grid.xF[1:end-1], model.grid.yC)
XV, YV = meshgrid(model.grid.xC, model.grid.yF[1:end-1])

function difference(u, v, grid, time)
    Δu = @. interior(u) - u(grid.xF[1:end-1], grid.yC, grid.zC, time)
    Δv = @. interior(v) - v(grid.xC, grid.yF[1:end-1], grid.zC, time) 
    return Δu, Δv
end

fig, axs = subplots(ncols=2, nrows=2)

while true

    time_step!(model, 10, Δt)

    Δu, Δv = difference(u, v, model.grid, model.clock.time)

    sca(axs[1])
    cla()
    contourf(XU, YU, interior(model.velocities.u)[:, :, 1])

    sca(axs[2])
    cla()
    contourf(XV, YV, interior(model.velocities.v)[:, :, 1])

    sca(axs[4])
    cla()
    contourf(XU, YU, Δu[:, :, 1])

    sca(axs[3])
    cla()
    contourf(XV, YV, Δv[:, :, 1])

    fig.suptitle("t = $(model.clock.time)")

    pause(0.1)

    pressenter()
end
