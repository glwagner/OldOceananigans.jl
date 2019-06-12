using Printf

import OffsetArrays: OffsetArray

# Source: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/src/trials.jl
function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        s = t / 1e9
        if s < 60
            value, units = s, "s"
        else
            value, units = (s / 60), "min"
        end
    end
    return string(@sprintf("%.3f", value), " ", units)
end

function OffsetArray(underlying_data, grid::Grid)
    # Starting and ending indices for the offset array.
    i1, i2 = 1 - grid.Hx, grid.Nx + grid.Hx
    j1, j2 = 1 - grid.Hy, grid.Ny + grid.Hy
    k1, k2 = 1 - grid.Hz, grid.Nz + grid.Hz
    OffsetArray(underlying_data, i1:i2, j1:j2, k1:k2)
end

function Base.zeros(T, ::CPU, grid)
    underlying_data = zeros(T, grid.Tx, grid.Ty, grid.Tz)
    OffsetArray(underlying_data, grid)
end

function Base.zeros(T, ::GPU, grid)
    underlying_data = CuArray{T}(undef, grid.Tx, grid.Ty, grid.Tz)
    underlying_data .= 0
    OffsetArray(underlying_data, grid)
end

# Default to type of Grid
Base.zeros(arch, g::Grid{T}) where T = zeros(T, arch, g)

Base.@kwdef mutable struct TimeStepWizard{T}
              cfl :: T = 0.1
    cfl_diffusion :: T = 2e-2
       max_change :: T = 2.0
       min_change :: T = 0.5
           max_Δt :: T = Inf
               Δt :: T = 0.01
end

function update_Δt!(wizard, model)
    Δt_advection = wizard.cfl           * cell_advection_timescale(model)
    Δt_diffusion = wizard.cfl_diffusion * cell_diffusion_timescale(model)

    # Desired Δt
    Δt = min(Δt_advection, Δt_diffusion)

    # Put the kibosh on if needed
    Δt = min(wizard.max_change * wizard.Δt, Δt)
    Δt = max(wizard.min_change * wizard.Δt, Δt)
    Δt = min(wizard.max_Δt, Δt)

    wizard.Δt = Δt

    return nothing
end

function cell_advection_timescale(u, v, w, grid)

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = u.grid.Δx
    Δy = u.grid.Δy
    Δz = u.grid.Δz

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

function cell_diffusion_timescale(ν, κ, grid)

    νmax = maximum(abs, ν)
    κmax = maximum(abs, κ)

    Δx = min(u.grid.Δx)
    Δy = min(u.grid.Δy)
    Δz = min(u.grid.Δz)

    Δ = min(Δx, Δy, Δz) # assuming diffusion is isotropic for now

    return min(Δ^2/νmax, Δ^2/κmax) 

end

function cell_advection_timescale(model)
    cell_advection_timescale(
                             model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid
                            )
end

function cell_diffusion_timescale(model)
    cell_diffusion_timescale(
                             model.closure.ν,
                             model.closure.κ,
                             model.grid
                            )
end


