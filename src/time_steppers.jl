@hascuda using CUDAnative, CuArrays

import GPUifyLoops: @launch, @loop, @unroll, @synchronize

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

include("kernels.jl")

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model::Model{A}, Nt, Δt) where A <: Architecture
    clock = model.clock
    model_start_time = clock.time
    model_end_time = model_start_time + Nt*Δt

    if clock.iteration == 0
        for output_writer in model.output_writers
            write_output(model, output_writer)
        end
        for diagnostic in model.diagnostics
            run_diagnostic(model, diagnostic)
        end
    end

    arch = A()

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    # Unpack model fields
         grid = model.grid
        clock = model.clock
          eos = model.eos
    constants = model.constants
            U = model.velocities
           tr = model.tracers
           pr = model.pressures
      forcing = model.forcing
      closure = model.closure
    poisson_solver = model.poisson_solver
     diffusivities = model.diffusivities

    bcs = model.bcs
      G = model.G
     Gp = model.Gp

    # We can use the same array for the right-hand-side RHS and the solution ϕ.
    RHS, ϕ = poisson_solver.storage, poisson_solver.storage

    gΔz = model.constants.g * model.grid.Δz
    fCor = model.constants.f

    uvw = U.u.data, U.v.data, U.w.data
    TS = tr.T.data, tr.S.data
    Guvw = G.Gu.data, G.Gv.data, G.Gw.data

    # Source terms at current (Gⁿ) and previous (G⁻) time steps.
    Gⁿ = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data
    G⁻ = Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    tb = (threads=(Tx, Ty), blocks=(Bx, By, Bz))
    FT = eltype(grid)

    threads = (Tx, Ty)
    blocks = (Bx, By, Bz)

    for n in 1:Nt
        χ = ifelse(model.clock.iteration == 0, FT(-0.5), FT(0.125)) # Adams-Bashforth (AB2) parameter.

        @launch device(arch) threads=threads blocks=blocks store_previous_source_terms!(grid, Gⁿ..., G⁻...)

        @launch device(arch) update_hydrostatic_pressure!(
            pr.pHY′.data, grid, constants, eos, tr.T.data, tr.S.data, threads=(Tx, Ty), blocks=(Bx, By))

        @launch device(arch) threads=threads blocks=blocks calculate_diffusivities!(diffusivities,
            grid, closure, eos, constants.g, uvw..., TS...)

        @launch device(arch) threads=threads blocks=blocks calculate_interior_source_terms!(
            grid, constants, eos, closure, pr.pHY′.data, uvw..., TS..., Gⁿ..., diffusivities, forcing)

        calculate_boundary_source_terms!(model)

        @launch device(arch) threads=threads blocks=blocks adams_bashforth_update_source_terms!(
            grid, Gⁿ..., G⁻..., χ)

        @launch device(arch) threads=threads blocks=blocks calculate_poisson_right_hand_side!(
            arch, grid, Δt, uvw..., Guvw..., RHS)

        solve_for_pressure!(arch, model)

        @launch device(arch) threads=threads blocks=blocks update_velocities_and_tracers!(
            grid, uvw..., TS..., pr.pNHS.data, Gⁿ..., G⁻..., Δt)

        @launch device(arch) threads=threads blocks=(Bx, By) compute_w_from_continuity!(grid, uvw...)

        clock.time += Δt
        clock.iteration += 1

        #for diagnostic in model.diagnostics
        #    (clock.iteration % diagnostic.diagnostic_frequency) == 0 && run_diagnostic(model, diagnostic)
        #end

        for output_writer in model.output_writers
            (clock.iteration % output_writer.output_frequency) == 0 && write_output(model, output_writer)
        end
    end

    return nothing
end

time_step!(model; Nt, Δt) = time_step!(model, Nt, Δt)

function solve_for_pressure!(::CPU, model::Model)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage

    solve_poisson_3d_ppn_planned!(model.poisson_solver, model.grid)
    data(model.pressures.pNHS) .= real.(ϕ)
end

function solve_for_pressure!(::GPU, model::Model)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage

    Tx, Ty = 16, 16  # Not sure why I have to do this. Will be superseded soon.
    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    solve_poisson_3d_ppn_planned!(Tx, Ty, Bx, By, Bz, model.poisson_solver, model.grid)
    @launch device(GPU()) threads=(Tx, Ty) blocks=(Bx, By, Bz) idct_permute!(model.grid, ϕ, model.pressures.pNHS.data)
end

#
# Boundary condition physics specification
#

"Apply boundary conditions by modifying the source term G."
function calculate_boundary_source_terms!(model::Model{A}) where A <: Architecture
    arch = A()

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    grid = model.grid
    clock = model.clock
    eos =  model.eos
    closure = model.closure
    bcs = model.bcs
    U = model.velocities
    tr = model.tracers
    G = model.G

    grav = model.constants.g
    t, iteration = clock.time, clock.iteration
    u, v, w, T, S = U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data
    Gu, Gv, Gw, GT, GS = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    coord = :z #for coord in (:x, :y, :z) when we are ready to support more coordinates.

    u_x_bcs = getproperty(bcs.u, coord)
    v_x_bcs = getproperty(bcs.v, coord)
    w_x_bcs = getproperty(bcs.w, coord)
    T_x_bcs = getproperty(bcs.T, coord)
    S_x_bcs = getproperty(bcs.S, coord)

    # Apply boundary conditions in the vertical direction.

    # *Note*: for vertical boundaries in xz or yz, the transport coefficients should be evaluated at
    # different locations than the ones speciifc below, which are specific to boundaries in the xy-plane.

    apply_bcs!(arch, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, grid, u, Gu, ν₃₃.ccc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, grid, v, Gv, ν₃₃.ccc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    #apply_bcs!(arch, Val(coord), Bx, By, Bz, w_x_bcs.left, w_x_bcs.right, grid, w, Gw, ν₃₃.cff,
    #    closure, eos, grav, t, iteration, u, v, w, T, S)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, T_x_bcs.left, T_x_bcs.right, grid, T, GT, κ₃₃.ccc,
        closure, eos, grav, t, iteration, u, v, w, T, S)
    apply_bcs!(arch, Val(coord), Bx, By, Bz, S_x_bcs.left, S_x_bcs.right, grid, S, GS, κ₃₃.ccc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    return nothing
end

# Do nothing if both boundary conditions are default.
apply_bcs!(::CPU, ::Val{:x}, Bx, By, Bz,
    left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::CPU, ::Val{:y}, Bx, By, Bz,
    left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::CPU, ::Val{:z}, Bx, By, Bz,
    left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing

apply_bcs!(::GPU, ::Val{:x}, Bx, By, Bz,
    left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::GPU, ::Val{:y}, Bx, By, Bz,
    left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::GPU, ::Val{:z}, Bx, By, Bz,
    left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing

# First, dispatch on coordinate.
apply_bcs!(arch, ::Val{:x}, Bx, By, Bz, args...) =
    @launch device(arch) threads=(Tx, Ty) blocks=(By, Bz) apply_x_bcs!(args...)
apply_bcs!(arch, ::Val{:y}, Bx, By, Bz, args...) =
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, Bz) apply_y_bcs!(args...)
apply_bcs!(arch, ::Val{:z}, Bx, By, Bz, args...) =
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By) apply_z_bcs!(args...)

"Apply a top and/or bottom boundary condition to variable ϕ. Note that this kernel
MUST be launched with blocks=(Bx, By). If launched with blocks=(Bx, By, Bz), the
boundary condition will be applied Bz times!"
function apply_z_bcs!(top_bc, bottom_bc, grid, ϕ, Gϕ, κ, closure, eos,
                      grav, t, iteration, u, v, w, T, S)

    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)

               κ_top = κ(i, j, 1,       grid, closure, ϕ, eos, grav, u, v, w, T, S)
            κ_bottom = κ(i, j, grid.Nz, grid, closure, ϕ, eos, grav, u, v, w, T, S)

               apply_z_top_bc!(top_bc,    i, j, grid, ϕ, Gϕ, κ_top,    t, iteration, u, v, w, T, S)
            apply_z_bottom_bc!(bottom_bc, i, j, grid, ϕ, Gϕ, κ_bottom, t, iteration, u, v, w, T, S)

        end
    end
    @synchronize
end
