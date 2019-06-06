@hascuda using CUDAnative, CuArrays

import GPUifyLoops: @launch, @loop, @unroll, @synchronize

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

include("kernels.jl")

dev(arch) = device(arch)

@inline data_tuple(obj, flds=propertynames(obj)) =
    Tuple(getproperty(getproperty(obj, fld), :data) for fld in flds)

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model::Model{A}, Nt, Δt) where A <: Architecture

    if model.clock.iteration == 0
        [ write_output(model, output_writer) for output_writer in model.output_writers ]
        [ run_diagnostic(model, diagnostic) for diagnostic in model.diagnostics ]
    end

    # Unpack model fields
              arch = model.arch
              grid = model.grid
               eos = model.eos
         constants = model.constants
           forcing = model.forcing
           closure = model.closure
     diffusivities = model.diffusivities

    # We can use the same array for the right-hand-side RHS and the solution ϕ.
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage
     U = data_tuple(model.velocities)
     ϕ = data_tuple(model.tracers)
    GU = data_tuple(model.G, (:Gu, :Gv, :Gw))
    pH, pN = data_tuple(model.pressures)
    Gⁿ = data_tuple(model.G)
    G⁻ = data_tuple(model.Gp)

    FT = eltype(grid)

     Txy = (Tx, Ty)
    Bxyz = (floor(Int, model.grid.Nx/Txy[1]), floor(Int, model.grid.Ny/Txy[2]), model.grid.Nz)
     Bxy = Bxyz[1:2]

    for n in 1:Nt

        # Adams-Bashforth (AB2) parameter.
        # Here we speciy that the loop always starts with a forward euler step. This
        # ensures that the algorithm is correct if Δt has changed since the previous step.
        χ = ifelse(n == 1, FT(-0.5), FT(0.125))

        # AB-2 preparation (could be done either before or after time-step):
        @launch dev(arch) threads=Txy blocks=Bxyz update_previous_source_terms!(grid, Gⁿ..., G⁻...)

        # Calc the right-hand-side of our PDE and store in Gⁿ:
        @launch dev(arch) threads=Txy blocks=Bxy update_hydrostatic_pressure!(pH, grid, constants, eos, ϕ...)
        @launch dev(arch) threads=Txy blocks=Bxyz calc_diffusivities!(diffusivities, grid, closure, eos, constants.g, U..., ϕ...)

        #@launch dev(arch) threads=Txy blocks=Bxyz calc_interior_source_terms!(grid, constants, eos, closure, pH, U..., ϕ..., Gⁿ..., diffusivities, forcing)

        @launch dev(arch) threads=Txy blocks=Bxyz calc_u_source_term!(grid, constants, eos, closure, pH, U..., ϕ..., Gⁿ[1], diffusivities, forcing, model.clock.iteration)
        @launch dev(arch) threads=Txy blocks=Bxyz calc_v_source_term!(grid, constants, eos, closure, pH, U..., ϕ..., Gⁿ[2], diffusivities, forcing, model.clock.iteration)
        @launch dev(arch) threads=Txy blocks=Bxyz calc_w_source_term!(grid, constants, eos, closure, pH, U..., ϕ..., Gⁿ[3], diffusivities, forcing, model.clock.iteration)
        @launch dev(arch) threads=Txy blocks=Bxyz calc_T_source_term!(grid, constants, eos, closure, pH, U..., ϕ..., Gⁿ[4], diffusivities, forcing, model.clock.iteration)
        #@launch dev(arch) threads=Txy blocks=Bxyz calc_S_source_term!(grid, constants, eos, closure, pH, U..., ϕ..., Gⁿ[5], diffusivities, forcing, model.clock.iteration)

        calc_boundary_source_terms!(model)

        # Use Gⁿ and G⁻ to perform the first AB-2 substep, obtaining u⋆:
        @launch dev(arch) threads=Txy blocks=Bxyz adams_bashforth_update_source_terms!(grid, Gⁿ..., G⁻..., χ)

        # Calculate the pressure correction based on the divergence of u⋆:
        @launch dev(arch) threads=Txy blocks=Bxyz calc_poisson_right_hand_side!(arch, grid, Δt, U..., GU..., RHS)
        solve_for_pressure!(arch, model)

        # Use the pressure correction to complete the second AB-2 substep, obtaining uⁿ⁺¹:
        @launch device(arch) threads=Txy blocks=Bxyz update_velocities_and_tracers!(grid, U..., ϕ..., pN, Gⁿ..., G⁻..., Δt)
        @launch device(arch) threads=Txy blocks=Bxy compute_w_from_continuity!(grid, U...)

        model.clock.time += Δt
        model.clock.iteration += 1

        for diagnostic in model.diagnostics
            (model.clock.iteration % diagnostic.diagnostic_frequency) == 0 && run_diagnostic(model, diagnostic)
        end

        for out in model.output_writers
            if model.clock.time > out.previous + out.interval
                write_output(model, out)
                out.previous = model.clock.time - rem(model.clock.time, out.interval)
            end
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
function calc_boundary_source_terms!(model::Model{A}) where A <: Architecture
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
    ϕ = model.tracers
    G = model.G

    grav = model.constants.g
    t, iteration = clock.time, clock.iteration
    u, v, w, T, S = U.u.data, U.v.data, U.w.data, ϕ.T.data, ϕ.S.data
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
