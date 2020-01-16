import Oceananigans.OutputWriters: saveproperty!

using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Solvers: permute_index, copy_pressure!, solve_poisson_equation!, PressureSolver
using Oceananigans.Utils: datatuples

"""
    RungeKuttaTimeStepper(float_type, arch, grid, tracers, χ)

Return an RungeKuttaTimeStepper object.
"""
struct RungeKuttaTimeStepper{T}
    G⁰ :: T
    G¹ :: T
end

function RungeKuttaTimeStepper(float_type, arch, grid, tracers)
   G⁰ = Tendencies(arch, grid, tracers)
   G¹ = Tendencies(arch, grid, tracers)
   return RungeKuttaTimeStepper{typeof(G¹)}(G⁰, G¹)
end

# Special savepropety! for the RK3 time stepper struct, which can be
# entirely reconstructed at any time-step.
saveproperty!(file, location, ts::RungeKuttaTimeStepper) = nothing

#####
##### Time steppping
#####

"""
    time_step!(model{<:RungeKuttaTimeStepper}, Nt, Δt)

Step forward `model` `Nt` time steps with step size `Δt` with a 3rd-order Runge-Kutta
timestepping method per Le and Moin (1991).
"""
function time_step!(model::Model{<:RungeKuttaTimeStepper}, Nt, Δt)

    if model.clock.iteration == 0
        [ run_diagnostic(model, diag) for diag in values(model.diagnostics) ]
        [ write_output(model, out)    for out  in values(model.output_writers) ]
    end

    for n in 1:Nt
        time_step!(model, Δt)
        [ time_to_run(model.clock, diag) && run_diagnostic(model, diag) for diag in values(model.diagnostics) ]
        [ time_to_run(model.clock, out) && write_output(model, out) for out in values(model.output_writers) ]
    end

    return nothing
end

"""
Step forward one time step with a 3rd-order Runge-Kutta method per 
Le and Moin (1991).
"""
function time_step!(model::Model{<:RungeKuttaTimeStepper}, Δt)

    FT = eltype(model.grid)

    # Constants per Le and Moin (1991)
    γ¹ = FT(8/15)
    γ² = FT(5/12)
    γ³ = FT(3/4)

    ζ¹ = -zero(FT)
    ζ² = -FT(17/60)
    ζ³ = -FT(5/12)

    # Substep times
    tⁿ = model.clock.time
    t¹ = tⁿ + (γ¹ + ζ¹) * Δt
    t² = t¹ + (γ² + ζ²) * Δt
    t³ = tⁿ + Δt

    # Convert NamedTuples of Fields to NamedTuples of OffsetArrays
    U, C, pressures, diffusivities = datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities)

    G⁰, G¹ = datatuples(model.timestepper.G⁰, model.timestepper.G¹)
    G² = G⁰ # Save some memory!
    G⁻ = G⁰ # Unused because ζ¹ = 0
        
    # Advance!
    take_rk3_substep!(U, C, pressures, diffusivities, G⁰, G⁻, γ¹, ζ¹, t¹, Δt, model)
    take_rk3_substep!(U, C, pressures, diffusivities, G¹, G⁰, γ², ζ², t², Δt, model)
    take_rk3_substep!(U, C, pressures, diffusivities, G², G¹, γ³, ζ³, t³, Δt, model)

    # We done now.
    model.clock.iteration += 1

    return nothing
end

function take_rk3_substep!(U, C, pressures, diffusivities, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, tᵏ, Δt, model)

    # Calculate Gᵏ⁻¹ = (Ψ̂ᵏ - Ψᵏ⁻¹) / Δt ≈ ∂t Ψ + ∇ᵤϕ
    calculate_tendencies!(Gᵏ⁻¹, U, C, pressures, diffusivities, model)

    # Perform pressure correction: Solve Δϕ = ∇⋅û / Δt
    update_predictor_velocities!(U, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, model.architecture, model.grid)
    calculate_pressure_correction!(pressures.pNHS, U, Δt, model)
    apply_pressure_correction!(U, pressures.pNHS, Δt, model.architecture, model.grid)

    # Step tracers forward with Cᵏ = Cᵏ⁻¹ + Δt * (γᵏ * Gᵏ⁻¹ + ζᵏ * Gᵏ⁻²)
    update_tracers!(C, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, model.architecture, model.grid)

    # Set model time to current substep time
    model.clock.time = tᵏ

    return nothing
end

#####
##### Runge-Kutta specific kernels
#####

function update_predictor_velocities!(U, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, arch, grid)
    @launch device(arch) config=launch_config(grid, :xyz) _update_predictor_velocities!(U, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, grid)
    return nothing
end

function _update_predictor_velocities!(U, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, grid)
    @loop_xyz i j k grid begin
        @inbounds U.u[i, j, k] += Δt * (γᵏ * Gᵏ⁻¹.u[i, j, k] + ζᵏ * Gᵏ⁻².u[i, j, k])
        @inbounds U.v[i, j, k] += Δt * (γᵏ * Gᵏ⁻¹.v[i, j, k] + ζᵏ * Gᵏ⁻².v[i, j, k])
        @inbounds U.w[i, j, k] += Δt * (γᵏ * Gᵏ⁻¹.w[i, j, k] + ζᵏ * Gᵏ⁻².w[i, j, k])
    end
    return nothing
end

function update_tracer!(c, Gcᵏ⁻¹, Gcᵏ⁻², γᵏ, ζᵏ, Δt, grid)
    @loop_xyz i j k grid begin
        @inbounds c[i, j, k] += Δt * (γᵏ * Gcᵏ⁻¹[i, j, k] + ζᵏ * Gcᵏ⁻²[i, j, k])
    end
    return nothing
end

function update_tracers!(C, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, arch, grid)
    for i in 1:length(C)
        @inbounds c = C[i]
        @inbounds Gcᵏ⁻¹ = Gᵏ⁻¹[i]
        @inbounds Gcᵏ⁻² = Gᵏ⁻²[i]
        @launch device(arch) config=launch_config(grid, :xyz) update_tracer!(c, Gcᵏ⁻¹, Gcᵏ⁻², γᵏ, ζᵏ, Δt, grid)
    end
    return nothing
end

function apply_pressure_correction!(U, Φᵏ, Δt, arch, grid)
    @launch device(arch) config=launch_config(grid, :xyz) _apply_pressure_correction!(U, Φᵏ, Δt, grid)
    return nothing
end

function _apply_pressure_correction!(U, Φᵏ, Δt, grid)
    @loop_xyz i j k grid begin
        @inbounds U.u[i, j, k] -= Δt * ∂xᶠᵃᵃ(i, j, k, grid, Φᵏ)
        @inbounds U.v[i, j, k] -= Δt * ∂yᵃᶠᵃ(i, j, k, grid, Φᵏ)
        @inbounds U.w[i, j, k] -= Δt * ∂zᵃᵃᶠ(i, j, k, grid, Φᵏ)
    end
    return nothing
end

#####
##### Pressure solver for RK3 scheme
#####

get_rhs(arch, solver) = solver.storage
get_rhs(arch::GPU, solver::PressureSolver{<:Channel}) = solver.storage.storage1

"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated with the velocity correction `velocities`, 
and step size `Δt` for the 3rd-order Runge-Kutta method described by Le and Moin (1991).
"""
function calculate_pressure_correction!(nonhydrostatic_pressure, predictor_velocities, Δt, model)
    predictor_velocities_boundary_conditions = (u=model.boundary_conditions.tendency.u,
                                                v=model.boundary_conditions.tendency.v,
                                                w=model.boundary_conditions.tendency.w)

    fill_halo_regions!(predictor_velocities, predictor_velocities_boundary_conditions, model.architecture,
                       model.grid, boundary_condition_function_arguments(model)...)

    solve_for_RK3_pressure!(nonhydrostatic_pressure, model.pressure_solver,
                            model.architecture, model.grid, predictor_velocities, Δt)

    fill_halo_regions!(nonhydrostatic_pressure, model.boundary_conditions.pressure,
                       model.architecture, model.grid)

    return nothing
end

function solve_for_RK3_pressure!(pressure, solver, arch, grid, U, Δt)
    ϕ = RHS = get_rhs(arch, solver)

    @launch(device(arch), config=launch_config(grid, :xyz),
            calculate_poisson_right_hand_side!(RHS, solver.type, arch, grid, U, Δt))

    solve_poisson_equation!(solver, grid)

    @launch(device(arch), config=launch_config(grid, :xyz),
            copy_pressure!(pressure, ϕ, solver.type, arch, grid))

    return nothing
end

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along any direction we need to perform a GPU fast cosine transform algorithm.
"""
function calculate_poisson_right_hand_side!(RHS, solver_type, arch, grid, U, Δt)
    @loop_xyz i j k grid begin
        i′, j′, k′ = permute_index(solver_type, arch, i, j, k, grid.Nx, grid.Ny, grid.Nz)

        @inbounds RHS[i′, j′, k′] = divᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w) / Δt
    end

    return nothing
end
