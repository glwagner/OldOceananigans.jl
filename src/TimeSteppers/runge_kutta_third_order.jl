import Oceananigans.OutputWriters: saveproperty!

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
function time_step!(model::Model{<:RungeKuttaTimeStepper}, Δt::FT) where FT

    # Constants per Le and Moin (1991)
    γ¹ = FT(8/15)
    γ² = FT(5/12)
    γ³ = FT(3/4)

    ζ¹ = zero(FT)
    ζ² = -FT(17/60)
    ζ³ = -FT(5/12)

    # Substep sizes
    h¹ = (γ¹ + ζ¹) * Δt
    h² = (γ² + ζ²) * Δt

    # Convert NamedTuples of Fields to NamedTuples of OffsetArrays
    U, C, pressures, diffusivities = datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities)

    ϕ = pressures.pNHS
    G⁰, G¹ = datatuples(model.timestepper.G⁰, model.timestepper.G¹)
    G² = G⁰ # Save some memory!
    G⁻ = G⁰ # Unused because ζ¹ = 0

    tⁿ = model.clock.time
    
    # Advance to first substep
    calculate_tendencies!(G⁰, U, C, pressures, diffusivities, model) # with solution at t = tⁿ.
    update_predictor_solution!(U, C, G⁰, G⁻, γ¹, ζ¹, Δt, model.architecture, model.grid)
    calculate_substep_pressure_correction!(ϕ, U, Δt, model)
    update_substep_velocities!(U, ϕ, Δt, model.grid)
    model.clock.time = tⁿ + h¹

    # Advance to second substep
    calculate_tendencies!(G¹, U, C, pressures, diffusivities, model) # with solution at first substep
    update_predictor_solution!(U, C, G¹, G⁰, γ², ζ², Δt, model.architecture, model.grid)
    calculate_substep_pressure_correction!(ϕ, U, Δt, model)
    update_substep_velocities!(U, ϕ, Δt, model.grid)
    model.clock.time = tⁿ + h¹ + h²

    # Complete third and final substep
    calculate_tendencies!(G², U, C, pressures, diffusivities, model) # with solution at second substep
    update_predictor_solution!(U, C, G², G¹, γ³, ζ³, Δt, model.architecture, model.grid)
    calculate_substep_pressure_correction!(ϕ, U, Δt, model)
    update_substep_velocities!(U, ϕ, Δt, model.grid)
    model.clock.time = tⁿ + Δt

    # We done now.
    model.clock.iteration += 1

    return nothing
end

#####
##### Runge-Kutta specific kernels
#####

"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated with the velocity correction `velocities`, and step size `Δt`.
"""
function calculate_substep_pressure_correction!(nonhydrostatic_pressure, velocities, Δt, model)
    velocities_boundary_conditions = (u=model.boundary_conditions.tendency.u,
                                      v=model.boundary_conditions.tendency.v,
                                      w=model.boundary_conditions.tendency.w)

    fill_halo_regions!(velocities, velocities_boundary_conditions, model.architecture,
                       model.grid, boundary_condition_function_arguments(model)...)

    solve_for_RK3_pressure!(nonhydrostatic_pressure, model.pressure_solver,
                             model.architecture, model.grid, velocities, Δt)

    fill_halo_regions!(nonhydrostatic_pressure, model.boundary_conditions.pressure,
                       model.architecture, model.grid)

    return nothing
end

function update_predictor_velocities!(U, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, grid)
    @loop_xyz i j k grid begin
        @inbounds U.u[i, j, k] += Δt * (γᵏ * Gᵏ⁻¹.u[i, j, k] + ζᵏ * Gᵏ⁻².u[i, j, k])
        @inbounds U.v[i, j, k] += Δt * (γᵏ * Gᵏ⁻¹.v[i, j, k] + ζᵏ * Gᵏ⁻².v[i, j, k])
        @inbounds U.w[i, j, k] += Δt * (γᵏ * Gᵏ⁻¹.w[i, j, k] + ζᵏ * Gᵏ⁻².w[i, j, k])
    end
    return nothing
end

function update_substep_velocities!(U, Φᵏ, Δt, grid)
    @loop_xyz i j k grid begin
        @inbounds U.u[i, j, k] -= Δt * ∂xᶠᵃᵃ(i, j, k, grid, Φᵏ)
        @inbounds U.v[i, j, k] -= Δt * ∂yᵃᶠᵃ(i, j, k, grid, Φᵏ)
        @inbounds U.w[i, j, k] -= Δt * ∂zᵃᵃᶠ(i, j, k, grid, Φᵏ)
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

function update_predictor_solution!(U, C, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, arch, grid)
    @launch device(arch) config=launch_config(grid, :xyz) update_predictor_velocities!(U, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, grid)
    update_tracers!(C, Gᵏ⁻¹, Gᵏ⁻², γᵏ, ζᵏ, Δt, arch, grid)
    return nothing
end
