using .TurbulenceClosures

mutable struct Model{A<:Architecture, FT, G, V, TT, P, F, C, BC, GT, PS, D, AT}
              arch :: A                         # Computer `Architecture` on which `Model` is run.
              grid :: G                         # Grid of physical points on which `Model` is solved.
             clock :: Clock{FT}                 # Tracks iteration number and simulation time of `Model`.
               eos :: LinearEquationOfState{FT} # Defines relationship between temp, salinity, buoyancy.
         constants :: PlanetaryConstants{FT}    # Set of physical constants, inc. gravitational acceleration.
        velocities :: V                        # Container for velocity fields `u`, `v`, and `w`.
           tracers :: TT                        # Container for tracer fields.
         pressures :: P                        # Container for hydrostatic and nonhydrostatic pressure.
           forcing :: F                        # Container for forcing functions defined by the user
           closure :: C                        # Diffusive 'turbulence closure' for all model fields
               bcs :: BC                       # Container for 3d bcs on all fields.
                 G :: GT                       # Container for right-hand-side of PDE that governs `Model`
                Gp :: GT                       # RHS at previous time-step (for Adams-Bashforth time integration)
    poisson_solver :: PS                        # ::PoissonSolver or ::PoissonSolverGPU
     diffusivities :: D
    output_writers :: Array{OutputWriter, 1}    # Objects that write data to disk.
       diagnostics :: Array{Diagnostic, 1}      # Objects that calc diagnostics on-line during simulation.
        attributes :: AT
end

"""
    Model(; kwargs...)

Construct a basic `Oceananigans.jl` model.
"""
function Model(;
    # Model resolution and domain size
             N,
             L,
    float_type = Float64,
          grid = RegularCartesianGrid(float_type, N, L),
    # Model architecture
          arch = CPU(),
    # Isotropic transport coefficients (exposed to `Model` constructor for convenience)
             ν = 1.05e-6, κ = 1.43e-7,
       closure = MolecularDiffusivity(float_type, ν=ν, κ=κ),
    # Fluid and physical parameters
     constants = Earth(float_type),
           eos = LinearEquationOfState(float_type),
    # Forcing and boundary conditions for (u, v, w, T, S)
       forcing = Forcing(nothing, nothing, nothing, nothing, nothing),
           bcs = ModelBoundaryConditions(),
    # Output and diagonstics
    output_writers = OutputWriter[],
       diagnostics = Diagnostic[],
             clock = Clock{float_type}(0, 0),
        attributes = nothing
)

    arch == GPU() && !HAVE_CUDA && throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))

    # Initialize fields, including source terms and temporary variables.
       velocities = VelocityFields(arch, grid)
          tracers = TracerFields(arch, grid)
        pressures = PressureFields(arch, grid)
                G = SourceTerms(arch, grid)
               Gp = SourceTerms(arch, grid)
    diffusivities = TurbulentDiffusivities(arch, grid, closure)

    # Initialize Poisson solver.
    poisson_solver = PoissonSolver(arch, grid)

    Model(arch, grid, clock, eos, constants,
          velocities, tracers, pressures, forcing, closure, bcs,
          G, Gp, poisson_solver, diffusivities, output_writers, diagnostics,
          attributes)
end

arch(model::Model{A}) where A <: Architecture = A
float_type(m::Model) = eltype(model.grid)
add_bcs!(model::Model; kwargs...) = add_bcs(model.bcs; kwargs...)

function initialize_with_defaults!(eos, tracers, sets...)

    # Default tracer initial condition is deteremined by eos.
    tracers.S.data .= eos.S₀
    tracers.T.data .= eos.T₀

    # Set all further fields to 0
    for set in sets
        for fldname in propertynames(set)
            fld = getproperty(set, fldname)
            fld.data .= 0 # promotes to eltype of fld.data
        end
    end

    return nothing
end

"Dummy function and forcing default."
@inline zero_func(args...) = 0

function Forcing(; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk)
    (u=u, v=v, w=w, T=T)
end

time(m::Model) = m.clock.time
