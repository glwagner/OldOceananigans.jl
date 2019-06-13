using Random, Printf, JLD
const seed = 420  # Random seed to use for all pseudorandom number generators.

mutable struct JLDOutputWriter{N} <: OutputWriter
                 dir :: String
              prefix :: String
           fieldsets :: NTuple{N, Symbol}
    output_frequency :: Int
             padding :: Int
end

ext(::JLDOutputWriter) = ".jld"

function JLDOutputWriter(; dir=".", prefix="", fieldsets=(:velocities, :tracers, :G), frequency=1, padding=9)
    return JLDOutputWriter(dir, prefix, fieldsets, frequency, padding)
end

filename(iter, fw::JLDOutputWriter) = joinpath(fw.dir, fw.prefix * lpad(iter, fw.padding, "0") * ext(fw))

function write_output(model, fw::JLDOutputWriter)
    filepath = filename(model.clock.iteration, fw)
    write_jld_output(Tuple(getproperty(model, s) for s in fw.fieldsets), filepath)
end

function read_output(name, iter, fw::JLDOutputWriter)
    filepath = filename(iter, fw)
    data = load(filepath, name)
    return data
end

function write_jld_output(sets, filepath)
    allvars = Dict{String, Any}()
    for s in sets
        svars = Dict((String(fld), Array(getproperty(s, fld).data)) for fld in propertynames(s))
        merge!(allvars, svars)
    end

    save(filepath, allvars)

    return nothing
end

Closure(::Val{:ConstantSmagorinsky}, ν, κ) =
    ConstantSmagorinsky(Cs=-0.0, Cb=-0.0, Pr=1.0, ν=ν, κ=κ)

Closure(::Val{:AnisotropicMinimumDissipation}, ν, κ) =
    AnisotropicMinimumDissipation(C=-0.0, ν=ν, κ=κ)

Closure(::Val{:ConstantIsotropicDiffusivity}, ν, κ) =
    ConstantIsotropicDiffusivity(ν=ν, κ=κ)

function run_rayleigh_benard_regression_test(arch, closure)

    #
    # Parameters
    #
          α = 2                 # aspect ratio
          n = 1                 # resolution multiple
         Ra = 1e6               # Rayleigh number
    Nx = Ny = 8n * α            # horizontal resolution
    Lx = Ly = 1.0 * α           # horizontal extent
         Nz = 16n               # vertical resolution
         Lz = 1.0               # vertical extent
         Pr = 0.7               # Prandtl number
          a = 1e-1              # noise amplitude for initial condition
         Δb = 1.0               # buoyancy differential

    # Rayleigh and Prandtl determine transport coefficients
    ν = sqrt(Δb * Pr * Lz^3 / Ra)
    κ = ν / Pr

    #
    # Model setup
    #

    model = Model(
         arch = arch,
            N = (Nx, Ny, Nz),
            L = (Lx, Ly, Lz),
          eos = LinearEquationOfState(βT=1.),
    constants = PlanetaryConstants(g=1., f=0.),
      closure = Closure(Val(closure), ν, κ),
          bcs = BoundaryConditions(T=FieldBoundaryConditions(z=ZBoundaryConditions(
                       top = BoundaryCondition(Value, 0.0),
                    bottom = BoundaryCondition(Value, Δb)
                )))
    )

    ArrayType = typeof(model.velocities.u.data.parent)  # The type of the underlying data, not the offset array.
    Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, model.grid.Δz)^2 / ν

    spinup_steps = 1000
      test_steps = 100

    prefix = "data_rayleigh_benard_regression_"
    outputwriter = JLDOutputWriter(dir=".", prefix=prefix, frequency=test_steps)

    #
    # Initial condition and spinup steps for creating regression test data
    #

    #=
    ξ(z) = a * rand() * z * (Lz + z) # noise, damped at the walls
    b₀(x, y, z) = (ξ(z) - z) / Lz

    x, y, z = model.grid.xC, model.grid.yC, model.grid.zC
    x, y, z = reshape(x, Nx, 1, 1), reshape(y, 1, Ny, 1), reshape(z, 1, 1, Nz)

    model.tracers.T.data .= ArrayType(b₀.(x, y, z))

    println("Spinning up... ")

    @time begin
        time_step!(model, spinup_steps-test_steps, Δt)
        push!(model.output_writers, outputwriter)
    end

    time_step!(model, 2test_steps, Δt)
    =#

    #
    # Regression test
    #

    # Load initial state
    u₀ = read_output("u",  spinup_steps, outputwriter)
    v₀ = read_output("v",  spinup_steps, outputwriter)
    w₀ = read_output("w",  spinup_steps, outputwriter)
    T₀ = read_output("T",  spinup_steps, outputwriter)

    Gu = read_output("Gu", spinup_steps, outputwriter)
    Gv = read_output("Gv", spinup_steps, outputwriter)
    Gw = read_output("Gw", spinup_steps, outputwriter)
    GT = read_output("GT", spinup_steps, outputwriter)

    data(model.velocities.u) .= ArrayType(u₀)
    data(model.velocities.v) .= ArrayType(v₀)
    data(model.velocities.w) .= ArrayType(w₀)
    data(model.tracers.T)    .= ArrayType(T₀)

    data(model.G.Gu) .= ArrayType(Gu)
    data(model.G.Gv) .= ArrayType(Gv)
    data(model.G.Gw) .= ArrayType(Gw)
    data(model.G.GT) .= ArrayType(GT)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt

    # Step the model forward and perform the regression test
    time_step!(model, test_steps, Δt)

    u₁ = read_output("u", spinup_steps + test_steps, outputwriter)
    v₁ = read_output("v", spinup_steps + test_steps, outputwriter)
    w₁ = read_output("w", spinup_steps + test_steps, outputwriter)
    T₁ = read_output("T", spinup_steps + test_steps, outputwriter)

    field_names = ["u", "v", "w", "T"]
    fields = [model.velocities.u, model.velocities.v, model.velocities.w, model.tracers.T]
    fields_gm = [u₁, v₁, w₁, T₁]
    @show model.closure
    @info(@sprintf("Rayleigh-Benard regression test with closure: %s", closure))
    for (field_name, φ, φ_gm) in zip(field_names, fields, fields_gm)
        φ_min = minimum(Array(data(φ)) - φ_gm)
        φ_max = maximum(Array(data(φ)) - φ_gm)
        φ_mean = mean(Array(data(φ)) - φ_gm)
        φ_abs_mean = mean(abs.(Array(data(φ)) - φ_gm))
        φ_std = std(Array(data(φ)) - φ_gm)
        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       field_name, φ_min, φ_max, φ_mean, φ_abs_mean, φ_std))
    end

    # Now test that the model state matches the regression output.
    @test all(Array(data(model.velocities.u)) .≈ u₁)
    @test all(Array(data(model.velocities.v)) .≈ v₁)
    @test all(Array(data(model.velocities.w)) .≈ w₁)
    @test all(Array(data(model.tracers.T))    .≈ T₁)

    return nothing
end
