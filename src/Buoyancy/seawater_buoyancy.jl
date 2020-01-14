"""
    SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancy{EOS}

Buoyancy model for seawater. `T` and `S` are either `nothing` if both
temperature and salinity are active, or of type `FT` if temperature
or salinity are constant, respectively.
"""
struct SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancy{EOS}
    gravitational_acceleration :: FT
    equation_of_state :: EOS
    T₀ :: T
    S₀ :: S
end

required_tracers(::SeawaterBuoyancy) = (:T, :S)
required_tracers(::SeawaterBuoyancy{FT, E, <:Nothing, <:Number}) where {FT, E} = (:T,) # active temperature only
required_tracers(::SeawaterBuoyancy{FT, E, <:Number, <:Nothing}) where {FT, E} = (:S,) # active salinity only

"""
    SeawaterBuoyancy([FT=Float64;] gravitational_acceleration = g_Earth,
                                  equation_of_state = LinearEquationOfState(FT))

Returns parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called 'g'), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy.
"""
function SeawaterBuoyancy(                        FT = Float64;
                          gravitational_acceleration = g_Earth,
                                   equation_of_state = LinearEquationOfState(FT),
                                         temperature = nothing,
                                            salinity = nothing)

    return SeawaterBuoyancy{FT, typeof(equation_of_state), 
                            temperature, salinity}(gravitational_acceleration, equation_of_state)
end

const TemperatureSeawaterBuoyancy = SeawaterBuoyancy{FT, E, <:Nothing, <:Number} where {FT, E}
const SalinitySeawaterBuoyancy = SeawaterBuoyancy{FT, E, <:Number, <:Nothing} where {FT, E}

@inline get_temperature_salinity(::SeawaterBuoyancy, C) = C.T, C.S
@inline get_temperature_salinity(b::TemperatureSeawaterBuoyancy, C) = C.T, b.S₀
@inline get_temperature_salinity(b::SalinitySeawaterBuoyancy, C) = b.T₀, C.S

"""
    ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the x-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_x b = g ( α ∂_x Θ - β ∂_x Sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `Sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂x_Θ`, `∂x_S`, `α`, and `β` are all evaluated at cell interfaces in `x`
and cell centers in `y` and `z`.
"""
@inline function ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂xᶠᵃᵃ(i, j, k, grid, T)
        - haline_contractionᶠᶜᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂xᶠᵃᵃ(i, j, k, grid, S) )
end

"""
    ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the y-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_y b = g ( α ∂_y Θ - β ∂_y Sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `Sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂y_Θ`, `∂y_S`, `α`, and `β` are all evaluated at cell interfaces in `y`
and cell centers in `x` and `z`.
"""
@inline function ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂yᵃᶠᵃ(i, j, k, grid, T)
        - haline_contractionᶜᶠᶜ(i, j, k, grid, b.equation_of_state, T, S) * ∂yᵃᶠᵃ(i, j, k, grid, S) )
end

"""
    ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the vertical derivative of buoyancy for temperature and salt-stratified water,

```math
∂_z b = N^2 = g ( α ∂_z Θ - β ∂_z Sᴬ ) ,
```

where `g` is gravitational acceleration, `α` is the thermal expansion
coefficient, `β` is the haline contraction coefficient, `Θ` is
conservative temperature, and `Sᴬ` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that `∂z_Θ`, `∂z_Sᴬ`, `α`, and `β` are all evaluated at cell interfaces in `z`
and cell centers in `x` and `y`.
"""
@inline function ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    T, S = get_temperature_salinity(b, C)
    return b.gravitational_acceleration * (
           thermal_expansionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * ∂zᵃᵃᶠ(i, j, k, grid, T)
        - haline_contractionᶜᶜᶠ(i, j, k, grid, b.equation_of_state, T, S) * ∂zᵃᵃᶠ(i, j, k, grid, S) )
end

@inline function buoyancy_perturbation(i, j, k, grid, b::SeawaterBuoyancy{FT, <:AbstractNonlinearEquationOfState}, C) where FT
    T, S = get_temperature_salinity(b, C)
    return - b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, T, S) / b.equation_of_state.ρ₀
end
