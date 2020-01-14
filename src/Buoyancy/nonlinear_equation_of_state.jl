""" Return the geopotential depth at `i, j, k` at cell centers. """
@inline Dᵃᵃᶜ(i, j, k, grid) = @inbounds -grid.zC[k]

""" Return the geopotential depth at `i, j, k` at cell z-interfaces. """
@inline Dᵃᵃᶠ(i, j, k, grid) = @inbounds -grid.zF[k]

# Dispatch shenanigans
T_and_S(i, j, k, T::AbstractArray, S::AbstractArray) = @inbounds T[i, j, k], S[i, j, k]
T_and_S(i, j, k, T::Number, S::AbstractArray) = @inbounds T, S[i, j, k]
T_and_S(i, j, k, T::AbstractArray, S::Number) = @inbounds T[i, j, k], S
T_and_S(i, j, k, T::Number, S::Number) = @inbounds T, S

# Basic functionality
@inline ρ′(i, j, k, grid, eos, T, S) = @inbounds ρ′(T_and_S(i, j, k, T, S)..., Dᵃᵃᶜ(i, j, k, grid), eos)

@inline thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, T, S) = @inbounds thermal_expansion(T_and_S(i, j, k, T, S)..., Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶠᶜᶜ(i, j, k, grid, eos, T, S) = @inbounds thermal_expansion(ℑxᶠᵃᵃ(i, j, k, grid, T), ℑxᶠᵃᵃ(i, j, k, grid, S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶠᶜ(i, j, k, grid, eos, T, S) = @inbounds thermal_expansion(ℑyᵃᶠᵃ(i, j, k, grid, T), ℑyᵃᶠᵃ(i, j, k, grid, S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, T, S) = @inbounds thermal_expansion(ℑzᵃᵃᶠ(i, j, k, grid, T), ℑzᵃᵃᶠ(i, j, k, grid, S), Dᵃᵃᶠ(i, j, k, grid), eos)

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos, T, S) = @inbounds haline_contraction(T_and_S(i, j, k, T, S)..., Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos, T, S) = @inbounds haline_contraction(ℑxᶠᵃᵃ(i, j, k, grid, T), ℑxᶠᵃᵃ(i, j, k, grid, S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos, T, S) = @inbounds haline_contraction(ℑyᵃᶠᵃ(i, j, k, grid, T), ℑyᵃᶠᵃ(i, j, k, grid, S), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos, T, S) = @inbounds haline_contraction(ℑzᵃᵃᶠ(i, j, k, grid, T), ℑzᵃᵃᶠ(i, j, k, grid, S), Dᵃᵃᶠ(i, j, k, grid), eos)
