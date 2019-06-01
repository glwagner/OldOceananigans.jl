Base.@kwdef struct AnisotropicMinimumDissipation{T} <: ScalarDiffusivity{T}
               C :: T = 0.33
    ν_background :: T = 1e-6
    κ_background :: T = 1e-7
end

"""
    AnisotropicMinimumDissipation(T=Float64; C=0.23, prandtl=1.0, ν_background=1e-6,
                                    κ_background=1e-7)

Returns a `AnisotropicMinimumDissipation` closure object of type `T` with

    * `C`            : Poincaré constant
    * `ν_background` : background viscosity
    * `κ_background` : background diffusivity
"""
AnisotropicMinimumDissipation(T; kwargs...) =
      typed_keyword_constructor(T, AnisotropicMinimumDissipation; kwargs...)

"Return the filter width for Anisotropic Minimum Dissipation on a Regular Cartesian grid."
Δx(i, j, k, grid::RegularCartesianGrid, ::AnisotropicMinimumDissipation) = grid.Δx
Δy(i, j, k, grid::RegularCartesianGrid, ::AnisotropicMinimumDissipation) = grid.Δy
Δz(i, j, k, grid::RegularCartesianGrid, ::AnisotropicMinimumDissipation) = grid.Δz

# We only have regular grids for now. When we have non-regular grids this will need to be changed.
const Δx_ccc = Δ
const Δy_ccc = Δ
const Δz_ccc = Δ

#
# Same-location products
#

# ccc
∂x_u²(ijk...) = ∂x_u(ijk...)^2
∂y_v²(ijk...) = ∂y_v(ijk...)^2
∂z_w²(ijk...) = ∂z_w(ijk...)^2

∂x_u²_Σ₁₁(ijk...) = ∂x_u²(ijk...) * Σ₁₁(ijk...)
∂y_v²_Σ₂₂(ijk...) = ∂y_v²(ijk...) * Σ₂₂(ijk...)
∂z_w²_Σ₃₃(ijk...) = ∂y_v²(ijk...) * Σ₃₃(ijk...)

# ffc
∂x_v²(ijk...) = ∂x_v(ijk...)^2
∂y_u²(ijk...) = ∂y_u(ijk...)^2

∂x_v_Σ₁₂(ijk...) = ∂x_v(ijk...) * Σ₁₂(ijk...)
∂y_u_Σ₁₂(ijk...) = ∂y_u(ijk...) * Σ₁₂(ijk...)

# fcf
∂z_u²(ijk...) = ∂z_u(ijk...)^2
∂x_w²(ijk...) = ∂x_w(ijk...)^2

∂x_w_Σ₁₃(ijk...) = ∂x_w(ijk...) * Σ₁₃(ijk...)
∂z_u_Σ₁₃(ijk...) = ∂z_u(ijk...) * Σ₁₃(ijk...)

# cff
∂z_v²(ijk...) = ∂z_v(ijk...)^2
∂y_w²(ijk...) = ∂y_w(ijk...)^2

∂z_v_Σ₂₃(ijk...) = ∂z_v(ijk...) * Σ₂₃(ijk...)
∂y_w_Σ₂₃(ijk...) = ∂y_w(ijk...) * Σ₂₃(ijk...)

#
# *** 30 terms ***
#

#
# the heinous
#

function Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, u, v, w)
    Δx = Δx_ccc(i, j, k, grid, closure)
    Δy = Δy_ccc(i, j, k, grid, closure)
    Δz = Δz_ccc(i, j, k, grid, closure)

    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ = Δx^2 * (
        ∂x_u²_Σ₁₁(ijkuvw...)
      +       Σ₂₂(ijkuvw...) * ▶xy_cca(ijk..., ∂x_v², uvw...)
      +       Σ₃₃(ijkuvw...) * ▶xz_cac(ijk..., ∂x_w², uvw...)

      +  2 * ∂x_u(ijkuvw...) * ▶xy_cca(ijk..., ∂x_v_Σ₁₂, uvw...)
      +  2 * ∂x_u(ijkuvw...) * ▶xz_cac(ijk..., ∂x_w_Σ₁₃, uvw...)
      +  2 * ▶xy_cca(ijk..., ∂x_v, uvw...) * ▶xz_cac(ijk..., ∂x_w, uvw...) * ▶yz_acc(ijk..., Σ₂₃, uvw...)
    )

    Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ = Δy^2 * (
      +       Σ₁₁(ijkuvw...) * ▶xy_ffa(ijk..., ∂y_u², uvw...)
      + ∂y_v²_Σ₂₂(ijkuvw...)
      +       Σ₃₃(ijkuvw...) * ▶yz_aff(ijk..., ∂y_w², uvw...)

      +  2 * ∂y_v(ijkuvw...) * ▶xy_ffa(ijk..., ∂y_u_Σ₁₂, uvw...)
      +  2 * ▶xy_cca(ijk..., ∂y_u, uvw...) * ▶yz_acc(ijk..., ∂y_w, uvw...) * ▶xz_cac(ijk..., Σ₁₃, uvw...)
      +  2 * ∂y_v(ijkuvw...) * ▶yz_acc(ijk..., ∂y_w_Σ₂₃, uvw...)
    )

    Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ = Δz^2 * (
      +       Σ₁₁(ijkuvw...) * ▶xz_cac(ijk..., ∂z_u², uvw...)
      +       Σ₂₂(ijkuvw...) * ▶yz_acc(ijk..., ∂z_v², uvw...)
      + ∂z_w²_Σ₃₃(ijkuvw...)

      +  2 * ▶xz_cac(ijk..., ∂z_u, uvw...) * ▶yz_acc(ijk..., ∂z_v, uvw...) * ▶xy_cca(ijk..., Σ₁₂, uvw...)
      +  2 * ∂z_w(ijkuvw...) * ▶xz_cac(ijk..., ∂z_u_Σ₁₃, uvw...)
      +  2 * ∂z_w(ijkuvw...) * ▶yz_acc(ijk..., ∂z_v_Σ₂₃, uvw...)
    )

    return Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ + Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ + Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ
end

#
# trace(∇u) = uᵢⱼ uᵢⱼ
#

tr_∇u_ccc(i, j, k, grid, u, v, w) = (
        # ccc
        ∂x_u²(i, j, k, grid, u, v, w)
      + ∂y_v²(i, j, k, grid, u, v, w)
      + ∂z_w²(i, j, k, grid, u, v, w)

        # ffc
      + ▶xy_cca(i, j, k, grid, ∂x_v², u, v, w)
      + ▶xy_cca(i, j, k, grid, ∂y_u², u, v, w)

        # fcf
      + ▶xz_cac(i, j, k, grid, ∂x_w², u, v, w)
      + ▶xz_cac(i, j, k, grid, ∂z_u², u, v, w)

        # cff
      + ▶yz_acc(i, j, k, grid, ∂y_w², u, v, w)
      + ▶yz_acc(i, j, k, grid, ∂z_v², u, v, w)
)
