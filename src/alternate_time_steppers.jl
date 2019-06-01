"""Store previous source terms before updating them."""
function store_previous_source_terms!(grid::RegularCartesianGrid{FT}, Gu::A, Gv::A, Gw::A, GT::A, GS::A, Gpu::A, 
                                      Gpv::A, Gpw::A, GpT::A, 
                                      GpS::A) where {FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gpu[i, j, k] = Gu[i, j, k]
                @inbounds Gpv[i, j, k] = Gv[i, j, k]
                @inbounds Gpw[i, j, k] = Gw[i, j, k]
                @inbounds GpT[i, j, k] = GT[i, j, k]
                @inbounds GpS[i, j, k] = GS[i, j, k]
            end
        end
    end
    @synchronize
end

"Update the hydrostatic pressure perturbation pHY′ and buoyancy δρ."
function update_buoyancy!(grid::RegularCartesianGrid{FT}, constants::PlanetaryConstants{FT}, eos::LinearEquationOfState{FT}, 
                          T::A, pHY′::A) where {FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}

    gΔz = constants.g * grid.Δz

    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = 0.5 * gΔz * δρ(eos, T, i, j, 1)
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = pHY′[i, j, k-1] + gΔz * 0.5 * (δρ(eos, T, i, j, k-1) + δρ(eos, T, i, j, k))
            end
        end
    end

    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_interior_source_terms!(grid::RegularCartesianGrid{FT}, constants::PlanetaryConstants{FT}, eos::LinearEquationOfState{FT}, 
                                          closure::TurbulenceClosure{FT}, u::A, v::A, w::A, T::A, S::A, pHY′::A, Gu::A, Gv::A, Gw::A, GT::A, 
                                          GS::A, eddy_diff) where {FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    grav = constants.g
    fCor = constants.f
    ρ₀ = eos.ρ₀

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # u-momentum equation
                @inbounds Gu[i, j, k] = (-u∇u(grid, u, v, w, i, j, k)
                                            + fv(grid, v, fCor, i, j, k)
                                            - δx_c2f(grid, pHY′, i, j, k) / (Δx * ρ₀)
                                            + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, eos, grav, u, v, w, T, S, eddy_diff)
                                            #+ F.u(grid, u, v, w, T, S, i, j, k)
                                           )

                # v-momentum equation
                @inbounds Gv[i, j, k] = (-u∇v(grid, u, v, w, i, j, k)
                                            - fu(grid, u, fCor, i, j, k)
                                            - δy_c2f(grid, pHY′, i, j, k) / (Δy * ρ₀)
                                            + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, eos, grav, u, v, w, T, S, eddy_diff)
                                           # + F.v(grid, u, v, w, T, S, i, j, k)
                                           )

                # w-momentum equation: comment about how pressure and buoyancy are handled
                @inbounds Gw[i, j, k] = (-u∇w(grid, u, v, w, i, j, k)
                                         + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, eos, grav, u, v, w, T, S, eddy_diff)
                                          #  + F.w(grid, u, v, w, T, S, i, j, k)
                                           )

                # temperature equation
                @inbounds GT[i, j, k] = (-div_flux(grid, u, v, w, T, i, j, k)
                                         + ∇_κ_∇ϕ(i, j, k, grid, T, closure, eos, grav, u, v, w, T, S, eddy_diff)
                                            #+ F.T(grid, u, v, w, T, S, i, j, k)
                                           )

                # salinity equation
                @inbounds GS[i, j, k] = (-div_flux(grid, u, v, w, S, i, j, k)
                                         + ∇_κ_∇ϕ(i, j, k, grid, S, closure, eos, grav, u, v, w, T, S, eddy_diff)
                                           # + F.S(grid, u, v, w, T, S, i, j, k)
                                          )
            end
        end
    end

    @synchronize
end

function adams_bashforth_update_source_terms!(grid::RegularCartesianGrid{FT}, Gu::A, Gv::A, Gw::A, GT::A, GS::A, Gpu::A, Gpv::A, Gpw::A, GpT::A, GpS::A, 
                                              χ::FT) where {FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gu[i, j, k] = (FT(1.5) + χ)*Gu[i, j, k] - (FT(0.5) + χ)*Gpu[i, j, k]
                @inbounds Gv[i, j, k] = (FT(1.5) + χ)*Gv[i, j, k] - (FT(0.5) + χ)*Gpv[i, j, k]
                @inbounds Gw[i, j, k] = (FT(1.5) + χ)*Gw[i, j, k] - (FT(0.5) + χ)*Gpw[i, j, k]
                @inbounds GT[i, j, k] = (FT(1.5) + χ)*GT[i, j, k] - (FT(0.5) + χ)*GpT[i, j, k]
                @inbounds GS[i, j, k] = (FT(1.5) + χ)*GS[i, j, k] - (FT(0.5) + χ)*GpS[i, j, k]
            end
        end
    end
    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_poisson_right_hand_side!(::CPU, grid::RegularCartesianGrid{FT}, Δt::TDT, u::A, v::A, w::A, Gu::A, Gv::A, Gw::A, 
                                            RHS) where {TDT, FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
            end
        end
    end

    @synchronize
end

function calculate_poisson_right_hand_side!(::GPU, grid::RegularCartesianGrid{FT}, Δt::FT, u::A, v::A, w::A, 
                                            Gu::A, Gv::A, Gw::A, 
                                            RHS) where {FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation
                # which is the first step in the DCT.
                if CUDAnative.ffs(k) == 1  # isodd(k)
                    @inbounds RHS[i, j, convert(UInt32, CUDAnative.floor(k/2) + 1)] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
                else
                    @inbounds RHS[i, j, convert(UInt32, Nz - CUDAnative.floor((k-1)/2))] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
                end
            end
        end
    end

    @synchronize
end

function idct_permute!(grid::RegularCartesianGrid{FT}, ϕ::A, pNHS::A) where {FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k <= Nz/2
                    @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
                else
                    @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
                end
            end
        end
    end

    @synchronize
end


function update_velocities_and_tracers!(grid::RegularCartesianGrid{FT}, u::A, v::A, w::A, T::A, S::A, pNHS::A, Gu::A, Gv::A, Gw::A, GT::A, 
                                        GS::A, Gpu::A, Gpv::A, Gpw::A, GpT::A, GpS::A, Δt::TDT) where {TDT, FT, A<:OffsetArray{FT, 3, <:AbstractArray{FT, 3}}}

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds u[i, j, k] = u[i, j, k] + (Gu[i, j, k] - (δx_c2f(grid, pNHS, i, j, k) / grid.Δx)) * Δt
                @inbounds v[i, j, k] = v[i, j, k] + (Gv[i, j, k] - (δy_c2f(grid, pNHS, i, j, k) / grid.Δy)) * Δt
                @inbounds T[i, j, k] = T[i, j, k] + (GT[i, j, k] * Δt)
                @inbounds S[i, j, k] = S[i, j, k] + (GS[i, j, k] * Δt)
            end
        end
    end

    @synchronize
end

"Compute the vertical velocity w from the continuity equation."
function compute_w_from_continuity!(grid::RegularCartesianGrid{T}, u::A, v::A, w::A) where {T, A<:OffsetArray{T, 3, <:AbstractArray{T, 3}}}
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds w[i, j, 1] = 0
            @unroll for k in 2:grid.Nz
                @inbounds w[i, j, k] = w[i, j, k-1] + grid.Δz * ∇h_u(i, j, k-1, grid, u, v)
            end
        end
    end

    @synchronize
end

#=
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
    bcs = model.boundary_conditions
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

    apply_bcs!(arch, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, grid, u, Gu, ν₃₃.ffc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, grid, v, Gv, ν₃₃.fcf,
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
function apply_z_bcs!(top_bc, bottom_bc, grid, ϕ, Gϕ, κ, closure, eos, g, t, iteration, u, v, w, T, S)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)

               κ_top = κ(i, j, 1,       grid, closure, eos, g, u, v, w, T, S)
            κ_bottom = κ(i, j, grid.Nz, grid, closure, eos, g, u, v, w, T, S)

               apply_z_top_bc!(top_bc,    i, j, grid, ϕ, Gϕ, κ_top,    t, iteration, u, v, w, T, S)
            apply_z_bottom_bc!(bottom_bc, i, j, grid, ϕ, Gϕ, κ_bottom, t, iteration, u, v, w, T, S)

        end
    end
    @synchronize
end
=#
