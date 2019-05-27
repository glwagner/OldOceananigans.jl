using
    NetCDF,
    Plots,
    PyPlot,
    Oceananigans,
    Statistics

@hascuda using CuArrays

xnodes(ϕ) = reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1)
ynodes(ϕ) = reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1)
znodes(ϕ) = reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz)

xnodes(ϕ::FaceFieldX) = reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1)
ynodes(ϕ::FaceFieldY) = reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1)
znodes(ϕ::FaceFieldZ) = reshape(ϕ.grid.zF[1:end-1], 1, 1, ϕ.grid.Nz)

nodes(ϕ) = (xnodes(ϕ), ynodes(ϕ), znodes(ϕ))

fieldkind(ϕ::F) where F = F

x3d(ϕ) = repeat(reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
y3d(ϕ) = repeat(reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
z3d(ϕ) = repeat(reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

x3d(ϕ::FaceFieldX) = repeat(reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
y3d(ϕ::FaceFieldY) = repeat(reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
z3d(ϕ::FaceFieldZ) = repeat(reshape(ϕ.grid.zF[1:end-1], 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

zerofunk(args...) = 0

arraytype(::CPU) = Array
@hascuda arraytype(::GPU) = CuArray

function set_ic!(model::Model{A}; u=zerofunk, v=zerofunk, w=zerofunk, 
                    T=zerofunk, S=zerofunk) where A

    ArrayType = arraytype(A())

    model.velocities.u.data .= ArrayType(u.(nodes(model.velocities.u)...)) 
    model.velocities.v.data .= ArrayType(v.(nodes(model.velocities.v)...))
    model.velocities.w.data .= ArrayType(w.(nodes(model.velocities.w)...))
    model.tracers.T.data    .= ArrayType(T.(nodes(model.tracers.T)...))
    model.tracers.S.data    .= ArrayType(S.(nodes(model.tracers.S)...))

    model.clock.time = 0
    model.clock.iteration = 0

    for ϕname in (:Gu, :Gv, :Gw, :GT, :GS)
        ϕ = getproperty(model.G, ϕname)
        @. ϕ.data .= 0
        ϕ = getproperty(model.Gp, ϕname)
        @. ϕ.data .= 0
    end

    return nothing
end

function set_noslip_bcs!(model)
    model.boundary_conditions.u.z.top = BoundaryCondition(Value, -0.0)
    model.boundary_conditions.v.z.top = BoundaryCondition(Value, -0.0)
    model.boundary_conditions.u.z.bottom = BoundaryCondition(Value, -0.0)
    model.boundary_conditions.v.z.bottom = BoundaryCondition(Value, -0.0)
    return nothing
end

plotxzslice(ϕ, slice=1, args...; kwargs...) = pcolormesh(
    view(x3d(ϕ), :, slice, :), view(z3d(ϕ), :, slice, :), view(ϕ.data, :, slice, :), args...; kwargs...)

plotxyslice(ϕ, slice=1, args...; kwargs...) = pcolormesh(
    view(x3d(ϕ), :, :, slice), view(y3d(ϕ), :, :, slice), view(ϕ.data, :, :, slice), args...; kwargs...)

ΔV(grid) = grid.Δx * grid.Δy * grid.Δz

function buoyancy_flux(model::Model{A}) where A
    Nz = model.grid.Nz
    βT = model.eos.βT
     g = model.constants.g
     w = model.velocities.w
     T = model.tracers.T

    wb = CellField(A(), model.grid)

    @views @. wb.data[:, :, 1:Nz-1] = g * βT * (
        T.data[:, :, 1:Nz-1] * 0.5 * (w.data[:, :, 1:Nz-1] + w.data[:, :, 2:Nz]))

    @views @. wb.data[:, :, Nz] = g * βT * (
        T.data[:, :, Nz] * 0.5 * w.data[:, :, Nz]) # w[Nz+1] = 0 due to no-penetration

    return wb
end

function cfl(Δt, model)
    umax = maximum(abs.(model.velocities.u.data))
    vmax = maximum(abs.(model.velocities.v.data))
    wmax = maximum(abs.(model.velocities.w.data))

    Δmin = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)

    return Δt * max(umax, vmax, wmax) / Δmin
end

total_kinetic_energy(model) = total_kinetic_energy(model.velocities...)

total_kinetic_energy(u, v, w) = ΔV(u.grid) / 2 * (
    sum(u.data.^2) + sum(v.data.^2) + sum(w.data.^2))

function relative_error(ϕ_num, ϕ_ans)
    Field = fieldkind(ϕ_num)
    ϕ_ans = Field(ϕ_ans.(nodes(ϕ_num)...))
    return mean((ϕ_num.data .- ϕ_ans.data).^2) / mean(ϕ_ans.data.^2)
end

"""
    make_vertical_slice_movie(model::Model, nc_writer::NetCDFOutputWriter,
                              var_name, Nt, Δt, var_offset=0, slice_idx=1)

Make a movie of a vertical slice produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀). A `slice_idx` can be specified to select
the index of the y-slice to be plotted (useful when plotting vertical slices
from a 3D model, it should be set to 1 for 2D xz-slice models).
"""
function make_vertical_slice_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0, slice_idx=1)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.contour(model.grid.xC, reverse(model.grid.zC), rotl90(var[:, slice_idx, :] .- var_offset),
                      fill=true, levels=9, linewidth=0, color=:balance,
                      clims=(-0.011, 0.011), title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
        # Plots.heatmap(model.grid.xC, model.grid.zC, rotl90(var[:, slice_idx, :]) .- var_offset,
        #               color=:balance, clims=(-0.01, 0.01), title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

"""
    make_horizontal_slice_movie(model::Model, nc_writer::NetCDFOutputWriter,
                                var_name, Nt, Δt, var_offset=0)

Make a movie of a horizontal slice produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀).
"""
function make_horizontal_slice_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.heatmap(model.grid.xC, model.grid.yC, var[:, :, 1] .- var_offset,
                      color=:balance, clims=(-0.01, 0.01),
                      title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

"""
    make_vertical_profile_movie(model::Model, nc_writer::NetCDFOutputWriter,
                                var_name, Nt, Δt, var_offset=0)

Make a movie of a vertical profile produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀).
"""
function make_vertical_profile_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.plot(var[1, 1, :] .- var_offset, model.grid.zC,
                   title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

function make_avg_temperature_profile_movie()
    Nt, dt = 86400, 0.5
    freq = 3600
    N_frames = Int(Nt/freq)
    filename_prefix = "convection"
    var_offset = 273.15

    Nz, Lz = 128, 100
    dz = Lz/Nz
    zC = -dz/2:-dz:-Lz

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")

        filepath = filename_prefix * lpad(freq*n, 9, "0") * ".nc"
        field_data = ncread(filepath, "T")
        ncclose(filepath)

        T_profile = mean(field_data; dims=[1,2])

        Plots.plot(reshape(T_profile, Nz) .- var_offset, zC,
                   title="t=$(freq*n*dt) s ($(round(freq*n*dt/86400; digits=2)) days)")
    end

    mp4(animation, filename_prefix * "$(round(Int, time())).mp4", fps=30)
end
