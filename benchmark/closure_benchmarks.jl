using TimerOutputs, Printf
using Oceananigans

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Axes of parameter variation
            Ns = [(32, 32, 32)]
      Closures = (ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity, ConstantSmagorinsky)
   float_types = [Float32, Float64]     # Float types to benchmark.
         archs = [CPU()]                # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]         # Benchmark GPU on systems with CUDA-enabled GPUs.

arch_name(::String) = ""
arch_name(::CPU) = "CPU"
arch_name(::GPU) = "GPU"

benchmark_name(N, id) = benchmark_name(N, id, "", "", "")

function benchmark_name(N, id, arch, FT, closure; npad=3)
    Nx, Ny, Nz = N

    bn = ""
    bn *= lpad(Nx, npad, " ") * "x" * lpad(Ny, npad, " ") * "x" * lpad(Nz, npad, " ")
    bn *= " $id"

    arch = arch_name(arch)
    bn *= " ($arch, $FT, $(Symbol(closure)))"

    return bn
end

for arch in archs, float_type in float_types, N in Ns, Closure in Closures
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 100, 100, 100

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type,
                        closure=Closure(float_type))
    time_step!(model, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "static ocean", arch, float_type, Closure)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Oceananigans.jl benchmarks")

#=
bid = "static ocean"  # Benchmark ID. We only have one right now.

println("\n\nCPU Float64 -> Float32 speedup:")
for N in Ns
    for Closure in Closures
        bn32 = benchmark_name(N, bid, CPU(), Float32, Closure)
        bn64 = benchmark_name(N, bid, CPU(), Float64, Closure)
        t32  = TimerOutputs.time(timer[bn32])
        t64  = TimerOutputs.time(timer[bn64])
        @printf("%s: %.3f\n", benchmark_name(N, bid), t64/t32)
    end
end
=#
