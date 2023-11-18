module ExtendedDistributions

using Random
using Reexport
using StatsFuns
@reexport using Distributions

const _EPSILON = 1f-6

include("univariates.jl")

export
    ArctanhNormal,
    Metalogistic,
    LogitMetalogistic,
    feasible4,
    feasible3,
    feasible2,
    isfeasible,
    analytical_entropy,
    no_analytical_entropy,
    analytical_kl,
    no_analytical_kl

end
