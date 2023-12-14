module ExtendedDistributions

using Random
using Reexport
@reexport using StatsFuns
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
    no_analytical_kl,

    # logpdf
    kumaraswamylogpdf,
    atanhnormlogpdf,
    laplacelogpdf,
    logitnormlogpdf,
    logisticlogpdf,

    # kldivergence
    atanhnormkldivergence,
    betakldivergence,
    laplacekldivergence,
    normkldivergence,
    logitnormkldivergence,

    # cdf and ccdf
    kumaraswamycdf,
    kumaraswamyccdf,
    logitnormcdf,
    logitnormccdf,
    laplacecdf,
    laplaceccdf,
    atanhnormcdf,
    atanhnormccdf,
    logisticcdf,
    logisticccdf,

    # quantile
    kumaraswamyquantile,
    logitnormquantile,
    laplacequantile,
    atanhnormquantile,
    normquantile,
    logisticquantile
end
