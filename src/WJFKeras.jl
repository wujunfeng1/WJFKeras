module WJFKeras

using Compat
using PyCall

using Compat: @__MODULE__
import PyCall: PyObject, @pydef

# We store our renamed aliases to aid with
# automatic docstring updating.
global const WJFKeras_func_aliases = Dict(
    "compile" => "compile!",
    "add" => "add!"
)

"""
Copied from LazyHelp in PyPlot.jl.

define a documentation object that lazily looks up
help from a PyObject via zero or more keys.
This saves us time when loading WJFKeras, since we don't have
to load up all of the documentation strings right away.
"""
struct PyDoc
    obj::PyObject
    names::Tuple{Vararg{Symbol}}

    PyDoc(obj::PyObject, name::Symbol) = new(obj, (name,))
end

function Base.show(io::IO, ::MIME"text/plain", doc::PyDoc)
    obj = doc.obj

    for name in doc.names
        obj = obj[name]
    end

    if haskey(obj, "__doc__")
        print(io, convert(AbstractString, obj.__doc__))
    else
        print(io, "no Python docstring found for PyDoc($(doc.obj), $(doc.names))")
    end
end

Base.show(io::IO, doc::PyDoc) = show(io, "text/plain", doc)

function Base.Docs.catdoc(docs::PyDoc...)
    Base.Docs.Text() do io
        for doc in docs
            show(io, MIME"text/plain"(), doc)
        end
    end
end

# We need to handle our python dependencies carefully here
global const _WJFKeras = PyNULL()
global const _backend = PyNULL()
global const _layers = PyNULL()
global const _models = PyNULL()
global const _regularizers = PyNULL()
global const _optimizers = PyNULL()
global const _callbacks = PyNULL()
global const _constraints = PyNULL()
global const _initializers = PyNULL()

function __init__()
    copy!(_WJFKeras, pyimport("tensorflow.keras"))
    copy!(_backend, pyimport("tensorflow.keras.backend"))
    copy!(_layers, pyimport("tensorflow.keras.layers"))
    copy!(_models, pyimport("tensorflow.keras.models"))
    copy!(_regularizers, pyimport("tensorflow.keras.regularizers"))
    copy!(_optimizers, pyimport("tensorflow.keras.optimizers"))
    copy!(_callbacks, pyimport("tensorflow.keras.callbacks"))
    copy!(_constraints, pyimport("tensorflow.keras.constraints"))
    copy!(_initializers, pyimport("tensorflow.keras.initializers"))

    include(joinpath(@__DIR__, "utils.jl"))
end

include("tensors.jl")
include("callbacks.jl")
include("constraints.jl")
include("initializers.jl")
include("optimizers.jl")
include("regularizers.jl")
include("layers.jl")
include("models.jl")

end
