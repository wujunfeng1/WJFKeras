module Initializations

import PyCall: PyObject, pycall

import ..WJFKeras
import ..WJFKeras: PyDoc

const WJFKeras_initializer_obj = [
    "Zeros",
    "Ones",
    "Constant",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Orthogonal",
    "Identity",

]

for i in WJFKeras_initializer_obj
    init_name = Symbol(i)

    @eval begin
        struct $init_name
            obj::PyObject

            @doc PyDoc(WJFKeras._initializers, Symbol($i)) function $init_name(args...; kwargs...)
                new(WJFKeras._initializers[Symbol($i)](args...; kwargs...))
            end
        end

        PyObject(initializer::$(init_name)) = initializer.obj
    end
end

const WJFKeras_initializer_funcs = [
    "lecun_uniform",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
]

for i in WJFKeras_initializer_funcs
    init_name = Symbol(i)

    @eval begin
        @doc PyDoc(WJFKeras._initializers, Symbol($i)) function $init_name(args...; kwargs...)
            WJFKeras._initializers[Symbol($i)](args...; kwargs...)
        end
    end
end

end
