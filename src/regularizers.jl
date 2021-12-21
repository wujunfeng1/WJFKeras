module Regularizers

import PyCall: PyObject, pycall, PyAny

import ..WJFKeras
import ..WJFKeras: PyDoc

const WJFKeras_regularizer_classes = [ "Regularizer", "L1L2"]
const WJFKeras_regularizer_aliases = [
    "l1",
    "l2",
    "l1_l2",
]

for r in WJFKeras_regularizer_classes
    reg_name = Symbol(r)

    @eval begin
        struct $reg_name
            obj::PyObject

            @doc PyDoc(WJFKeras._regularizers, Symbol($r)) function $reg_name(args...; kwargs...)
                new(WJFKeras._regularizers.$r(args...; kwargs...))
            end
        end

        # convert(::Type{$(reg_name)}, obj::PyObject) = $reg_name(obj)
        PyObject(reg::$(reg_name)) = reg.obj
        pycall(reg::$(reg_name), args...; kws...) = pycall(reg.obj, args...; kws...)
    end
end

for r in WJFKeras_regularizer_aliases
    rf = Symbol(r)

    @eval begin
        @doc PyDoc(WJFKeras._regularizers, Symbol($r)) function $rf(args...; kwargs...)
            return pycall(WJFKeras._regularizers.$r, PyAny, args...; kwargs...)
        end
    end
end

end
