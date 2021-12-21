module Optimizers

import PyCall: PyObject, pycall

import ..WJFKeras
import ..WJFKeras: PyDoc

const WJFKeras_optimizers = [
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adam",
    "Adamax",
    "Nadam",
]

for o in WJFKeras_optimizers
    opt_name = Symbol(o)

    @eval begin
        struct $opt_name
            obj::PyObject

            @doc PyDoc(WJFKeras._optimizers, Symbol($o)) function $opt_name(args...; kwargs...)
                new(WJFKeras._optimizers[Symbol($o)](args...; kwargs...))
            end
        end

        # convert(::Type{$(opt_name)}, obj::PyObject) = $opt_name(obj)
        PyObject(opt::$(opt_name)) = opt.obj
    end
end

end
