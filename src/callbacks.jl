module Callbacks

import PyCall: PyObject, pycall

import ..WJFKeras
import ..WJFKeras: PyDoc

const WJFKeras_callbacks = [
    "BaseLogger",
    "ProgbarLogger",
    "History",
    "ModelCheckpoint",
    "EarlyStopping",
    "RemoteMonitor",
    "LearningRateScheduler",
    "TensorBoard",
    "ReduceLROnPlateau",
    "CSVLogger",
    "LambdaCallback",
]

for c in WJFKeras_callbacks
    cb_name = Symbol(c)

    @eval begin
        struct $cb_name
            obj::PyObject

            @doc PyDoc(WJFKeras._callbacks, Symbol($c)) function $cb_name(args...; kwargs...)
                new(WJFKeras._callbacks[Symbol($c)](args...; kwargs...))
            end
        end

        PyObject(callback::$(cb_name)) = callback.obj
    end
end

end
