module Constraints

import PyCall: PyObject, pycall

import ..WJFKeras
import ..WJFKeras: PyDoc

const WJFKeras_constraints = ["max_norm", "non_neg", "unit_norm"]

for c in WJFKeras_constraints
    const_name = Symbol(c)

    @eval begin
        @doc PyDoc(WJFKeras._constraints, Symbol($c)) function $const_name(args...; kwargs...)
            WJFKeras._constraints[Symbol($c)](args...; kwargs...)
        end
    end
end

end
