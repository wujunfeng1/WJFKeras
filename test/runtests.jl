using Random
using Statistics
using Test
using PyCall
using WJFKeras
using StatsBase

import WJFKeras: metric
import WJFKeras.Layers: Dense, Activation, LSTM

mse(actual, pred) = Statistics.mean(square(actual - pred))
mae(actual, pred) = Statistics.mean(abs(actual - pred))
rmse(actual, pred) = sqrt(mse(actual, pred))

@testset "WJFKeras" begin
    @testset "Basic Usage" begin
        #model = Sequential()
        #add!(model, Dense(80, input_dim=30))
        #add!(model, Activation(:relu))
        #add!(model, Dense(10))
        #add!(model, Activation(:softmax))
        model = Sequential([
            Dense(80, input_dim=30),
            Activation(:relu),
            Dense(10),
            Activation(:softmax),
        ])
        compile!(
            model;
            loss=:categorical_crossentropy,
            optimizer=:sgd,
            metrics=[:accuracy]
        )

        h = fit!(model, rand(100, 30), rand(100, 10); epochs=20, batch_size=10, verbose=0)

        @test haskey(h.history, "accuracy")
        @test haskey(h.history, "loss")

        evaluate(model, rand(10, 30), rand(10, 10); batch_size=5, verbose=0)
        predict(model, rand(10, 30); batch_size=5, verbose=0)
    end

    if VERSION >= v"0.7.0"
        @testset "Custom Objectives & Metrics" begin
            model = Sequential()
            add!(model, Dense(20, input_dim=30))
            add!(model, Activation(:relu))
            add!(model, Dense(10))
            add!(model, Activation(:softmax))

            compile!(
                model;
                loss=metric(mse),
                optimizer=:sgd,
                metrics=[:accuracy, metric(mae), metric(rmse)]
            )

            h = fit!(model, rand(100, 30), rand(100, 10); epochs=10, batch_size=10, verbose=10)

            @test haskey(h.history, "accuracy")
            @test haskey(h.history, "loss")
            @test haskey(h.history, "mae")
            @test haskey(h.history, "rmse")

            evaluate(model, rand(10, 30), rand(10, 10); batch_size=5, verbose=10)
            predict(model, rand(10, 30); batch_size=5, verbose=0)
        end
    end

    @testset "Tensor Operations" begin
        @testset "Single Tensor Operations" begin
            x = rand(Float32, 4, 3)
            x_t = variable(x)

            @testset "Testing $op" for op in [-, transpose, sqrt, exp, log, round, sin, cos]
                expected, result = if op in [sqrt, exp, log, round, sin, cos]
                    op_result = WJFKeras.eval(op(x_t))
                    bc_result = WJFKeras.eval(broadcast(op, x_t))
                    @test op_result == bc_result

                    broadcast(op, x), bc_result
                else
                    op(x), WJFKeras.eval(op(x_t))
                end

                @test size(expected) == size(result)
                @test op === transpose || typeof(expected) == typeof(result)
                match = all(map(isapprox, expected, result))
                if !match
                    println(expected)
                    println(result)
                end
                @test match
            end

            @testset "Testing $op" for op in [maximum, minimum, sum, prod, Statistics.var, Statistics.std, Statistics.mean]
                expected = Float32(op(x))
                result = Float32(WJFKeras.eval(op(x_t))[1])

                @test isapprox(expected, result; atol=0.1)
            end

            x = rand(Bool, 4)
            x_t = variable(x)
            @testset "Testing $op" for op in [all, any]
                expected = op(x)
                result = Bool(WJFKeras.eval(op(x_t))[1])
                @test expected == result
            end

            expected = broadcast(~, x)
            result = map(i -> i[1], (~Tensor(x)).o)
            @test expected == result
        end

        @testset "Two Tensor Operations" begin
            a = rand(Float32, 4, 3)
            a_t = variable(a)

            b = rand(Float32, 4, 3)
            b_t = variable(b)

            @testset "$op" for op in [==, !=, >, <, >=, <=]
                expected = broadcast(op, a, b)
                result = BitArray(WJFKeras.eval(broadcast(op, a_t, b_t)))
                @test expected == result
            end

            @testset "Testing $op" for op in [mod, ^, -, +, *, /]
                expected = broadcast(op, a, b)
                result = WJFKeras.eval(broadcast(op, a_t, b_t))
                @test size(expected) == size(result)
                @test typeof(expected) == typeof(result)
                @test all(map(isapprox, expected, result))
            end

            b = rand(Float32, 3, 4)
            b_t = variable(b)

            @testset "Testing $op" for op in [*]
                expected = op(a, b)
                result = WJFKeras.eval(op(a_t, b_t))

                @test size(expected) == size(result)
                @test typeof(expected) == typeof(result)
                @test all(map(isapprox, expected, result))
            end

            a = bitrand(4)
            b = bitrand(4)
            expected = broadcast(&, a, b)
            result = map(i -> i[1], (Tensor(a) & Tensor(b)).o)
            @test expected == result

            expected = broadcast(|, a, b)
            result = map(i -> i[1], (Tensor(a) | Tensor(b)).o)
            @test expected == result
        end

        @testset "Custom Tensor Operations" begin
            x = Float32[-1.1, 0.8, -0.5, 1.2]
            x_t = variable(x)

            @test WJFKeras.eval(clip(x_t, -1.0, 1.0)) == Float32[-1.0, 0.8, -0.5, 1.0]
            square(x_t)
            WJFKeras.cast(x_t, :float64)
        end
    end

    @testset "Layers" begin
        @testset "Basic Usage" begin
            a = WJFKeras._layers.Input(shape=(140, 256))

            lstm = LSTM(32)
            encoded_a = lstm(a)

            W = weights(lstm)
            weights!(lstm, W)
            config(lstm)
            _input(lstm)
            output(lstm)
            input_shape(lstm)
            output_shape(lstm)
        end
    end

    @testset "Regularizers" begin
        @testset "Regularizer classes: $rc" for rc in WJFKeras.Regularizers.WJFKeras_regularizer_classes
            q = getfield(WJFKeras.Regularizers, Symbol(rc))()
            @test isa(q.obj, PyCall.PyObject)
        end

        @testset "Creating $reg regularizer" for reg in WJFKeras.Regularizers.WJFKeras_regularizer_aliases
            q = getfield(WJFKeras.Regularizers, Symbol(reg))()
            @test isa(q, PyCall.PyObject)
        end
    end

    @testset "PyDoc" begin
        output = sprint(show, WJFKeras.PyDoc(WJFKeras._models, :Sequential))
        @test startswith(output, "`Sequential` groups a linear stack of layers into a `tf.keras.Model`.")
        text = Base.Docs.catdoc(WJFKeras.PyDoc(WJFKeras._models, :Sequential), WJFKeras.PyDoc(WJFKeras._layers, :Dense))
        output = string(text)
        @test startswith(output, "`Sequential` groups a linear stack of layers into a `tf.keras.Model`.")
        @test endswith(strip(output), "the output would have shape `(batch_size, units)`.")
    end
end
