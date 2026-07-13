function quadratic_problem(; T=Float64)
    nstate, ninput = 3, 2
    x = T[0.7, -0.4]
    y = T[-0.2, 0.5, 0.9]
    W = T[0.2 -0.1; 0.4 0.3; -0.5 0.2]
    b = T[0.1, -0.2, 0.05]
    parameters = [vec(W); b]

    unpack(ps) = (reshape(view(ps, 1:(nstate * ninput)), nstate, ninput),
                  view(ps, (nstate * ninput + 1):length(ps)))
    drive(ps, input) = begin
        weights, bias = unpack(ps)
        weights * input + bias
    end

    model = EPModel(
        energy=(s, ps, input, st) -> sum(abs2, s) / 2 - dot(s, drive(ps, input)),
        cost=(s, target, ps, st) -> sum(abs2, s .- target) / 2,
        readout=(s, ps, st) -> s,
        initial_state=(ps, input, st) -> zeros(T, nstate),
    )
    problem = EPProblem(model, parameters, NamedTuple(), x, y)
    return problem, drive(parameters, x), y
end

function quadratic_algorithm(protocol)
    return EPAlgorithm(
        protocol,
        Relaxation(dt=0.5, maxiters=200, abstol=1e-11);
        state_ad=AutoForwardDiff(),
        parameter_ad=AutoForwardDiff(),
    )
end
