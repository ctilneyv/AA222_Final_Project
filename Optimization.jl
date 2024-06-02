using CSV
using DataFrames
using Downloads
using LinearAlgebra
using Random
using Optim

# polynomial basis functions
function polynomial_basis(x)
    x1, x2, x3, x4 = x
    return [1, x1, x2, x3, x4, x1 * x2, x1 * x3, x1 * x4, x2 * x3, x2 * x4, x3 * x4, x1 * x2 * x3, x1 * x2 * x4, x1 * x3 * x4, x2 * x3 * x4, x1 * x2 * x3 * x4]
end

# evaluation of surrogate model
function evaluate_surrogate_model(x, θ)
    basis = polynomial_basis(x)
    return dot(basis, θ)
end

csv_file_path = Downloads.download("https://raw.githubusercontent.com/ctilneyv/AA222_Final_Project/main/O-470-U_Performance_Data_Processed.csv")
engineData = CSV.read(csv_file_path, DataFrame)
engineData = Matrix(engineData)

# constructs input/ouput variable vectors
x1 = engineData[:, 1]       # pressure altitude
x2 = engineData[:, 2]       # temperature
x3 = engineData[:, 3]       # propeller pitch
x4 = engineData[:, 4]       # manifold pressure
y1 = engineData[:, 5]       # power
y2 = engineData[:, 6]       # airspeed
y3 = engineData[:, 7]       # fuel consumption

# writes the B matrix using equation 14.16
B = zeros(Float64, size(engineData, 1), 16)
for i in 1:size(engineData, 1)
    B[i, :] = polynomial_basis([x1[i], x2[i], x3[i], x4[i]])
end

# takes the pseudoinverse, then uses equation 14.17 to find θ_i
BpInv = pinv(B)

Θ1 = BpInv * y1
Θ2 = BpInv * y2
Θ3 = BpInv * y3

function evaluate_surrogate_model_all(x)
    return [evaluate_surrogate_model(x, θ) for θ in (Θ1, Θ2, Θ3)]
end

# constraint functions
function manifold_pressure_constraint(x)
    x1, x2, ~, x4 = x
    P = 29.92 * exp(-931.9 * x1 / (89494.6 * (x2 + 273)))
    return x4 - P
end

function overspeed_constraint(x)
    ~, ~, x3, x4 = x
    return 100 * x4 - x3
end

# constraint violations penalty
function constraint_penalty(x)
    penalty = 0.0

    # check then add penalties
    penalty += max(0, manifold_pressure_constraint(x)^2)
    penalty += max(0, overspeed_constraint(x)^2)
    return penalty
end

# base lower and upper bounds for design variables
lb_base = [0.0, -10.0, 2100.0, 15.0]
ub_base = [16500.0, 32.0, 2400.0, 23.0]

# set the weights
w_power = 0.3
w_airspeed = 0.3
w_fc = 0.4

results = []

# discretization
steps = [8, 8, 8, 8]
for i in 0:steps[1]
    for j in 0:steps[2]
        for k in 0:steps[3]
            for l in 0:steps[4]
                lb = [lb_base[1] + i * (ub_base[1] - lb_base[1]) / steps[1],
                    lb_base[2] + j * (ub_base[2] - lb_base[2]) / steps[2],
                    lb_base[3] + k * (ub_base[3] - lb_base[3]) / steps[3],
                    lb_base[4] + l * (ub_base[4] - lb_base[4]) / steps[4]]

                ub = [lb_base[1] + (i + 1) * (ub_base[1] - lb_base[1]) / steps[1],
                    lb_base[2] + (j + 1) * (ub_base[2] - lb_base[2]) / steps[2],
                    lb_base[3] + (k + 1) * (ub_base[3] - lb_base[3]) / steps[3],
                    lb_base[4] + (l + 1) * (ub_base[4] - lb_base[4]) / steps[4]]

                x0 = [(lb[m] + ub[m]) / 2 for m in eachindex(lb)]

                function objective_function(x)
                    power, airspeed, fc = evaluate_surrogate_model_all(x)
                    objective_value = -(w_power * power + w_airspeed * airspeed + -w_fc * fc)
                    return objective_value
                end

                # modified objective function with penalties
                function penalty_function(x)
                    return objective_function(x) + 1e6 * constraint_penalty(x)
                end

                # optimization and get optimal solution
                result = optimize(penalty_function, lb, ub, x0, Fminbox(LBFGS()))
                optimal_x = result.minimizer

                # evaluate the surrogate model @ optimum
                optimal_power, optimal_airspeed, optimal_fc = evaluate_surrogate_model_all(optimal_x)

                # store results w/ design variables
                push!(results, (optimal_x..., -optimal_power, -optimal_airspeed, optimal_fc, result.minimum))
            end
        end
    end
end

results_df = DataFrame(results, [:pressure_altitude, :temperature, :propeller_pitch, :manifold_pressure, :optimal_power, :optimal_airspeed, :optimal_fc, :objective_value])
CSV.write("computation/results.csv", results_df)

function is_dominated(x, y)
    return all(x .<= y) && any(x .< y)
end

function find_pareto(df)
    pareto = []
    for i in 1:size(df, 1)
        dominated = false
        for j in 1:size(df, 1)
            if i != j && is_dominated(-[df.optimal_airspeed[j], -df.optimal_fc[j]], -[df.optimal_airspeed[i], -df.optimal_fc[i]])
                dominated = true
                break
            end
        end
        if !dominated
            push!(pareto, df[i, :])
        end
    end
    return DataFrame(pareto)
end

pareto_df = find_pareto(results_df)
CSV.write("computation/pareto_optima.csv", pareto_df)

#println("Pareto Optimal Points:")
#println(pareto_df)

function pareto_optimum(pareto_df, w_airspeed, w_power, w_fc)
    optimum = pareto_df[1, :]
    max_score = w_airspeed * optimum.optimal_airspeed + w_power * optimum.optimal_power - w_fc * optimum.optimal_fc

    for point in eachrow(pareto_df)
        score = w_airspeed * point.optimal_airspeed + w_power * point.optimal_power - w_fc * point.optimal_fc
        if score > max_score
            optimum = point
            max_score = score
        end
    end

    return optimum
end

w_power = (1-0.91)/2
w_airspeed = (1-0.91)/2
w_fc = 0.91

optimum = pareto_optimum(pareto_df, w_airspeed, w_power, w_fc)

CSV.write("computation/pareto_optimum.csv", DataFrame(optimum))

println("Weights:")
println("  Power                            $w_power")
println("  Airspeed                         $w_airspeed")
println("  Fuel Consumption                 $w_fc")

println("Design Variables:")
println("  Pressure Altitude (ft)           $(optimum.pressure_altitude)")
println("  Temperature (C)                  $(optimum.temperature)")
println("  Propeller Pitch (RPM)            $(optimum.propeller_pitch)")
println("  Manifold Pressure (inHg)         $(optimum.manifold_pressure)")

println("Outputs:")
println("  Optimal Power (%BHP)             $(optimum.optimal_power)")
println("  Optimal Airspeed (kts)           $(optimum.optimal_airspeed)")
println("  Optimal Fuel Cons. (gal/hr)      $(optimum.optimal_fc)")

function extrapolate_aero(file_path::String, target_altitude::Float64, target_airspeed::Float64)

    data = CSV.read(file_path, DataFrame)
    closest_altitude = data.Altitude[argmin(abs.(data.Altitude .- target_altitude))]

    data_altitude_filtered = filter(row -> row.Altitude == closest_altitude, data)
    closest_airspeed = data_altitude_filtered."Airspeed V(mph)"[argmin(abs.(data_altitude_filtered."Airspeed V(mph)" .- target_airspeed))]

    result = filter(row -> row.Altitude == closest_altitude && row."Airspeed V(mph)" == closest_airspeed, data)
    return result
end

file_path = "O-470-U_Aerodynamic_Data_Processed.csv"
aero_df = extrapolate_aero(file_path, optimum.pressure_altitude, optimum.optimal_airspeed)

println("Extrapolated Aerodynamic Data:")
println(aero_df)