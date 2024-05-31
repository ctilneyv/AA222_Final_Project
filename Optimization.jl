using CSV
using DataFrames
using Downloads
using LinearAlgebra
using Random
using Optim

# polynomial basis functions
function polynomial_basis(x)
    x1, x2, x3, x4 = x
    return [1, x1, x2, x3, x4, x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4, x1*x2*x3, x1*x2*x4, x1*x3*x4, x2*x3*x4, x1*x2*x3*x4]
end

# evaluation of surrogate model
function evaluate_surrogate_model(x, θ)
    basis = polynomial_basis(x)
    return dot(basis, θ)
end

csv_file_path = Downloads.download("https://raw.githubusercontent.com/ctilneyv/AA222_Final_Project/main/O-470-U_Performance%20Data_Processed.csv")

# read the CSV file into a DataFrame
engineData = CSV.read(csv_file_path, DataFrame)
engineData = Matrix(engineData)

# constructs input variable vectors
x1 = engineData[:, 1]
x2 = engineData[:, 2]
x3 = engineData[:, 3]
x4 = engineData[:, 4]

# constructs output variable vectors
y1 = engineData[:, 5]
y2 = engineData[:, 6]
y3 = engineData[:, 7]

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
    power = evaluate_surrogate_model(x, Θ1)
    airspeed = evaluate_surrogate_model(x, Θ2)
    fuel_consumption = evaluate_surrogate_model(x, Θ3)
    return power, airspeed, fuel_consumption
end

# constraint functions
function manifold_pressure_constraint(x)
    x1, x2, x3, x4 = x
    P = 29.92 * exp(-931.9 * x1 / (89494.6 * (x2 + 273)))
    return x4 - P
end

function overspeed_constraint(x)
    x1, x2, x3, x4 = x
    return 100 * x4 - x3
end

# constraint violations penalty
function constraint_penalty(x)
    penalty = 0.0

    # check constraints then add penalties
    penalty += max(0, manifold_pressure_constraint(x))^2
    penalty += max(0, overspeed_constraint(x))^2

    return penalty
end

# upper and lower bounds for x1, x2, x3, and x4
lb = [00000.0, -9.0, 2100.0, 15.0]
ub = [16500.0, 31.0, 2400.0, 23.0]

x0 = [12000.0, 15.0, 2200.0, 20.0]

# array to store results
results = []

# check all combinations of weights while preserving weight summation
step = 0.01
for w_fc in 0:step:1
    for w_power in 0:step:(1 - w_fc)
        w_airspeed = 1 - w_power - w_fc
        if w_fc >= 0
            # Define the objective function
            function objective_function(x)
                power, airspeed, fuel_consumption = evaluate_surrogate_model_all(x)
                objective_value = -(w_power * power + w_airspeed * airspeed + w_fc * fuel_consumption)
                return objective_value
            end

            # Modified objective function with penalties
            function penalized_objective_function(x)
                return objective_function(x) + 1e3 * constraint_penalty(x)  # Large penalty factor
            end

            # optimization and get optimal solution
            result = optimize(penalized_objective_function, lb, ub, x0, Fminbox())
            optimal_x = result.minimizer

            # evaluate the surrogate model @ optimum
            optimal_power, optimal_airspeed, optimal_fuel_consumption = evaluate_surrogate_model_all(optimal_x)

            # store results w/ design variables
            push!(results, (w_power, w_airspeed, w_fc, optimal_x..., -optimal_power, -optimal_airspeed, optimal_fuel_consumption, result.minimum))
        end
    end
end

# DataFrame for results
results_df = DataFrame(results, [:w_power, :w_airspeed, :w_fc, :x1, :x2, :x3, :x4, :optimal_power, :optimal_airspeed, :optimal_fuel_consumption, :objective_value])

# find index with the smallest objective value
min_index_with_design_vars = argmin(results_df.objective_value)

# row with the smallest objective value
min_result_with_design_vars = results_df[min_index_with_design_vars, :]

println("Weights:")
println("  w_power: $(min_result_with_design_vars.w_power)")
println("  w_airspeed: $(min_result_with_design_vars.w_airspeed)")
println("  w_fc: $(min_result_with_design_vars.w_fc)")

println("Design Variables:")
println("  Pressure Altitude (ft): $(min_result_with_design_vars.x1)")
println("  Temperature (C): $(min_result_with_design_vars.x2)")
println("  Propeller Pitch (RPM): $(min_result_with_design_vars.x3)")
println("  Manifold Pressure (inHg): $(min_result_with_design_vars.x4)")

println("Outputs:")
println("  Optimal Power (%BHP): $(min_result_with_design_vars.optimal_power)")
println("  Optimal Airspeed (kts): $(min_result_with_design_vars.optimal_airspeed)")
println("  Optimal Fuel Consumption (gal/hr): $(min_result_with_design_vars.optimal_fuel_consumption)")
println("  Objective Value: $(min_result_with_design_vars.objective_value)")

using CSV

# Filter the results DataFrame to include only rows with equal w_airspeed and w_power
results_filtered = filter(row -> row.w_airspeed ≈ row.w_power, results_df)

# Extract the relevant columns
objective_values = results_filtered.objective_value
w_fc_values = results_filtered.w_fc

# Create a DataFrame for the extracted data
data_df = DataFrame(w_fc = w_fc_values, objective_value = objective_values)

# Define the path to save the CSV file
csv_output_path = "ObjectiveValues.csv"

# Write the DataFrame to a CSV file
CSV.write(csv_output_path, data_df)

println("CSV file saved successfully at $csv_output_path.")