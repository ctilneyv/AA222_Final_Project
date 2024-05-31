using CSV
using DataFrames
using Downloads
using LinearAlgebra
using Random
using Optim

# Polynomial basis functions
function polynomial_basis(x)
    x1, x2, x3, x4 = x
    return [1, x1, x2, x3, x4, x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4, x1*x2*x3, x1*x2*x4, x1*x3*x4, x2*x3*x4, x1*x2*x3*x4]
end

# Evaluate surrogate model
function evaluate_surrogate_model(x, θ)
    basis = polynomial_basis(x)
    return dot(basis, θ)
end

# Download the CSV file
csv_file_path = Downloads.download("https://raw.githubusercontent.com/ctilneyv/AA222_Final_Project/main/O-470-U_Performance%20Data_Processed.csv")

# Read the CSV file into a DataFrame
engineData = CSV.read(csv_file_path, DataFrame)
engineData = Matrix(engineData)

# Constructs input variable vectors
x1 = engineData[:, 1]
x2 = engineData[:, 2]
x3 = engineData[:, 3]
x4 = engineData[:, 4]

# Constructs output variable vectors
y1 = engineData[:, 5]
y2 = engineData[:, 6]
y3 = engineData[:, 7]

# Writes the B matrix using equation 14.16
B = zeros(Float64, size(engineData, 1), 16)
for i in 1:size(engineData, 1)
    B[i, :] = polynomial_basis([x1[i], x2[i], x3[i], x4[i]])
end

# Takes the pseudoinverse, then uses equation 14.17 to find θ_i
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

# Define the constraint functions
function manifold_pressure_constraint(x)
    x1, x2, x3, x4 = x
    P = 29.92 * exp(-931.9 * x1 / (89494.6 * (x2 + 273)))
    return x4 - P
end

function overspeed_constraint(x)
    x1, x2, x3, x4 = x
    return 100 * x4 - x3
end

# Penalty for constraint violations
function constraint_penalty(x)
    penalty = 0.0

    # Check the constraints and add penalties
    penalty += max(0, manifold_pressure_constraint(x))^2
    penalty += max(0, overspeed_constraint(x))^2

    return penalty
end

# Define the bounds for x1, x2, x3, and x4
lower_bounds = [00000.0, -9.0, 2100.0, 15.0]  # Temperature can be below freezing
upper_bounds = [16500.0, 31.0, 2400.0, 23.0]

# Initial guess
initial_guess = [14000.0, 15.0, 2200.0, 20.0]

# Arrays to store results including design variables
results_with_design_vars = []

# Iterate over all combinations of w_power, w_airspeed, and w_fuel_consumption that sum to 1
step = 0.01
for w_fuel_consumption in 0:step:1
    for w_power in 0:step:(1 - w_fuel_consumption)
        w_airspeed = 1 - w_power - w_fuel_consumption
        if w_fuel_consumption >= 0
            # Define the objective function
            function objective_function(x)
                power, airspeed, fuel_consumption = evaluate_surrogate_model_all(x)
                objective_value = -(w_power * power + w_airspeed * airspeed + w_fuel_consumption * fuel_consumption)
                return objective_value
            end

            # Modified objective function with penalties
            function penalized_objective_function(x)
                return objective_function(x) + 1e3 * constraint_penalty(x)  # Large penalty factor
            end

            # Perform the optimization
            result = optimize(penalized_objective_function, lower_bounds, upper_bounds, initial_guess, Fminbox())

            # Get the optimal solution
            optimal_x = result.minimizer

            # Evaluate the surrogate model at the optimal solution
            optimal_power, optimal_airspeed, optimal_fuel_consumption = evaluate_surrogate_model_all(optimal_x)

            # Store the results including design variables
            push!(results_with_design_vars, (w_power, w_airspeed, w_fuel_consumption, optimal_x..., -optimal_power, -optimal_airspeed, optimal_fuel_consumption, result.minimum))
        end
    end
end

# Create a DataFrame for the results including design variables
results_with_design_vars_df = DataFrame(results_with_design_vars, [:w_power, :w_airspeed, :w_fuel_consumption, :x1, :x2, :x3, :x4, :optimal_power, :optimal_airspeed, :optimal_fuel_consumption, :objective_value])

# Find the index of the row with the smallest objective value
min_index_with_design_vars = argmin(results_with_design_vars_df.objective_value)

# Retrieve the row with the smallest objective value including design variables
min_result_with_design_vars = results_with_design_vars_df[min_index_with_design_vars, :]

# Print the respective weights, design variables, and outputs for the minimized objective value
println("Weights:")
println("  w_power: $(min_result_with_design_vars.w_power)")
println("  w_airspeed: $(min_result_with_design_vars.w_airspeed)")
println("  w_fuel_consumption: $(min_result_with_design_vars.w_fuel_consumption)")

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
filtered_results = filter(row -> row.w_airspeed ≈ row.w_power, results_with_design_vars_df)

# Extract the relevant columns
objective_values = filtered_results.objective_value
w_fuel_consumption_values = filtered_results.w_fuel_consumption

# Create a DataFrame for the extracted data
data_df = DataFrame(w_fuel_consumption = w_fuel_consumption_values, objective_value = objective_values)

# Define the path to save the CSV file
csv_output_path = "ObjectiveValues.csv"

# Write the DataFrame to a CSV file
CSV.write(csv_output_path, data_df)

println("CSV file saved successfully at $csv_output_path.")