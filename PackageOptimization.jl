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
lower_bounds = [10000.0, -9.0, 2100.0, 15.0]  # Temperature can be below freezing
upper_bounds = [16500.0, 31.0, 2400.0, 23.0]

# Initial guess
initial_guess = [14000.0, 15.0, 2200.0, 20.0]

# Arrays to store results
results = []

# Iterate over all combinations of w_power, w_airspeed, and w_fuel_consumption that sum to 1
step = 0.1
for w_power in 0:step:1
    for w_airspeed in 0:step:1
        w_fuel_consumption = 1 - w_power - w_airspeed
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

            # Store the results
            push!(results, (w_power, w_airspeed, w_fuel_consumption, -optimal_power, -optimal_airspeed, optimal_fuel_consumption, result.minimum))
        end
    end
end

# Create a DataFrame for the results
results_df = DataFrame(results, [:w_power, :w_airspeed, :w_fuel_consumption, :optimal_power, :optimal_airspeed, :optimal_fuel_consumption, :objective_value])

# Print the results
println(results_df)