using CSV
using DataFrames
using Downloads
using LinearAlgebra
using Random

# Polynomial basis functions
# Basically the same as Coopers multilinear_basis function but different
function polynomial_basis(x)
    x1, x2, x3, x4 = x
    return [1, x1, x2, x3, x4, x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4, x1*x2*x3, x1*x2*x4, x1*x3*x4, x2*x3*x4, x1*x2*x3*x4]
end

# Evaluate surrogate model
function evaluate_surrogate_model(x, θ)
    basis = polynomial_basis(x)
    return dot(basis, θ)
end

# Download the CSV fileUs
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

function objective_function(x)
    power, airspeed, fuel_consumption = evaluate_surrogate_model_all(x)

    # we can tune these later
    w_power = 0.3
    w_airspeed = 0.3
    w_fuel_consumption = -0.4  # Negative because we want to minimize this

    objective_value = -(w_power * power + w_airspeed * airspeed + w_fuel_consumption * fuel_consumption)

    return objective_value
end

# I chose tournament selection bc its prolly the best as to not go to local min
function select(S, fitness)
    selected_parents = []
    for _ in 1:S
        candidates = rand(1:length(fitness), 2)
        push!(selected_parents, (candidates[1], candidates[2]))
    end
    return selected_parents
end

# Combines the chomes of parents to form the children 
# This is single point cross over since its the easiest to implement
function crossover(C, parent1, parent2)
    crossover_point = rand(1:length(parent1)-1)
    child = vcat(parent1[1:crossover_point], parent2[crossover_point+1:end])
    print("SoyBoy")
    return child
end

# Added constraints here but should really have a constraint function
function mutate(M, child)
    for i in 1:length(child)
        if rand() < M
            if i == 1  # Pressure Altitude
                child[i] = clamp(child[i] + 100 * randn(), 0.0, 16500.0)  # Valid range: 0 to 16500 ft
            elseif i == 2  # Temperature
                child[i] = clamp(child[i] + randn(), -50.0, 50.0)  # Valid range: -50 to 50 °C
            elseif i == 3  # Propeller Pitch
                child[i] = clamp(child[i] + 10 * randn(), 2100.0, 2400.0)  # Valid range: 2100 to 2400 RPM
            elseif i == 4  # Manifold Pressure
                child[i] = clamp(child[i] + randn(), 15.0, 23.0)  # Valid range: 15 to 23 inHG
            end
        end
    end
    return child
end

# Genetic Algorithm 
function genetic_algorithm(f, population, k_max, S, C, M)
    for k in 1:k_max
        parents = select(S, f.(population))
        children = [crossover(C, population[p[1]], population[p[2]]) for p in parents]
        children = [mutate(M, child) for child in children]
        population = vcat(population, children)
        population = population[1:end]
    end
    best_index = argmin(f.(population))
    return population[best_index]
end

# Example usage
population_size = 50
num_generations = 100
num_parents = 25
crossover_rate = 0.8
mutation_rate = 0.02
num_variables = 4

# Initial population with constraints
function generate_initial_population(size, num_variables)
    population = []
    for _ in 1:size
        individual = [
            rand(0.0:16500.0),       # Pressure Altitude (ft)
            rand(-50.0:50.0),        # Temperature (°C)
            rand(2100.0:2400.0),     # Propeller Pitch (RPM)
            rand(15.0:23.0)          # Manifold Pressure (inHG)
        ]
        push!(population, individual)
    end
    return population
end

initial_population = generate_initial_population(population_size, num_variables)

best_solution = genetic_algorithm(objective_function, initial_population, num_generations, num_parents, crossover_rate, mutation_rate)

println("Best solution found: ", best_solution)
println("Best objective value: ", objective_function(best_solution))