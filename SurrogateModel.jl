using CSV
using DataFrames
using Downloads
using LinearAlgebra
using Statistics

"""
    multilinear_basis(x1, x2, x3, x4)
    Defines a function that returns the 16 multilinear basis function of 4 input variables
"""
function multilinear_basis(x1, x2, x3, x4)
    # Create a 1x16 matrix to store the basis functions
    basis = zeros(Float64, 1, 16)
    
    # Calculate each basis function
    basis[1, 1]  = 1
    basis[1, 2]  = x1
    basis[1, 3]  = x2
    basis[1, 4]  = x3
    basis[1, 5]  = x4
    basis[1, 6]  = x1 * x2
    basis[1, 7]  = x1 * x3
    basis[1, 8]  = x1 * x4
    basis[1, 9]  = x2 * x3
    basis[1, 10] = x2 * x4
    basis[1, 11] = x3 * x4
    basis[1, 12] = x1 * x2 * x3
    basis[1, 13] = x1 * x2 * x4
    basis[1, 14] = x1 * x3 * x4
    basis[1, 15] = x2 * x3 * x4
    basis[1, 16] = x1 * x2 * x3 * x4

    return basis
end

"""
    z_scores_relative_to_rest(data)
    Defines a function that returns the mean, standard deviation, and z scores of data
"""
function z_scores_relative_to_rest(data::Vector{Float64})
    n = length(data)
    μ = mean(data)
    σ = std(data)
    z_scores = Vector{Float64}(undef, n)
    
    for i in 1:n
        # Calculate the z-score of the current data point
        z_scores[i] = (data[i] - μ) / σ
    end
    
    return z_scores, μ, σ
end

#Download the CSV file
csv_file_path = Downloads.download("https://raw.githubusercontent.com/ctilneyv/AA222_Final_Project/main/data/O-470-U_Performance_Data_Processed.csv")

# Read the CSV file into a DataFrame
engineData = CSV.read(csv_file_path, DataFrame)
engineData = Matrix(engineData)

#constructs input variable vectors
x1 = engineData[: , 1]
x2 = engineData[: , 2]
x3 = engineData[: , 3]
x4 = engineData[: , 4]

#To construct our output variables, we need to standandardize our output data to have a fair comparison. We do this based on z-score
#constructs output variable vectors
y1, power_mean, power_std = z_scores_relative_to_rest(engineData[: , 5])
y2, TAS_mean, TAS_std = z_scores_relative_to_rest(engineData[: , 6])
y3, FuelConsumption_mean, FuelConsumption_std = z_scores_relative_to_rest(engineData[: , 7])

#Writes the B matrix using equation 14.16
B = zeros(Float64, size(engineData, 1), 16)
for i in 1:size(engineData, 1)
    B[i, :] = multilinear_basis(x1[i], x2[i], x3[i], x4[i])
end

#Takes the pseudoinverse, then uses equation 14.17 to find θ_i
BpInv = pinv(B)

# Compute Θ1, Θ2, and Θ3
Θ1 = BpInv * y1
Θ2 = BpInv * y2
Θ3 = BpInv * y3



println("Θ1 = ", Θ1)
println("power mean = ", power_mean, ", power std = ", power_std)
println("Θ2 = ", Θ2)
println("TAS mean = ", TAS_mean, ", TAS std = ", TAS_std)
println("Θ3 = ", Θ3)
println("Fuel Consumption mean = ", FuelConsumption_mean, ", Fuel Consumption std = ", FuelConsumption_std)
println()


#Create data frame with standardized output variables
df = DataFrame(
    PressureAltitude = x1,
    Temperature = x2,
    PropellerPitch = x3,
    ManifoldPressure = x4,
    Power = y1,
    Airspeed = y2,
    FuelConsumption = y3
)

"""
Additional code for debugging and veryfing Θ 
"""
#=
println(engineData[201, 1:4])
zscore = (multilinear_basis(x1[201], x2[201], x3[201], x4[201]) * Θ2)[1]
speed = TAS_mean + (zscore * TAS_std)
println(speed)
=#