using CSV
using Downloads
using DataFrames

#include("SurrogateModel.jl")

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
Computes error of the surrogate model over all combinations of input variables
"""

# Download the Processed Dakota CSV
csv_file_path = Downloads.download("https://github.com/ctilneyv/AA222_Final_Project/blob/838fdbef838b13d8ac7ba04b86209c0b89cabfeb/Engine%20Data/O-540_processed.csv")

# Read the CSV file into a DataFrame
dakotaEngineData = CSV.read(csv_file_path, DataFrame; delim=',', missingstring="")

# Display the first few rows
println(first(dakotaEngineData, 5))


#=

dakotaEngineData = Matrix(dakotaEngineData)

#constructs input variable vectors
alt_Dakota = dakotaEngineData[: , 1]
temp_Dakota = dakotaEngineData[: , 2]
rpm_Dakota = dakotaEngineData[: , 3]
throttle_Dakota = dakotaEngineData[: , 4]




#Parameters from SurrogateModel.jl
Θ1 = [8.898986775460411, 0, 0.06571015453379125, 0.0007893114438193372, 0.07154677259694037, 0, 0, 0, 0, -0.0038127411502383753, -0.0002665566701794286, 0, 0, 0, 0, 0]
power_mean = -60.42718446601942; power_std = 8.456844544180367

Θ2 = [32.74236828456448, -0.00016950542916782533, 0.14352790235186688, -0.010440053681092897, -1.0705546130899912, 0, 0, 0, 0, -0.006191496854121049, 0.0002973569794650911, 0, 0, 0, 0, 0]
TAS_mean = -128.42071197411002; TAS_std = 8.153420810924608

Θ3 = [4.257429713062449, -0.0006264112637819055, -0.042189866958947775, -0.006937351175611412, -0.7147645560302736, 0, 0, 0, 0, 0.0029040018935038627, 0.0005672703192249791, 0, 0, 0, 0, 0]
FuelConsumption_mean = 10.385436893203883; FuelConsumption_std = 1.358838868847817

for i in 1:eachindex(alt_Dakota)
    powerZscorePredicted_Dakota = multilinear_basis(alt_Dakota[i], temp_Dakota[i], rpm_Dakota[i], throttle_Dakota[i]) * Θ1
    TASZscorePredicted_Dakota = multilinear_basis(alt_Dakota[i], temp_Dakota[i], rpm_Dakota[i], throttle_Dakota[i]) * Θ2
    FuelConsumptionZscorePredicted_Dakota = multilinear_basis(alt_Dakota[i], temp_Dakota[i], rpm_Dakota[i], throttle_Dakota[i]) * Θ3
end


power_predicted = power_mean .+ (powerZscorePredicted_Dakota .* power_std)
println(power_predicted)



"""
Computes error of the surrogate model over all combinations of input variables
"""


y1_error = zeros(length(y1)); y2_error = zeros(length(y2)); y3_error = zeros(length(y3));

for i in 1:length(y1)
   
    y1_predicted = multilinear_basis(x1[i], x2[i], x3[i], x4[i]) * Θ1
    y2_predicted = multilinear_basis(x1[i], x2[i], x3[i], x4[i]) * Θ2
    y3_predicted = multilinear_basis(x1[i], x2[i], x3[i], x4[i]) * Θ3


    y1_error[i] = abs((y1_predicted[1] - y1[i]) /  y1[i])
    y2_error[i] = abs((y2_predicted[1] - y2[i]) /  y2[i])
    y3_error[i] = abs((y3_predicted[1] - y3[i]) /  y3[i])
end

println(y3_error)

=#