using CSV
using Downloads
using DataFrames
using LinearAlgebra


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
csv_file_path = Downloads.download("https://raw.githubusercontent.com/ctilneyv/AA222_Final_Project/main/data/O-540_processed.csv")

# Read the CSV file into a DataFrame
dakotaEngineData = CSV.read(csv_file_path, DataFrame; delim=',', missingstring="")
dakotaEngineData = Matrix(dakotaEngineData)

#constructs input variable vectors
alt_Dakota = dakotaEngineData[: , 1]
temp_Dakota = dakotaEngineData[: , 2]
rpm_Dakota = dakotaEngineData[: , 3]
throttle_Dakota = dakotaEngineData[: , 4]

#Parameters from SurrogateModel.jl
Θ1 = [9.17577430306679, -4.7557885696992465e-5, 0.04923128339434957, 0.0006365033476316167, 0.05462754869500769, -6.995017600983073e-6, -1.678348924646099e-8, -2.3834937329715204e-6, -2.3793074101433534e-5, -0.002995670845174034, -0.0002575796577085033, 3.099824463990813e-9, 3.981634063220685e-7, 1.3817268948998618e-9, 1.697513224454838e-6, -1.7188657548907494e-10]
power_mean = -60.42718446601942; power_std = 8.456844544180367

Θ2 = [31.958421927160316, -0.0001228735647452611, 0.13993272914939492, -0.010116980134956391, -1.0360574410180097, 2.2866289140954825e-6, 1.2745411372779054e-7, -5.938216112207404e-6, -5.61376198391156e-5, -0.0060228171757997166, 0.00028311460277943834, -5.411087306814735e-10, -7.724207173442258e-8, -6.419598039021787e-9, 2.2832467634002086e-6, 1.545166054144279e-11]
TAS_mean = -128.42071197411002; TAS_std = 8.153420810924608

Θ3 = [3.836497173735875, -0.0005779111284220919, -0.025767569746562358, -0.0067173808713525684, -0.6908967491652546, 4.198929028449591e-6, 3.26388297970366e-7, 2.960431269890797e-5, 2.001802699139081e-5, 0.0020846142626538312, 0.0005550464246643209, -2.3029250176558857e-9, -2.2863025208945416e-7, -1.506634077393683e-8, -1.6286734140756168e-6, 1.186263819359782e-10]
fuelConsumption_mean = 10.385436893203883; fuelConsumption_std = 1.358838868847817

powerZscorePredicted_Dakota = zeros(length(alt_Dakota))
TASZscorePredicted_Dakota = zeros(length(alt_Dakota))
FuelConsumptionZscorePredicted_Dakota = zeros(length(alt_Dakota))

for i in eachindex(alt_Dakota)
    basis = multilinear_basis(alt_Dakota[i], temp_Dakota[i], rpm_Dakota[i], throttle_Dakota[i])
    powerZscorePredicted_Dakota[i] = (basis * Θ1)[1];
    TASZscorePredicted_Dakota[i] = (basis * Θ2)[1];
    FuelConsumptionZscorePredicted_Dakota[i] = (basis * Θ3)[1];
end

#Backs out predicted values from the model-prediced z scores
power_predicted = power_mean .+ (powerZscorePredicted_Dakota .* power_std)
TAS_predicted = TAS_mean .+ (TASZscorePredicted_Dakota .* TAS_std)
fuelConsumption_predicted = fuelConsumption_mean .+ (FuelConsumptionZscorePredicted_Dakota .* fuelConsumption_std)

#constructs observed output variable vectors
power_Dakota_observed = dakotaEngineData[: , 5]
airspeed_Dakota_observed = dakotaEngineData[: , 6]
fuelConsumption_Dakota_observed = dakotaEngineData[: , 7]

#Computes error of the surrogate model over all combinations of input variables
power_error = zeros(length(power_Dakota_observed)); airspeed_error = zeros(length(airspeed_Dakota_observed)); fuelConsumption_error = zeros(length(fuelConsumption_Dakota_observed));

for i in 1:length(power_Dakota_observed)
    power_error[i] = abs((power_predicted[i] - power_Dakota_observed[i]) /  power_Dakota_observed[i])
    airspeed_error[i] = abs((TAS_predicted[i] - airspeed_Dakota_observed[i]) /  airspeed_Dakota_observed[i])
    fuelConsumption_error[i] = abs((fuelConsumption_predicted[i] - fuelConsumption_Dakota_observed[i]) /  fuelConsumption_Dakota_observed[i])
end


#=
println("Power Error = ", power_error)
println()
println("Airspeed Error = ", airspeed_error)
println()
println("Fuel Consumption Error = ", fuelConsumption_error)
println()
=#