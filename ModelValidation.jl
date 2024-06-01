using CSV
using Downloads

#Download the Processed Dakota CSV
csv_file_path = Downloads.download("https://raw.githubusercontent.com/ctilneyv/AA222_Final_Project/main/O-470-U_Performance%20Data_Processed.csv")

# Read the CSV file into a DataFrame
engineData = CSV.read(csv_file_path, DataFrame)
engineData = Matrix(engineData)

#constructs input variable vectors
x1 = engineData[: , 1]
x2 = engineData[: , 2]
x3 = engineData[: , 3]
x4 = engineData[: , 4]







"""
Computes error of the surrogate model over all combinations of input variables
"""
#=

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