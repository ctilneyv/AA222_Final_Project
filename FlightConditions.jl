using CSV
using DataFrames
using LinearAlgebra

function find_closest_aerodynamic_data(file_path::String, target_altitude::Float64, target_airspeed::Float64)
    # Reads CSV
    data = CSV.read(file_path, DataFrame)
    
    closest_altitude = data.Altitude[argmin(abs.(data.Altitude .- target_altitude))]
    data_altitude_filtered = filter(row -> row.Altitude == closest_altitude, data)
    
    closest_airspeed = data_altitude_filtered."Airspeed V(mph)"[argmin(abs.(data_altitude_filtered."Airspeed V(mph)" .- target_airspeed))]
    
    result = filter(row -> row.Altitude == closest_altitude && row."Airspeed V(mph)" == closest_airspeed, data)
    
    return result
end

# Example usage
file_path = "O-470-U_Aerodynamic_Data_Processed.csv"

# Make sure that the target alt and speed are in floats or it breaks
target_altitude = 10000.0
target_airspeed = 62.0 
result = find_closest_aerodynamic_data(file_path, target_altitude, target_airspeed)
println(result)
