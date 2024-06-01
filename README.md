# AA222_Final_Project

## Code overview
## /main
- `SurrogateModel.jl` is a Julia file that computes the weighting parameters for a surrogate model defineid by the 16 multilinear basis functions
- `ModelValidation.jl` is a Julia file that computes the weighting parameters for a surrogate model defineid by the 16 multilinear basis functions
- `Optimize.jl` is a Julia file that performs all the optimization provided the surrogate model is computed


## /main/data
- `O-470-U_Parameters.csv` is a csv file containing the aircraft input parameters, output parameters, and reference data (in case required after optimization)
- `O-470-U_Performance_Data_Processed.csv` is a csv file containing real-world processed data of pressure altitude (ft), temperature (C), propeller pitch (rpm), manifold pressure (inHG), power (%bHP), airspeed (KTAS), and fuel Consumption (gal/hr)
- `O-470-U_Aerodynamic_Data_Processed.csv` is a csv file containing real-world processed data of altitude, airspeed V(mph), rate-of-climb RC (fpm), sink rate RS (fpm), Reynolds No, THP (hp), thrust (lb), drag (lb), L/D, SEP (ft/sec), and Mach No.

