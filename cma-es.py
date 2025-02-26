import cma
import numpy as np

# Import the solver function
from clutch import electro_thermal_simulation  # Ensure your solver is saved as fenics_solver.py

def objective_function(params):
    """
    Objective function for optimizing 3D ICs.
    Args:
        params: Array of parameters to optimize (e.g., resistivity, thermal conductivity).
    Returns:
        Objective value (e.g., average temperature or max power density).
    """
    # Unpack parameters
    rho_0, alpha, k_0 = params
    
    # Run the solver with current parameters
    results = electro_thermal_simulation(
        rho_0=rho_0,
        alpha=alpha,
        k_0=k_0,
        nx=20, ny=20, nz=20,  # Adjust for quick evaluation
        Lx=1.0, Ly=1.0, Lz=1.0,  # Dimensions
        V_boundary=10.0,  # Boundary voltage
        T_boundary=300.0,  # Boundary temperature
    )
    
    # Extract results
    T = results['temperature']
    P = results['power_density']
    
    # Compute metrics (e.g., average temperature or max power density)
    avg_temperature = T.vector().get_local().mean()
    max_power_density = P.vector().get_local().max()
    
    # Combine metrics into a single objective
    # Example: Minimize average temperature and max power density
    objective = avg_temperature + 0.1 * max_power_density
    
    return objective

# Set up initial guess and bounds for parameters
initial_guess = [1.0, 0.004, 1.0]  # Initial values for rho_0, alpha, k_0
bounds = [
    [0.5, 2.0],   # Bounds for rho_0
    [0.001, 0.01],  # Bounds for alpha
    [0.1, 5.0],   # Bounds for k_0
]
optimized_params = []

for bound in bounds:
    es = cma.CMAEvolutionStrategy(initial_guess, 0.5, {'bounds': bound})
    print("Starting CMA-ES optimization...")
    while not es.stop():
        # Ask for new parameter set from CMA-ES
        solutions = es.ask()
        
        # Evaluate the objective for each solution
        fitness_values = [objective_function(sol) for sol in solutions]
        
        # Update CMA-ES with evaluated fitness values
        es.tell(solutions, fitness_values)
        
        # Print progress
        es.logger.add()
        es.disp()

        # Retrieve optimized parameters
        local_optimized_params = es.result.xbest
        print("Optimization complete!")
        print("Optimized parameters:", local_optimized_params)
        optimized_params.append(local_optimized_params)

final_result = objective_function(optimized_params)
print("Final objective value:", final_result)

# Save the optimized parameters to a file
with open("optimized_params.txt", "w") as f:
    f.write(f"Optimized parameters: {optimized_params}\n")
    f.write(f"Final objective value: {final_result}\n")
