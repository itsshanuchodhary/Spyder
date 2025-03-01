import numpy as np
from clutch import electro_thermal_simulation  # Import your solver
import GPyOpt  # Bayesian Optimization library

# Define the objective function
def objective_function(params):
    """
    Objective function for Bayesian Optimization.
    Args:
        params: Array of shape (1, n) where n is the number of parameters.
                Example: [[k_0, h, TIM_thickness]]
    Returns:
        T_max: Maximum temperature computed by the solver.
    """
    # Extract parameters from input
    k_0, h, TIM_thickness = params[0]  # Single parameter set

    # Run the solver
    results = electro_thermal_simulation(
        k_0=k_0,
        h=h,
        TIM_thickness=TIM_thickness,
        nx=40, ny=40, nz=40  # Set a reasonable mesh resolution
    )

    # Extract maximum temperature
    T = results['temperature']
    T_max = T.vector().get_local().max()

    print(f"Parameters: k_0={k_0}, h={h}, TIM_thickness={TIM_thickness} -> T_max={T_max}")
    return T_max

# Define parameter bounds
bounds = [
    {'name': 'k_0', 'type': 'continuous', 'domain': (1.0, 20.0)},  # Thermal conductivity
    {'name': 'h', 'type': 'continuous', 'domain': (5.0, 50.0)},    # Heat transfer coefficient
    {'name': 'TIM_thickness', 'type': 'continuous', 'domain': (0.005, 0.1)}  # TIM thickness
]

# Create Bayesian Optimization model
bo = GPyOpt.methods.BayesianOptimization(
    f=objective_function,  # Objective function
    domain=bounds,         # Parameter bounds
    acquisition_type='EI',  # Expected Improvement acquisition function
    exact_feval=True,       # Function is deterministic
    initial_design_numdata=5  # Number of initial points
)

# Run the optimization
print("Starting Bayesian Optimization...")
bo.run_optimization(max_iter=20)  # Maximum number of iterations

# Print optimized parameters
print("Optimization complete!")
print("Optimized parameters:", bo.x_opt)
print("Minimum T_max achieved:", bo.fx_opt)

# Save optimization results
bo.plot_convergence()
