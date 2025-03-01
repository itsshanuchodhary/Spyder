import cma
from clutch import electro_thermal_simulation 

def objective_function(params):
    """
    Objective function for optimizing maximum temperature (T_max) in 3D ICs.
    Args:
        params: List of parameters [k_TIM, h, thickness_TIM].
    Returns:
        T_max: Maximum temperature computed by the solver.
    """
    k_TIM, h, thickness_TIM = params  # Unpack parameters

    results = electro_thermal_simulation(
        k_0=k_TIM,  # Thermal conductivity of TIM
        h=h,        # Heat transfer coefficient
        TIM_thickness=thickness_TIM,  # Pass TIM thickness to solver
        nx=40, ny=40, nz=40,          # Use a moderate mesh for testing
        max_iterations=20             # Ensure convergence
    )

    # Extract maximum temperature (T_max) from the solver
    T = results['temperature']
    T_max = T.vector().get_local().max()

    return T_max

# CMA-ES Optimization Setup
initial_guess = [5.0, 10.0, 0.02]  # Initial guess: [k_TIM, h, thickness_TIM]
bounds = [[1.0, 0.1, 0.005], [20.0, 50.0, 0.1]]  # Bounds for parameters

# Run CMA-ES
es = cma.CMAEvolutionStrategy(initial_guess, 0.5, {'bounds': bounds})

print("Starting CMA-ES optimization...")
while not es.stop():
    # Ask for a set of candidate solutions
    solutions = es.ask()
    print("Current solutions  -  ", solutions)

    # Evaluate the objective function for each solution
    fitness_values = [objective_function(sol) for sol in solutions]

    # Update CMA-ES with fitness values
    es.tell(solutions, fitness_values)

    # Display progress
    es.logger.add()
    es.disp()

# Retrieve optimized parameters
optimized_params = es.result.xbest
print("Optimization complete!")
print("Optimized parameters:", optimized_params)

# Run the solver one last time with optimized parameters
final_results = electro_thermal_simulation(
    k_0=optimized_params[0],
    h=optimized_params[1],
    TIM_thickness=optimized_params[2]
)
final_T = final_results['temperature'].vector().get_local().max()
print("Final maximum temperature:", final_T)
