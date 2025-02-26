from fenics import *
import numpy as np
import matplotlib.pyplot as plt

parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
parameters["form_compiler"]["cache_dir"] = "clear"

def electro_thermal_simulation(
    nx=40, ny=40, nz=40,  # Mesh resolution
    Lx=1.0, Ly=1.0, Lz=1.0,  # Domain dimensions
    rho_0=1.0,  # Base resistivity
    alpha=0.004,  # Temperature coefficient of resistivity
    T_0=300.0,  # Reference temperature
    k_0=1.0,  # Base thermal conductivity
    max_iterations=20,  # Maximum iterations for nonlinear coupling
    tolerance=1e-6,  # Convergence tolerance
    V_boundary=10.0,  # Voltage at one boundary
    T_boundary=300.0,  # Temperature at boundaries
):
    """
    Solves coupled electro-thermal problem with the following equations:
    1. div((1/resistivity) * grad(V)) = 0
    2. div(thermal_conductivity * grad(T)) = -P
    3. P = J * E = (1/resistivity) * grad(V) * grad(V)
    4. resistivity = rho_0 * (1 + alpha * (T - T_0))
    """
    # Create mesh
    mesh = BoxMesh(Point(0, 0, 0), Point(Lx, Ly, Lz), nx, ny, nz)

    # Define function spaces (P1 elements for both voltage and temperature)
    V_space = FunctionSpace(mesh, 'P', 1)
    T_space = FunctionSpace(mesh, 'P', 1)

    # Define trial and test functions
    v = TrialFunction(V_space)
    w = TestFunction(V_space)
    t = TrialFunction(T_space)
    s = TestFunction(T_space)

    # Define functions for solutions
    V = Function(V_space)  # Voltage
    T = Function(T_space)  # Temperature
    T_prev = Function(T_space)  # Previous temperature iteration

    # Initialize temperature to reference temperature
    T_prev.interpolate(Constant(T_0))
    
    # Define boundary conditions
    def voltage_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, 1e-14)
    
    def ground_boundary(x, on_boundary):
        return on_boundary and near(x[0], Lx, 1e-14)
    
    def temperature_boundary(x, on_boundary):
        return on_boundary
    
    # Apply Dirichlet boundary conditions
    bc_V_high = DirichletBC(V_space, Constant(V_boundary), voltage_boundary)
    bc_V_ground = DirichletBC(V_space, Constant(0.0), ground_boundary)
    bc_T = DirichletBC(T_space, Constant(T_boundary), temperature_boundary)
    
    bcs_V = [bc_V_high, bc_V_ground]
    bcs_T = [bc_T]
    
    # Define nonlinear iteration loop
    iteration = 0
    error = 1.0
    
    # Arrays to store convergence data
    errors = []
    
    print("Starting nonlinear iterations...")
    while iteration < max_iterations and error > tolerance:
        iteration += 1
        
        # Calculate resistivity based on current temperature
        resistivity_expr = rho_0 * (1.0 + alpha * (T_prev - T_0))
        
        # 1. Solve electrical problem: div((1/resistivity) * grad(V)) = 0
        a_V = (1.0 / resistivity_expr) * dot(grad(v), grad(w)) * dx
        L_V = Constant(0.0) * w * dx
        
        solve(a_V == L_V, V, bcs_V)
        
        # 2. Calculate power density P = J * E = (1/resistivity) * |grad(V)|^2
        V_grad = grad(V)
        power_density = (1.0 / resistivity_expr) * dot(V_grad, V_grad)
        
        # 3. Solve thermal problem: div(k * grad(T)) = -P
        a_T = k_0 * dot(grad(t), grad(s)) * dx
        L_T = -power_density * s * dx
        
        solve(a_T == L_T, T, bcs_T)
        
        # Calculate error between iterations
        T_diff = T.copy(deepcopy=True)
        T_diff.vector().axpy(-1.0, T_prev.vector())  # T_diff = T - T_prev
        error = norm(T_diff.vector()) / norm(T.vector())
        errors.append(error)
        
        print(f"Iteration {iteration}: error = {error:.6e}")
        
        # Update previous temperature
        T_prev.assign(T)
        
        # Break if converged
        if error < tolerance:
            print(f"Converged after {iteration} iterations!")
            break
    
    if iteration == max_iterations and error > tolerance:
        print(f"Warning: Did not converge after {max_iterations} iterations. Final error: {error:.6e}")
    
    # Calculate important derived quantities
    # Current density vector J = (1/resistivity) * grad(V)
    J = project((1.0 / resistivity_expr) * grad(V), VectorFunctionSpace(mesh, 'P', 1))
    
    # Electric field E = -grad(V)
    E = project(-grad(V), VectorFunctionSpace(mesh, 'P', 1))
    
    # Joule heating power density
    P = project(power_density, V_space)
    
    # Return results
    return {
        'mesh': mesh,
        'voltage': V,
        'temperature': T,
        'current_density': J,
        'electric_field': E,
        'power_density': P,
        'convergence': errors
    }

# Example usage and visualization
if __name__ == "__main__":
    # Run simulation
    results = electro_thermal_simulation()
    
    # Extract results
    V = results['voltage']
    T = results['temperature']
    J = results['current_density']
    P = results['power_density']
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot convergence
    plt.subplot(2, 2, 1)
    plt.semilogy(results['convergence'])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence')
    plt.grid(True)
    
    # Save results to file
    File("results/voltage.pvd") << V
    File("results/temperature.pvd") << T
    File("results/current_density.pvd") << J
    File("results/power_density.pvd") << P
    
    print("Simulation completed and results saved.")
