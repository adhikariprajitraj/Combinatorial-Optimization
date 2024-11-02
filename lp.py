# importing the pyomo module, we will be using gurobi as the solver
import pyomo.environ as pyomo
import matplotlib.pyplot as plt
import numpy as np

# calling the instance of the ConcreteModel
model = pyomo.ConcreteModel()

# defining the variables, constraints and the objective function
model.x1 = pyomo.Var(within=pyomo.NonNegativeReals)
model.x2 = pyomo.Var(within=pyomo.NonNegativeReals)

# constraints here
model.c = pyomo.ConstraintList()
model.c.add(model.x1 + model.x2 <= 10)  # Total resource constraint
model.c.add(2*model.x1 + model.x2 <= 16)  # Production capacity constraint
model.c.add(model.x1 >= 2)  # Minimum production requirement


# objective function of the model
model.obj = pyomo.Objective(
    rule=lambda model: model.x1 + model.x2*10,
    sense=pyomo.maximize
)

# calling the solver
solver = pyomo.SolverFactory('gurobi')

# solving the model
result = solver.solve(model)

print(result)
# print(model.x1(), model.x2())
# print(model.obj())

print(
    f"Objective function value: {model.obj()} "
    f"with x1 = {model.x1()} and x2 = {model.x2()}."
)

# Create points for plotting
x1_points = np.linspace(0, 10, 100)

# Plot constraints
plt.plot(x1_points, 10 - x1_points, label='x1 + x2 ≤ 10')
plt.plot(x1_points, 16 - 2*x1_points, label='2x1 + x2 ≤ 16')
plt.axvline(x=2, color='r', linestyle='--', label='x1 ≥ 2')
plt.axhline(y=0, color='k', linestyle='-')

# Plot optimal solution point
plt.plot(model.x1(), model.x2(), 'ro', label='Optimal Solution')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend()
plt.show()
