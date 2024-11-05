import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("mip_example")

# Add variables
x = model.addVar(vtype=GRB.INTEGER, name="x")  # x is an integer
y = model.addVar(vtype=GRB.CONTINUOUS, name="y")  # y is continuous

# Set the objective function
model.setObjective(3 * x + 5 * y, GRB.MAXIMIZE)

# Add constraints
model.addConstr(x + 2 * y <= 10, "c1")
model.addConstr(3 * x + 4 * y <= 18, "c2")
model.addConstr(x >= 0, "c3")
model.addConstr(y >= 0, "c4")

# Optimize the model
model.optimize()

# Display the results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found: x = {x.x}, y = {y.x}")
    print(f"Objective value: {model.objVal}")
else:
    print("No optimal solution found.")
