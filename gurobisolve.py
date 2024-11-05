from gurobipy import Model, GRB

# Step 1: Initialize Model
model = Model("Production Optimization")

# Step 2: Define Decision Variables
# Variables x and y represent the units of Product A and B respectively
x = model.addVar(name="Product_A", vtype=GRB.CONTINUOUS, lb=0)
y = model.addVar(name="Product_B", vtype=GRB.CONTINUOUS, lb=0)

# Step 3: Set the Objective Function
# Objective: Maximize profit = 20x + 30y
model.setObjective(20 * x + 30 * y, GRB.MAXIMIZE)

# Step 4: Add Constraints
# 1. Labor hours constraint: 2x + y <= 100
model.addConstr(2 * x + y <= 100, name="Labor_Constraint")

# 2. Material usage constraint: x + y <= 80
model.addConstr(x + y <= 80, name="Material_Constraint")

# 3. Demand constraint for Product A: x <= 40
model.addConstr(x <= 40, name="Demand_Constraint")

# Step 5: Solve the Model
model.optimize()

# Step 6: Output Results
# Check if the model has an optimal solution
if model.status == GRB.OPTIMAL:
    print(f"Optimal Solution: Profit = ${model.objVal:.2f}")
    print(f"Units of Product A: {x.x:.2f}")
    print(f"Units of Product B: {y.x:.2f}")
else:
    print("No optimal solution found.")
