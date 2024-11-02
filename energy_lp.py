import pyomo.environ as pyomo
import matplotlib.pyplot as plt

price_schedule = {
    0: 0.5, 1: 0.7, 2: 0.9, 3: 1.1, 4: 1.3, 5: 1.5,
    6: 1.3, 7: 1.1, 8: 0.9, 9: 0.7, 10: 0.5
}

charge_schedule = {
    0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9, 4: 1.1, 5: 1.3,
    6: 1.5, 7: 1.3, 8: 1.1, 9: 0.9, 10: 0.7
}

model = pyomo.ConcreteModel()


# number of time periods
model.nt = pyomo.Param(initialize=len(price_schedule), domain=pyomo.Integers)

# set of time periods
model.T = pyomo.Set(initialize=range(model.nt()), ordered=True)

# sales price (rename from p to price)
model.price = pyomo.Param(model.T, initialize=price_schedule)

# charging price
model.c = pyomo.Param(model.T, initialize=charge_schedule)

# battery capacity
model.B = pyomo.Param(initialize=40, domain=pyomo.NonNegativeReals)

# battery efficiency
model.eta = pyomo.Param(initialize=0.95, domain=pyomo.NonNegativeReals)

# initial battery level
model.b0 = pyomo.Param(initialize=10, domain=pyomo.NonNegativeReals)

# power level (keep as p)
model.p = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# battery level
model.b = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# objective function (update to use price instead of p)
model.obj = pyomo.Objective(expr=sum(model.price[t] * model.p[t] for t in model.T), sense=pyomo.maximize)


# Create a ConstraintList to hold constraints
model.constraints = pyomo.ConstraintList()

# constraints
model.constraints.add(model.b[0] == model.b0)
model.constraints.add(model.b[model.nt()-1] == 0)

# Add battery dynamics constraints
for t in model.T:
    if t < model.nt()-1:
        # Battery level in next period equals current level minus power output
        model.constraints.add(model.b[t+1] == model.b[t] - model.p[t])
    # Battery level must not exceed capacity
    model.constraints.add(model.b[t] <= model.B) 
    # Power output cannot exceed current battery level
    model.constraints.add(model.p[t] <= model.b[t])

# solver
solver = pyomo.SolverFactory('gurobi')
results = solver.solve(model, keepfiles=True, logfile='energy_lp.log')

# results
print(pyomo.value(model.obj))
for t in model.T:
    print(f"t = {t}, p = {pyomo.value(model.p[t]):.2f}, b = {pyomo.value(model.b[t]):.2f}")

# plot
plt.plot([pyomo.value(model.p[t]) for t in model.T], label='Power')
plt.plot([pyomo.value(model.b[t]) for t in model.T], label='Battery')
plt.legend()
plt.show()

