import pyomo.environ as pyomo

model = pyomo.ConcreteModel()

model.x1 = pyomo.Var(within=pyomo.NonNegativeReals)
model.x2 = pyomo.Var(within=pyomo.NonNegativeReals)

model.c = pyomo.ConstraintList()
model.c.add(model.x1*10 + 1 >= model.x2)
model.c.add(model.x1*0.2 + 4 >= model.x2)
model.c.add(mode.x1*(-0.2)+6 >= model.x2)

model.obj = pyomo.Objective(expr=model.x1 + model.x2*10, sense=pyomo.minimize)

