import numpy as np

from poli import objective_factory

# Using create
_, f, x0, y0, _ = objective_factory.create(name="ddr3_docking")
print(x0)
print(y0)
f.terminate()

# Using start
x0 = np.array([["C"]])
with objective_factory.start(name="ddr3_docking") as f:
    print(x0)
    print(f(x0))
