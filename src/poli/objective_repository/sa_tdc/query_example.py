import numpy as np

from poli import objective_factory

# Using create
_, f, x0, y0, _ = objective_factory.create(name="sa_tdc", force_register=True)
print(x0)
print(y0)  # Should be ~ 2.8548...
f.terminate()

# Using start
x0 = np.array([["C"]])
with objective_factory.start(name="sa_tdc") as f:
    print(x0)
    print(f(x0))
