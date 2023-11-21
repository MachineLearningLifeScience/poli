import numpy as np

from poli import objective_factory

# Using create
_, f, x0, y0, _ = objective_factory.create(name="penalized_logp_lambo")
print(x0)
print(y0)
f.terminate()

# Using start
x0 = np.array([["C"]])
with objective_factory.start(name="penalized_logp_lambo") as f:
    print(x0)
    print(f(x0))
