"""
This tiny example tests whether we can run
the foldx_rfp_lambo objective function.
"""

from poli import objective_factory


_, f, x0, y0, _ = objective_factory.create(name="foldx_rfp_lambo", force_register=True)

print(x0)
print(y0)
print(f)

f.terminate()
