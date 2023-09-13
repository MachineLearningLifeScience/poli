"""
This tiny example tests whether we can run
the foldx_rfp_lambo objective function.

You will unfortunately need to
- have foldx installed (we expect the binary to live 
  in ~/foldx/foldx, and for the rotabase.txt file
  to live in ~/foldx/rotabase.txt)
- To have lambo installed in either the environment you're
  running this in, or in the poli__lambo environment you
  can find in objective_repository/foldx_rfp_lambo/environment.yml
"""

from poli import objective_factory


_, f, x0, y0, _ = objective_factory.create(name="foldx_rfp_lambo")

print(x0)
print(y0)

f.terminate()
