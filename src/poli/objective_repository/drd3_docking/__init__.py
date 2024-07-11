"""A wrapper around TDC's DRD3 docking objective function.

Prerequisites
-------------

There are several requirements for running this objective function. We expect you to

- have AutoDock Vina installed in the path,
- have the `prepare_receptor` binary from the ADFR suite installed and in the path,
- have the `poli__lambo` environment created.

The rest of this description shows you how to do all this:

Installing AutoDock Vina
========================

1. Download AutoDock Vina from the Center for Computational Structural Biology's website (https://vina.scripps.edu/downloads/). Uncompress them.

2. Add this to the path by including `export PATH=path/to/AutoDock_vina/bin:$PATH` in your `~/.bashrc` or `~/.zshrc`.

```bash
# In your ~/.bashrc or ~/.zshrc
export PATH=path/to/AutoDock_vina/bin:$PATH
```

Installing the ADFR suite
=========================

1. Download the installable files (https://ccsb.scripps.edu/adfr/downloads/). It's likely that you will have to run the `./install.sh` script inside the folder, and thus you might have to change its permissions for execution using `chmod +x`
2. After running `./install.sh`, you should be able to find `.../bin/prepare_receptor`.
3.  For the docking to run, `pyscreener` needs access to the `prepare_receptor` binary. However, adding all of the ADFR `bin` folder is sometimes problematic, since it has a version of Python inside. Thus, we recommend creating a symlink. Write this in your `~/.bashrc` or `~/.zshrc`:

```bash
# In your ~/.bashrc or ~/.zshrc
ln -sf /path/to/ADFR/bin/prepare_receptor /path/to/AutoDock_vina/bin
```

Create the `poli__lambo` environment
====================================

This can easily be done by running

.. code-block:: bash

        # From the base of the poli repo
        conda env create --file src/poli/objective_repository/ddr3_docking/environment.yml


We also need `lambo`'s tasks to be available in Python's path for `poli__lambo`:

.. code-block:: bash

            git clone https://github.com/samuelstanton/lambo    # For reference, we use 431b052
            cd lambo
            pip install -e .


In particular, we need
- `lambo.tasks.proxy_rfp.proxy_rfp.ProxyRFPTask`
- the rfp data: see `~/lambo/assets/fpbase`

And now you should be set! Running `register.py` and `query_example.py` should work.
"""
