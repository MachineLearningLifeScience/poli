# Using the DRD3 objective function

There are several requirements for running this objective function. We expect you to

- have AutoDock Vina installed in the path,
- have the `prepare_receptor` binary from the ADFR suite installed and in the path,
- have the `poli__lambo` environment created.

The rest of this readme shows you how to do all this:

## Installing AutoDock Vina

### Download the files

[Download AutoDock Vina from the Center for Computational Structural Biology's website](https://vina.scripps.edu/downloads/). Uncompress them.

### Add the binary folder to the path.

Add this to the path by including `export PATH=path/to/AutoDock_vina/bin:$PATH` in your `~/.bashrc` or `~/.zshrc`.

```bash
# In your ~/.bashrc or ~/.zshrc
export PATH=path/to/AutoDock_vina/bin:$PATH
```

## Installing the ADFR suite

### Download the files

[Download the installable files](https://ccsb.scripps.edu/adfr/downloads/). It's likely that you will have to run the `./install.sh` script inside the folder, and thus you might have to change its permissions for execution using `chmod +x`

### Install it

After running `./install.sh`, you should be able to find `.../bin/prepare_receptor`.

### Add `prepare_receptor` to the path

For the docking to run, `pyscreener` needs access to the `prepare_receptor` binary. However, adding all of the ADFR `bin` folder is sometimes problematic, since it has a version of Python inside.

Thus, we recommend creating a symlink. Write this in your `~/.bashrc` or `~/.zshrc`:

```bash
# In your ~/.bashrc or ~/.zshrc
ln -sf /path/to/ADFR/bin/prepare_receptor /path/to/AutoDock_vina/bin
```

## Create the `poli__lambo` environment

### Create the environment from the yml file

This can easily be done by running

```bash
# From the base of the poli repo
conda env create --file src/poli/objective_repository/ddr3_docking/environment.yml
```

You will need to install the `lambo` repository manually.

### Cloning lambo

We also need `lambo`'s tasks to be available in Python's path for `poli__lambo`:

```bash
git clone https://github.com/samuelstanton/lambo    # For reference, we use 431b052
cd lambo
pip install -e .  
```

In particular, we need
- `lambo.tasks.proxy_rfp.proxy_rfp.ProxyRFPTask`
- the rfp data: see `~/lambo/assets/fpbase`

And now you should be set! Running `register.py` and `query_example.py` should work.
