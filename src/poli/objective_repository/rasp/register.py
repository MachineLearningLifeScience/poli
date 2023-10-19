from pathlib import Path

from poli.core.util.proteins.rasp import (
    RaspInterface,
    load_cavity_and_downstream_models,
)

NUM_ENSEMBLE = 10
DEVICE = "cpu"

THIS_DIR = Path(__file__).parent.resolve()
HOME_DIR = THIS_DIR.home()
RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
RASP_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    THIS_DIR = Path(__file__).parent.resolve()
    wildtype_pdb_path = THIS_DIR / "101m.pdb"
    chain_to_keep = "A"

    rasp_interface = RaspInterface(THIS_DIR / "tmp")

    rasp_interface.raw_pdb_to_unique_chain(wildtype_pdb_path, chain_to_keep)
    rasp_interface.unique_chain_to_clean_pdb(wildtype_pdb_path)
    rasp_interface.cleaned_to_parsed_pdb(wildtype_pdb_path)

    df_structure = rasp_interface.create_df_structure(wildtype_pdb_path)
    print(df_structure.head())

    # Loading the models
    cavity_model_net, ds_model_net = load_cavity_and_downstream_models()

    # Predicting
    dataset_key = "predictions"
    df_ml = rasp_interface.predict(
        cavity_model_net, ds_model_net, df_structure, dataset_key, NUM_ENSEMBLE, DEVICE
    )

    print(df_ml.head())
