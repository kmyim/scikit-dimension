# Experiment procedures

Prescribe experiment parameters with  `/expt_params/expt_name.json`. To run experiment, execute

    python expt.py expt_name.json

Output datafile in `/results/expt_name.json` contains estimated dimensions. 

## Datasets
Should use the same dataset for all classifiers (exceptions?). They are listed in `datasets.json`.

The `json` file contains the following key-value pairs. There are two toplevel keys: "datasets" and "meta_params". The object under "datasets" is another set of key-value pairs, where the keys are the dataset names in `skdim`, and the values are arrays containing different parameters. The top level object under "meta_params" is a kv pair containing the number of repetitions and dataset generator random seed.

To load a dataset:

    import json
    import numpy as np
    from importlib import import_module 

    skdim_ds = import_module("skdim.datasets")

    with open("datasets.json", "r") as dummy:   data_params = json.load(dummy)
    ds_names = data_params['datasets'].keys()
    ds_random_state = data_params['meta_params']['ds_random_seed']

    ds_rng = np.random.default_rng(ds_random_state) #for reproducibility

    # example: 
    ds = ds_names[0] #dataset name
    ds_cl = getattr(skdim_ds, ds)

    in_params = data_params[ds][0] #parameters to instantiate data
    ds_obj = ds_cl(random_state = ds_rng.integers(low = 0, high = 10**5, size = 1), **in_params)


## Dimension Estimator

The dimension estimator parameters are stored as an array in `expt_params.json`.

To create an estimator:

    skdim_id = import_module("skdim.id_flex")

    with open("expt_params.json", "r") as dummy:   expt_params = json.load(dummy)
    est_name = expt_params['est_name']

    est_random_seed = expt_params['meta_params']['est_random_seed']
    est_rng = np.random.default_rng(est_random_seed) #for reproducibility

    # example: 
    est_cl = getattr(skdim_id, est_name)

    est_params = expt_params['est_params'][0] #parameters to instantiate data
    est_obj = est_cl(random_state = est_rng.integers(low = 0, high = 10**5, size = 1), **est_params)



