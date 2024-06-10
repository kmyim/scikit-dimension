import json
import numpy as np
from importlib import import_module 
import sys

skdim_ds = import_module("skdim.datasets")
skdim_id = import_module("skdim.id_flex")

## load datasets
with open("../datasets.json", "r") as f: data_params = json.load(f)

expt_params_filename = sys.argv[1]
with open('expt_params/' + expt_params_filename, "r") as f: expt_params = json.load(f)

ds_names = data_params['datasets'].keys()

#for reproducibility
ds_random_state = data_params['meta_params']['ds_random_seed']
n_repeats = data_params['meta_params']['n_repeats']


# No randomness in MLE_basic
#est_random_seed = expt_params['est_meta_params']['est_random_seed']
#est_rng = np.random.default_rng(est_random_seed) 

est_name = expt_params['est_name']
est_cl = getattr(skdim_id, est_name)

# todo: parallelise these for loops
results = { 'dataset_meta_params': data_params['meta_params'],
            'estimator_params': expt_params}

for ds in ds_names:

    ds_cl = getattr(skdim_ds, ds)
    ds_rng = np.random.default_rng(ds_random_state) #for reproducibility
    ds_random_seeds = ds_rng.choice(a = 10**5, size = (len(data_params['datasets'][ds]), n_repeats), replace = False)
    results[ds] = [] 

    for idx, ds_params in enumerate(data_params['datasets'][ds]):

        dummy = dict()
        dummy['dataset_parameters'] = ds_params
        dimlist = np.zeros([len(expt_params['est_params']), n_repeats])

        for k in range(n_repeats):
            ds_obj = ds_cl(random_state = ds_random_seeds[idx][k], **ds_params)
            for jdx, est_params in enumerate(expt_params['est_params']): 
                estimator = est_cl()
                estimator.fit(X = ds_obj, **est_params)
                dimlist[jdx][k] = estimator.dimension_
        
        dummy['estimated_dim'] = (np.array([np.nanmean(dimlist, axis = -1), np.nanstd(dimlist, axis = -1)]).T).tolist()
        results[ds].append(dummy)


with open('results/'+ expt_params_filename, 'w') as f: json.dump(results, f)




        

