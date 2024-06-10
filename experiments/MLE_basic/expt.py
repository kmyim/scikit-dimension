import json
import numpy as np
from importlib import import_module 

skdim_ds = import_module("skdim.datasets")
skdim_id = import_module("skdim.id_flex")

## load datasets
with open("../datasets.json", "r") as f: data_params = json.load(f)
with open("expt_params.json", "r") as f: expt_params = json.load(f)

ds_names = data_params['datasets'].keys()

#for reproducibility
ds_random_state = data_params['meta_params']['ds_random_seed']


# No randomness in MLE_basic
#est_random_seed = expt_params['est_meta_params']['est_random_seed']
#est_rng = np.random.default_rng(est_random_seed) 

est_name = expt_params['est_name']
est_cl = getattr(skdim_id, est_name)

# todo: parallelise these for loops
results = { 'dataset_params': data_params,
            'estimator_params': expt_params}

for ds in ds_names:
    ds_cl = getattr(skdim_ds, ds)
    ds_rng = np.random.default_rng(ds_random_state) #for reproducibility
    ds_random_seeds = ds_rng.choice(a = 10**5, size = len(data_params['datasets'][ds]), replace = False)
    results[ds] = [] #list of lists of results
    for idx, ds_params in enumerate(data_params['datasets'][ds]):
        ds_obj = ds_cl(random_state = ds_random_seeds[idx], **ds_params)
        results[ds].append([]) #list of estimator 
        for jdx, est_params in enumerate(expt_params['est_params']): 
            estimator = est_cl()
            estimator.fit(X = ds_obj, **est_params)
            results[ds][-1].append({'dim_est': estimator.dimension_, 'dim_std':  np.nanstd(estimator.dimension_pw_)})
            #results[ds][-1].append((estimator.dimension_, np.std(estimator.dimension_pw_)))

with open('results.json', 'w') as f: json.dump(results, f)




        

