import json
from skdim.exptutils import Experiments
import sys

## load datasets
with open("../datasets.json", "r") as f: data_params = json.load(f)

expt_params_filename = sys.argv[1]
with open('expt_params/' + expt_params_filename, "r") as f: expt_params = json.load(f)

ds_names = data_params['datasets'].keys()
es_name = expt_params["est_name"]

#for reproducibility
ds_random_state = data_params['meta_params']['ds_random_seed']
n_repeats = data_params['meta_params']['n_repeats']

results = dict()
results['data_params'] = data_params
results['expt_params'] = expt_params

results['numerical_output'] = {ds: [] for ds in ds_names}

for ds in ds_names:
    for idx, ds_params in enumerate(data_params['datasets'][ds]):
        results['numerical_output'][ds].append([])
        expt = Experiments(dataset_name=ds, estimator_name=es_name, random_state=ds_random_state)
        for jdx, es_params in enumerate(expt_params['est_params']): 
            results['numerical_output'][ds][-1].append(expt.basic_dimension_estimate(dataset_params=ds_params,estimator_params=es_params, n_repeats = n_repeats))

with open('results/'+ expt_params_filename, 'w') as f: json.dump(results, f)




        

