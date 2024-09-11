import json
from skdim.datasets import BenchmarkManifolds
from skdim.exptutils import experiment_with_param_search, experiment_with_benchmark_parameters, hyperparameter_on_benchmark
from skdim.id_flex import lPCA
import numpy as np
from itertools import product


### generate estimator parameter space

with open('macro_estimator_params_basic.json', "r") as f: 
    macro_est_params = json.load(f)
with open('lpca_demo.json', "r") as f: 
    spec_est_params = json.load(f)

estimator_params = macro_est_params | spec_est_params['est_params']

### generate estimator data space
with open('../data_params/macro_dataset_params.json', "r") as f: 
    macro_ds_params = json.load(f)


ds_generator_params = macro_ds_params['generator_params']
ds_params_list = product(*[ds_generator_params[pm] for pm in ds_generator_params])

ds_best_params = dict()
for ds_params in ds_params_list: 
    input_ds_pms = {pm_name: ds_params[i] for i, pm_name in enumerate(ds_generator_params.keys())}
    estimator_train = hyperparameter_on_benchmark(lPCA, estimator_params, verbose = False, error_norm = np.inf,
                                **input_ds_pms, **macro_ds_params['object_params'])
    
    best_train_params = min(estimator_train.keys(), key = lambda x: estimator_train[x])
    ds_best_params[tuple([(k,v) for k,v in input_ds_pms.items()])] = {pm_name: best_train_params[i] for i, pm_name in enumerate(estimator_params.keys())}

with open('lPCA_train_params.json', "w") as f: 
    json.dump(ds_best_params, f)
    
### to do:   rescaling datasets by effective knn/quantile