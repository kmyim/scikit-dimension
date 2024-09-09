import json
from skdim.exptutils import experiment_with_benchmark_parameters
from skdim.datasets import mnist
from skdim.id_flex import lPCA
import numpy as np

## load datasets
dataset, _ = mnist()
ds_name = 'mnist'
est_params_f = 'lpca_demo'
with open('est_params/' + est_params_f + '.json', "r") as f: expt_params = json.load(f)

out_f = ds_name + '_' + est_params_f + '.json'

error_criterion = np.inf
random_state = 12345


hyperparameter_search, best_benchmark_params = experiment_with_benchmark_parameters(dataset=dataset, estimator= lPCA, estimator_params = expt_params['est_params'], error_norm = error_criterion, random_state= expt_params['random_seed'])

results = dict()
results['error_criterion'] = error_criterion
results['random_state'] = random_state
results['expt_params'] = expt_params

results['hyperparameter_search'] = hyperparameter_search
results['best_benchmark_params'] = best_benchmark_params


with open('results/'+ out_f, 'w') as f: json.dump(results, f)




        

