import json
from skdim.exptutils import best_parameters, eval_estimator_parameters_on_benchmark
from skdim.id_flex import lPCA
import pickle


### get estimator parameter space

with open('macro_estimator_params_basic.json', "r") as f: 
    macro_est_params = json.load(f)
with open('lpca_demo.json', "r") as f: 
    spec_est_params = json.load(f)

estimator_params = macro_est_params | spec_est_params['est_params']

### get estimator data space
with open('../data_params/macro_dataset_params_basic.json', "r") as f: 
    macro_ds_params = json.load(f)


performance = eval_estimator_parameters_on_benchmark(lPCA, dataset_params = macro_ds_params['generator_params'], 
                                                     estimator_params = estimator_params, 
                                                     random_state = macro_ds_params['object_params']['random_state'])

with open('benchmark_perf_full.pkl', 'wb') as f:
    pickle.dump(performance, f)

best_parameters_on_benchmark = best_parameters(performance_dict=performance, criterion= 'quantiles', q = 0.5)
with open('est_params/best_parameters_benchmark.pkl', 'wb') as f:
    pickle.dump(best_parameters_on_benchmark, f)
