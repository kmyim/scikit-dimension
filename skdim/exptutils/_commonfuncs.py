import numpy as np
from importlib import import_module 
from skdim.datasets import random_embedding, BenchmarkManifolds
from itertools import product, chain

class Experiments():
    def __init__(self, dataset_name, estimator_name, random_state, n_jobs = 1):

        self.ds_name = dataset_name
        self.es_name = estimator_name
        self.random_state = random_state
        self.n_jobs = n_jobs

        skdim_ds = import_module("skdim.datasets")
        skdim_id = import_module("skdim.id_flex")

        self.ds = getattr(skdim_ds, dataset_name)
        self.es = getattr(skdim_id, estimator_name)

    def basic_dimension_estimate(self, dataset_params, estimator_params, n_repeats, verbose = False):

        ds_rng = np.random.default_rng(self.random_state) #for reproducibility
        ds_random_seeds = ds_rng.choice(a = 1000*n_repeats, size =  n_repeats, replace = False)

        dim_list = []

        for k in range(n_repeats):
            ds_instance = self.ds(random_state = ds_random_seeds[k], **dataset_params)
            estimator = self.es(**estimator_params)
            estimator.fit(X = ds_instance)
            dim_list.append(estimator.dimension_)
        if verbose:
            return dim_list
        else:
            return np.nanmean(dim_list), np.nanstd(dim_list), np.mean(np.isnan(dim_list))
        
    def fix_estimator_vary_n_pts(self, n_range, dataset_params, estimator_params, n_repeats):

        ds_rng = np.random.default_rng(self.random_state) #for reproducibility
        ds_random_seeds = ds_rng.choice(a = 1000*n_repeats, size =  n_repeats, replace = False)

        dim_list = []

        estimator = self.es() # estimator = self.es(**estimator_params)
        pruned_ds_params = {k: dataset_params[k] for k in dataset_params if k != 'n'}
        for n in n_range:
            dim_list_n = []
            for k in range(n_repeats):
                ds_instance = self.ds(random_state = ds_random_seeds[k], n = n, **pruned_ds_params)
                estimator.fit(X = ds_instance, **estimator_params)
                dim_list_n.append(estimator.dimension_)
            dim_list.append((np.nanmean(dim_list_n), np.nanstd(dim_list_n), np.mean(np.isnan(dim_list_n))))

        return dim_list
    

def experiment_with_benchmark_parameters(dataset, estimator, estimator_params, error_norm = np.inf, random_state =12345):
    
    n_pts, extrinsic_dim = dataset.shape
    hyperparam_search = experiment_with_param_search(dataset, estimator, estimator_params)

    estimator_train = hyperparameter_on_benchmark(estimator, estimator_params,
                                                                     n_pts, extrinsic_dim =extrinsic_dim,
                                                                     random_state = random_state,
                                                                     error_norm = error_norm,
                                                                     verbose = False)
    
    best_train_params = min(estimator_train.keys(), key = lambda x: estimator_train[x])
    hyperparam_search = experiment_with_param_search(dataset, estimator, estimator_params)

    return hyperparam_search, best_train_params


def hyperparameter_on_benchmark(estimator, estimator_params, 
                                n_pts, extrinsic_dim, noise_type = "uniform", noise = 0.0, random_state = 12345, 
                                verbose = False, error_norm = np.inf):
    
    benchmark = BenchmarkManifolds(random_state = random_state, noise_type= noise_type)
    raw_datasets = benchmark.generate('all', n = n_pts, noise = noise)
    datasets = dict()
    for ds in raw_datasets:
        data = raw_datasets[ds]
        native_dim = data.shape[1]
        if native_dim < extrinsic_dim:
            datasets[ds] = random_embedding(data, extrinsic_dim, random = True, state = random_state)
    
    if 'nbhd_type' in estimator_params: #local method
        pms_search_list = []
        if 'knn' in estimator_params['nbhd_type']:
            pms_list = [[(pm, a) for a in estimator_params[pm]] for pm in estimator_params if pm != 'radius']
            pms_search_list = product(*pms_list)
        if 'eps' in estimator_params['nbhd_type']:
            pms_list = [[(pm, a) for a in estimator_params[pm]] for pm in estimator_params if pm != 'n_neighbors']
            pms_search_list = chain(pms_search_list, product(*pms_list))
    else: #global method
        pms_list = [[(pm, a) for a in estimator_params[pm]] for pm in estimator_params]
        pms_search_list = product(*pms_list)

    estimator_raw_performance = dict()
    estimator_performance = dict()

    for pms in pms_search_list:
        est = estimator(**dict(pms))

        ds_perf = dict()
        for ds in datasets:
            id = benchmark._dict_truth[ds][0]
            est.fit(X = datasets[ds])
            ds_perf[ds] = (est.dimension_ , id)
        
        pm_key = tuple(pms)
        estimator_performance[pm_key] = np.linalg.norm([a[1] - a[0] for a in ds_perf.values()], ord = error_norm)
        estimator_raw_performance[pm_key] = ds_perf

    if verbose:
        return estimator_raw_performance
    else:
        return estimator_performance
    
def experiment_with_param_search(data, estimator, estimator_params):
    
    pms_list = [estimator_params[pm] for pm in estimator_params]
    pms_search_list = product(*pms_list)

    hyperparam_search = dict()

    for pms in pms_search_list:
        input_pms = {pm: pms[i] for i, pm in enumerate(estimator_params.keys())}
        pm_key = tuple([(pm, input_pms[pm]) for pm in input_pms])
        est = estimator(**input_pms)
        est.fit(data)
        hyperparam_search[pm_key] = est.dimension_
    
    return hyperparam_search




        
        
     
        


