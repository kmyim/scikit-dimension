import numpy as np
from importlib import import_module 
from skdim.datasets import random_embedding, BenchmarkManifolds
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

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
    


def best_parameters(performance_dict, criterion= 'quantiles', p = np.inf, q = 0.5):
    train_opt_params = dict() #stores for each dataset parameter in performance_dict, the best estimator parameters
    for dataset_pm in performance_dict:
        est_perf = dict() #for each set of estimator parameters, stores performance
        for est_params in performance_dict[dataset_pm]:
            errors = np.array([abs(a[1] - a[2]) for a in performance_dict[dataset_pm][est_params]]) # diff between estimated id and ground truth
            if criterion == 'norm':
                est_perf[est_params] = np.linalg.norm(errors, p)
            elif criterion == 'quantiles':
                est_perf[est_params] = np.quantile(errors, q)
        train_opt_params[dataset_pm] = min(est_perf.keys(), key = lambda k: est_perf[k]) #gets estimator parameter that minimises error proxy
    return train_opt_params



    
def eval_estimator_parameters_on_benchmark(estimator, dataset_params, estimator_params, random_state = 12345):
    #outer loop over datasets, then inner loop over parameters, so that each dataset is only generated once

    performance = dict() #catalogue performance of estimators for each (n_pts, extrinsic_dim) pair
    benchmark = BenchmarkManifolds(random_state = random_state)

    dataset_parameter_space = [[(pm, a) for a in dataset_params[pm]] for pm in dataset_params]

    for dataset_parameters in product(*dataset_parameter_space): #loop over (n_pts, extrinsic_dim) pairs
        estimator_parameters_perf = dict() #records performance of parameters on each benchmark dataset with (n_pts, extrinsic_dim)
        input_dataset_params = dict(dataset_parameters)
        dataset_collection = [ds_name for ds_name in benchmark._dict_truth 
                              if benchmark._dict_truth[ds_name][1] <=input_dataset_params['extrinsic_dim']] #compatible datasets
        
        for ds_name in dataset_collection: #loop over benchmark datasets

            id = benchmark._dict_truth[ds_name][0]
            performance_on_ds = benchmark_performance(ds_name, benchmark, estimator, estimator_params, random_state = random_state, **input_dataset_params)
            #dictionary {estimator parameters: estimated id}
            for est_pms in performance_on_ds:
                record = (ds_name, performance_on_ds[est_pms], id)
                if est_pms in estimator_parameters_perf:
                    estimator_parameters_perf[est_pms].append(record)
                else:
                    estimator_parameters_perf[est_pms]= [record]

        performance[dataset_parameters] = estimator_parameters_perf #given (n_pts, extrinsic_dim), performance of estimator paramaters over benchmark datasets
    
    return performance

def benchmark_performance(dataset_name, benchmark, estimator, estimator_params,  n_pts = 100, extrinsic_dim = 10, noise = 0.0, random_state = 12345):
    int_dim, ext_dim, _ = benchmark._dict_truth[dataset_name]
    data = benchmark.generate(dataset_name, n = n_pts, d = int_dim, dim = ext_dim, noise = noise) #generates the benchmark dataset with the default settings
    data = random_embedding(data, extrinsic_dim, random = True, state = random_state)

    performance = performance_eval(data, estimator, estimator_params)    
    return performance

    
def performance_eval(data, estimator, estimator_params):

    eps_radius = None
    if 'nbhd_type' in estimator_params:
        if 'eps' in estimator_params['nbhd_type']:
            eps_radius = {k: normalise_scale(data, metric = 'euclidean', n_neighbors = k, aggr = 'median', n_jobs = -1) for k in estimator_params['n_neighbors']}
            #future work: relax from euclidean
            
    pms_list = [[(pm, a) for a in estimator_params[pm]] for pm in estimator_params]

    performance = {pms: basic_estimation(data, estimator, pms, eps_radius = eps_radius) for pms in product(*pms_list)}
    return performance
            
def basic_estimation(dataset, estimator, estimator_params, eps_radius = None):
    input_params = dict(estimator_params)
    if 'nbhd_type' in input_params:
        if input_params['nbhd_type'] == 'eps':
            est = estimator(radius = eps_radius[input_params['n_neighbors']], **input_params)
        else:
            est = estimator(**input_params)
    else:
        est = estimator(**input_params)
    est.fit(X = dataset)

    return est.dimension_




def normalise_scale(data, metric = 'euclidean', n_neighbors = 10, aggr = 'median', n_jobs = 1, **kwargs):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs = n_jobs, **kwargs)
    nn.fit(data)
    dists, _= nn.kneighbors(X=data, n_neighbors=n_neighbors, return_distance=True)
    if aggr == 'median':
        scale = np.median(dists[:,-1])
    elif aggr == 'hmean':
        scale = np.hmean(dists[:,-1])
    else:
        scale = np.mean(dists[:,-1])
    
    return scale
