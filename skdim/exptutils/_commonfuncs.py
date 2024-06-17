import numpy as np
from importlib import import_module 

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

    def basic_dimension_estimate(self, dataset_params, estimator_params, n_repeats):

        ds_rng = np.random.default_rng(self.random_state) #for reproducibility
        ds_random_seeds = ds_rng.choice(a = 1000*n_repeats, size =  n_repeats, replace = False)

        dim_list = []

        for k in range(n_repeats):
            ds_instance = self.ds(random_state = ds_random_seeds[k], **dataset_params)
            estimator = self.es()
            estimator.fit(X = ds_instance, **estimator_params)
            dim_list.append(estimator.dimension_)

        return np.nanmean(dim_list), np.nanstd(dim_list), np.mean(np.isnan(dim_list))
    
        
