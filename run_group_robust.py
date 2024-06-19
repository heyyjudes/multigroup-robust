import os

import numpy as np
import argparse
import pandas as pd

from sklearn import cluster

# local libs
import datasets as ds
import metrics as mt

def run_y_shift(ds_str, target_group, num_runs, alpha, sanitize=False, model=None): 
    dataset = ds.Dataset(ds_str, one_hot=True, remove_sensitive=False)
    results = pd.DataFrame() 
    if target_group not in dataset.subgroups_dict.keys():
        raise ValueError(f"target group {target_group} not in dataset {ds_str} please choose: {dataset.subgroups_dict.keys()}")
    group=dataset.subgroups_dict[target_group]
    
    target = 1 if ds_str == 'lawschool' else 0
    for i in range(num_runs): 
        for noise_rate in [0.0, 0.1, 0.2, 0.5, 0.7]: 
            dataset.group_label_shift(group=group, 
                                    noise_rate=noise_rate, 
                                    random_state=i, 
                                    target=target)
            if sanitize: 
                results = pd.concat([results, mt.train_and_test_onehot(dataset=dataset,
                                                                        model=model,
                                                                        noise_rate=noise_rate,
                                                                        run=i,
                                                                        sanitize=sanitize,
                                                                        poison=True)])   
            else:
                results = pd.concat([results, mt.train_postprocess_test_onehot(dataset=dataset, 
                                                                                noise_rate=noise_rate, 
                                                                                run=i, 
                                                                                alpha=alpha, 
                                                                                max_T = 20,
                                                                                model = model,
                                                                                poison = True)])
        if model: 
            model_str = f"_{model}"
        else: 
            model_str = ""
        if sanitize: 
            results.to_csv(f"results/sanitize/{ds_str}_label_shift_hot{target}{1-target}_g{target_group}_sanitize{model_str}.csv")
        else: 
            results.to_csv(f"results/{ds_str}_label_shift_hot{target}{1-target}_g{target_group}_postprocess{model_str}.csv")


def run_poison_addition(ds_str, modify_group, target_group, num_runs, alpha, sanitize=False, model=None):
    results = pd.DataFrame() 
    for n in range(num_runs):
        dataset = ds.Dataset(ds_str, one_hot=True, remove_sensitive=False)
    
        # groups are now string and not vectors 
        km = cluster.KMeans(n_clusters=100)
        km.fit(dataset.x_valid)
        train_km = km.predict(dataset.x_train)
        
        cl_inds, cl_cts = np.unique(km.labels_, return_counts=True)

        # translate group str
        if target_group not in dataset.subgroups_dict.keys():
            raise ValueError(f"target group {target_group} not in dataset {ds_str} please choose: {dataset.subgroups_dict.keys()}")
        tgt_group = dataset.subgroups_dict[target_group]
        
        if modify_group not in dataset.subgroups_dict.keys():
            raise ValueError(f"modify group {modify_group} not in dataset {ds_str} please choose: {dataset.subgroups_dict.keys()}")

        md_group = dataset.subgroups_dict[modify_group]

    
        # first check clean performance
        if sanitize: 
            clean_results = mt.train_and_test_onehot(dataset=dataset,
                                                    model=model,
                                                    noise_rate=0,
                                                    run=n,
                                                    sanitize=sanitize)
        else: 
            clean_results = mt.train_postprocess_test_onehot(dataset=dataset, 
                                                noise_rate=0, 
                                                run=n, 
                                                alpha=alpha, 
                                                max_T = 30, 
                                                model= model)
        results = pd.concat((results, clean_results), axis=0)
        for eps in [1, 2, 4, 8]: 
            x_poison = []
            y_poison = []
            g_poison = []
            for i, (cl_ind, cl_ct) in enumerate(zip(cl_inds, cl_cts)):
                if cl_ct > 10:
                    train_subclass = train_km == cl_ind
                    aux_subclass = km.labels_ == cl_ind

                    poison_x, poison_y, poison_g = (
                        dataset.x_valid[aux_subclass],
                        dataset.y_valid[aux_subclass],
                        dataset.g_valid[aux_subclass],
                    )

                    # poison on class in cluster,
                    sg, sg_ct = np.unique(poison_g.astype(int), return_counts=True, axis=0)
                    # flip groups where its divided:
                    tgt_ct = (poison_g == tgt_group).all(axis=1)
                    modify_ct = (poison_g == md_group).all(axis=1)

                    if tgt_ct.sum() > 0 and modify_ct.sum() > 0:

                        x_p = dataset.x_valid[aux_subclass][modify_ct]
                        # shift
                        if ds_str == 'lawschool':
                            # add negative label examples since 1 is majority class in lawschool
                            y_p = dataset.y_valid[aux_subclass][modify_ct] * 0
                        else: 
                            y_p = dataset.y_valid[aux_subclass][modify_ct] * 0 + 1

                        g_p = dataset.g_valid[aux_subclass][modify_ct]

                        x_p = np.repeat(x_p, eps, axis=0)
                        y_p = np.repeat(y_p, eps, axis=0)
                        g_p = np.repeat(g_p, eps, axis=0)

                        x_poison.append(x_p)
                        y_poison.append(y_p)
                        g_poison.append(g_p)


            dataset.x_poison = np.concatenate(
                (dataset.x_train, np.concatenate(x_poison, axis=0)), axis=0
            )
            dataset.y_poison = np.concatenate(
                (dataset.y_train, np.concatenate(y_poison, axis=0)), axis=0
            )
            dataset.g_poison = np.concatenate(
                (dataset.g_train, np.concatenate(g_poison, axis=0)), axis=0
            )
            
            print(f"adding {len(x_poison)} datapoints for a total size of {len(dataset.x_poison)}") 
            
            if sanitize: 
                poison_results = mt.train_and_test_onehot(dataset=dataset,
                                                                        model=model,
                                                                        noise_rate=eps,
                                                                        run=n,
                                                                        sanitize=sanitize,
                                                                        poison=True)  
            else:
                poison_results = mt.train_postprocess_test_onehot(dataset=dataset, 
                                                                                noise_rate=eps, 
                                                                                run=n, 
                                                                                alpha=alpha, 
                                                                                max_T = 30,
                                                                                model = model,
                                                                                poison = True)
            results = pd.concat((results, poison_results), axis=0)
        print(f"finished run {n}, saving results")
        
    if model: 
        model_str = f"_{model}"
    else: 
        model_str = ""
    if sanitize:
        results.to_csv(f"results/sanitize/{ds_str}_addition_attack_t{target_group}_m{modify_group}_sanitize{model_str}.csv")
    else: 
        results.to_csv(f"results/{ds_str}_addition_attack_t{target_group}_m{modify_group}{model_str}.csv")
    
if __name__ == "__main__": 
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run multigroup robusness experiments")

    # Add the arguments
    parser.add_argument("--modify", type=str, default='white-male', help="group to modify")
    parser.add_argument("--target", type=str, default='white-female', help="group to modify")
    parser.add_argument("--shift", action="store_true", default=False, help="label shift for modify group")
    parser.add_argument("--addition", action="store_true", default=False, help="data poisoning for modify group")
    parser.add_argument("--model", type=str, default="all", help="model to use")
    
    parser.add_argument("--num_runs", type=int, default=5, help="number of runs")
    parser.add_argument("--alpha", type=float, default=1e-5, help="error allowance")
    parser.add_argument("--sanitize", action="store_true", default=False, help="sanitize the data")
    
    # add dataset selection
    parser.add_argument("--dataset", type=str, default="income", help="dataset to use")

    # Parse the arguments
    args = parser.parse_args()
    
    # will run for all models if not specified
    model = None if args.model == "all" else args.model
        
    if args.shift: 
        run_y_shift(ds_str=args.dataset, 
                    target_group=args.modify,
                    num_runs =args.num_runs, 
                    alpha=args.alpha,
                    model=model,
                    sanitize=args.sanitize)
        
    if args.addition:
        run_poison_addition(ds_str=args.dataset,
                            modify_group=args.modify,
                            target_group=args.target,
                            num_runs=args.num_runs,
                            model=model,
                            alpha=args.alpha, 
                            sanitize=args.sanitize)
