import numpy as np
import argparse
import pandas as pd

from sklearn.linear_model import Ridge 
from sklearn import metrics, cluster

# local libs
import datasets as ds
import models as md
import metrics as mt

exclude_attr_list = ['RAC1P_Native Hawaiian and Other Pacific Islander alone', 
                     'RAC1P_American Indian alone', 
                     'RAC1P_Alaska Native alone', 
                     'RAC1P_Some Other Race alone', 
                     'RAC1P_Two or More Races', 
                     'RAC1P_American Indian or Alaska Native, not specified',]

subgroups_dict = {"white-male": [0, 1, 0, 0, 1], 
                  "white-female": [1, 0, 0, 0, 1], 
                  "black-male": [0, 1, 1, 0, 0],
                  "black-female": [1, 0, 1, 0, 0],
                  "asian-male": [0, 1, 0, 1, 0],
                  "asia-female": [1, 0, 0, 1, 0]}


def run_y_shift(target_group, num_runs, alpha): 
    adult = ds.Dataset('income', one_hot=True, remove_sensitive=False, exclude_sensitive=exclude_attr_list)
    dataset = adult
    results = pd.DataFrame() 
    group=subgroups_dict[target_group]
    target = 0
    for i in range(num_runs): 
        for noise_rate in [0, 0.1, 0.2, 0.5, 0.7]: 
            adult.restore_training()
            adult.group_label_shift(group=group, 
                                    noise_rate=noise_rate, 
                                    random_state=i, 
                                    target=target)
            results = pd.concat([results, mt.train_postprocess_test_onehot(dataset=adult, 
                                                                            noise_rate=noise_rate, 
                                                                            run=i, 
                                                                            alpha=alpha, 
                                                                            max_T = 20)])
            
            
            
    results.to_csv(f"results/adult_label_shift_hot{target}{1-target}_g{target_group}_postprocess.csv")
    

def run_poison_addition(modify_group, target_group, num_runs, alpha): 
    
    results = pd.DataFrame() 
    for n in range(num_runs):
        adult = ds.Dataset('income', one_hot=True, remove_sensitive=False, exclude_sensitive=exclude_attr_list)
        dataset = adult
    
        # groups are now string and not vectors 
        km = cluster.KMeans(n_clusters=100)
        km.fit(dataset.x_valid)
        train_km = km.predict(dataset.x_train)
        
        cl_inds, cl_cts = np.unique(km.labels_, return_counts=True)

        # translate group str
        md_group = subgroups_dict[modify_group]
        tgt_group = subgroups_dict[target_group]
    
        # first check clean performance
        clean_results = mt.train_postprocess_test_onehot(dataset=adult, 
                                                noise_rate=0, 
                                                run=n, 
                                                alpha=alpha, 
                                                max_T = 30)
        results = pd.concat((results, clean_results), axis=0)
        
        for eps in [2, 4, 8, 16]: 
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
                        y_p = dataset.y_valid[aux_subclass][modify_ct] * 0 + 1

                        # flip
                        # flip doesn't really work
                        # y_p = 1 - dataset.y_valid[aux_subclass][modify_ct]
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
            poison_results = mt.train_postprocess_test_onehot(dataset=adult, 
                                                noise_rate=eps, 
                                                run=n, 
                                                alpha=alpha, 
                                                max_T = 30, 
                                                poison=True)
            
            results = pd.concat((results, poison_results), axis=0)
        
            results.to_csv(f"results/adult_addition_attack_t{target_group}_m{modify_group}.csv")
    
if __name__ == "__main__": 
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run multigroup robusness experiments")

    # Add the arguments
    parser.add_argument("--modify", type=str, default='white-male', help="group to modify")
    parser.add_argument("--target", type=str, default='white-female', help="group to modify")
    parser.add_argument("--shift", action="store_true", default=False, help="label shift for modify group")
    parser.add_argument("--addition", action="store_true", default=False, help="data poisoning for modify group")
    
    parser.add_argument("--num_runs", type=int, default=5, help="number of runs")
    parser.add_argument("--alpha", type=float, default=1e-5, help="error allowance")
    

    # Parse the arguments
    args = parser.parse_args()
    
    if args.shift: 
        run_y_shift(args.modify, args.num_runs, args.alpha)
        
    if args.addition:
        run_poison_addition(args.modify, args.target, args.num_runs, args.alpha)