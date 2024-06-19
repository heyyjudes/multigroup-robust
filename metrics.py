import copy
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# local methods
import datasets as ds
import models as md
import pdb

race_encoding =  {
        1.0: "white",
        2.0: "black",
        3.0: "other",
        6.0: "asian",
    }

# train a logistic variety of models and evaluate it on the test set
def train_and_test(dataset, noise_rate, run=0, metric=None):
    results = []
    for model in ["LR", "XGB", "MAE"]:
        if clf == "MAE": 
            clf = md.MAEmp()
            clf = md.multi_accuracy_boost_emp_onehot(f_0=clf, 
                                                        alpha=1e-3,
                                                        x_valid=dataset.x_train, 
                                                        y_valid=dataset.y_train, 
                                                        g_valid=dataset.g_train, 
                                                        max_T=20)
            
        else: 
            clf = md.model_choice(model, xtrain=dataset.x_train, ytrain=dataset.y_train)
            clf.fit(dataset.x_train, dataset.y_train.ravel())
            
        if metric == "acc":
            results_dict = group_accuracy((dataset.y_test == clf.predict(dataset.x_test)), dataset.g_test)
        elif metric == "auc":
            results_dict = group_auc(dataset.y_test, 
                                        clf.predict_proba(dataset.x_test)[:, 1],
                                        dataset.g_test)
        elif metric == "ma": 
            results_dict = group_ma(dataset.y_test, 
                                    clf.predict_proba(dataset.x_test)[:, 1], 
                                    dataset.g_test)
        else: 
            # report all three if no metric specified 
            results_dict = {}
            acc_dict = group_accuracy((dataset.y_test == clf.predict(dataset.x_test)), dataset.g_test)
            auc_dict = group_auc(dataset.y_test, 
                                        clf.predict_proba(dataset.x_test)[:, 1],
                                        dataset.g_test)
            ma_dict = group_ma(dataset.y_test, clf.predict_proba(dataset.x_test)[:, 1], dataset.g_test)
            for suffix, _dict in zip(["_acc", "_auc", "_ma"], [acc_dict, auc_dict, ma_dict]):
                for key in _dict.keys():
                    results_dict[key + suffix] = _dict[key]

        results_dict["model"] = model 
        results_dict["noise"] = noise_rate
        results_dict["run"] = run
        results.append(results_dict)
    return pd.DataFrame(results)

def train_and_test_onehot(dataset, noise_rate, run=0, sanitize=False, poison=False, model=None):
    results = [] 
    # select data
    if poison: 
        x_train = dataset.x_poison
        y_train = dataset.y_poison
        g_train = dataset.g_poison
    else: 
        x_train = dataset.x_train
        y_train = dataset.y_train
        g_train = dataset.g_train
        
    if sanitize:
        if not poison: 
            print("WARNING: Sanitizing training data without poisoning")
        y_train = copy.deepcopy(md.data_sanitation_knn(x_train, y_train, k=5, num_iter=1).ravel())
    else: 
        y_train = y_train.ravel()
            
            
    if model == None: 
        model_arr = ["LR", "XGB", "KNN", "DT", "NN"]
    else: 
        if type(model) == list: 
            model_arr = model 
        else: 
            model_arr = [model]
    
    
    for model in model_arr:

        clf = md.model_choice(model, xtrain=x_train, ytrain=y_train)
        clf.fit(x_train, y_train)

        results_dict = eval_all_metrics(dataset, clf)

        results_dict["model"] = "-".join([model, "San"]) if sanitize else model
        results_dict["noise"] = noise_rate
        results_dict["run"] = run
        results.append(results_dict)
    return pd.DataFrame(results)


def train_postprocess_test_onehot(dataset, noise_rate, run=0, alpha=1e-4, max_T=20, poison=False, model=None):
    
    results = [] 
    
    if poison: 
        x_train = dataset.x_poison
        y_train = dataset.y_poison
        g_train = dataset.g_poison
    else: 
        x_train = dataset.x_train
        y_train = dataset.y_train
        g_train = dataset.g_train
    # MA boost from scratch
    print("MAEmp")
    ma_clf = md.MAEmp()
    ma_clf = md.multi_accuracy_boost_emp_onehot(f_0=ma_clf, 
                                                    alpha=alpha,
                                                    x_valid=x_train, 
                                                    y_valid=y_train, 
                                                    g_valid=g_train, 
                                                    max_T=max_T)

    # report all three if no metric specified 
    results_dict = eval_all_metrics(dataset, ma_clf, single_attr=dataset.single_attr)

    results_dict["model"] = "MAEmp"
    results_dict["noise"] = noise_rate
    results_dict["run"] = run
    results.append(results_dict)
    
    if model == None: 
        model_arr = ["LR", "XGB", "KNN", "DT", "NN"]
    else: 
        if type(model) == list: 
            model_arr = model
        else: 
            model_arr = [model]
    for model in model_arr:  
        clf = md.model_choice(model, xtrain=x_train, ytrain=y_train.ravel())
        clf.fit(x_train, y_train.ravel())

        # report all three if no metric specified 
        results_dict = eval_all_metrics(dataset, clf, single_attr=dataset.single_attr)
        print(model)
        results_dict["model"] = model 
        results_dict["noise"] = noise_rate
        results_dict["run"] = run
        results.append(results_dict)
        print("post processed")
        # post process on training data
        ma_clf = md.MAEmp(clf)
        ma_clf = md.multi_accuracy_boost_emp_onehot(f_0=ma_clf, 
                                                        alpha=alpha,
                                                        x_valid=x_train, 
                                                        y_valid=y_train, 
                                                        g_valid=g_train, 
                                                        max_T=max_T)
        
        # report all three if no metric specified 
        results_dict = eval_all_metrics(dataset, ma_clf, single_attr=dataset.single_attr)

        results_dict["model"] = "-".join([model, "PP"])
        results_dict["noise"] = noise_rate
        results_dict["run"] = run
        results.append(results_dict)
        
    return pd.DataFrame(results)

def eval_all_metrics(dataset, clf, single_attr=False): 
    results_dict = {}
    
    # pick threshold based on validation set
    thresh = optimal_threshold(dataset.y_valid, clf.predict_proba(dataset.x_valid)[:, 1])
    pred = clf.predict_proba(dataset.x_test)[:, 1] > thresh
    # sanity check
    # print(f"ratio predictions {pred.sum()/len(pred)}")
    acc_dict = accuracy_allgroups_onehot(dataset.y_test,
                    pred, 
                    dataset.g_test, 
                    dataset.s_labels, 
                    single_attr=single_attr)
    auc_dict = auc_allgroups_onehot(dataset.y_test,
                    clf.predict_proba(dataset.x_test)[:, 1], 
                    dataset.g_test, 
                    dataset.s_labels, 
                    single_attr=single_attr)
    ma_dict = ma_allgroups_onehot(dataset.y_test,
                    clf.predict_proba(dataset.x_test)[:, 1], 
                    dataset.g_test, 
                    dataset.s_labels, 
                    single_attr=single_attr)
    for suffix, _dict in zip(["_acc", "_auc", "_ma"], [acc_dict, auc_dict, ma_dict]):
        for key in _dict.keys():
            results_dict[key + suffix] = _dict[key]
            
    return results_dict 

def optimal_threshold(y_true, y_score):
    '''
    Alternative to 0.5 threshold: 
    Usage: 
        thresh = mt.optimal_threshold(dataset.y_valid, 
        clf.predict_proba(dataset.x_valid)[:, 1])
        pred = clf.predict_proba(dataset.x_test)[:, 1] > thresh
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def group_accuracy_1group(correct_arr, group, group_arr, min_size=1):
    '''
    correct_arr: array of booleans indicating whether prediction was correct
    group: group id (boolean pattern length k)
    group_arr: array of size nxk indicating group membership
    '''
    C = (group_arr == group).all(axis=1)
    return correct_arr[C].mean()

def group_accuracy(correct_arr, group_arr, min_size=10): 
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_acc = np.mean(correct_arr[group_arr == g])
            group_dict[race_encoding[g]] = g_acc
    # find non-white accuracy
    group_dict["non-white"] = np.mean(correct_arr[group_arr != 1])
    return group_dict

def group_ma(target, pred, group_arr, min_size=10): 
    '''
    target: true labels \in {0, 1}
    pred: predicted labels \in [0, 1]
    '''
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            g_ma = pred[group_arr == g] - target[group_arr == g].astype(int)
            g_ma = np.sum(g_ma)/len(group_arr) 
            group_dict[g] = np.abs(g_ma)
    # find non-white accuracy
    # g_ma = pred[group_arr != 1] - target[group_arr != 1].astype(int)
    # g_ma = np.sum(g_ma)/len(group_arr) 
    # group_dict["non-white"] = np.abs(g_ma)
    return group_dict

def group_auc(target, pred, group_arr, min_size=10): 
    group_dict = {}
    vals, counts = np.unique(group_arr, return_counts=True)
    for g, g_count in zip(vals, counts): 
        if g_count > min_size: 
            # AUC only valid if two classes exist 
            if (len(np.unique(target[group_arr == g])) > 1): 
                g_auc = roc_auc_score(target[group_arr == g], pred[group_arr == g])

                group_dict[race_encoding[g]] = g_auc
    # find all non-white AUC                 
    group_dict["non-white"] =roc_auc_score(target[group_arr != 1], pred[group_arr != 1])
    
    return group_dict 

def check_ma_1group(target, pred, group_arr, alpha): 
    ma_dict = group_ma(target, pred, group_arr, min_size=1)
    for g, ma in ma_dict.items(): 
        if ma > alpha: 
            print(f"MA violation found for {g} group {ma:0.2e}")
            return False
    return True

def check_ma_allgroups(target, pred, group_arr, alpha): 
    groups_arr = [] 
    if group_arr.ndim == 1 or group_arr.shape[1] == 1: 
        return check_ma_1group(target, pred, group_arr, alpha)
    
    for i in range(group_arr.shape[1]): 
        groups_arr.append(np.unique(group_arr[:, i]))

    # for multiple attributes we need to check each intersection of groups
    all_pairs = np.meshgrid(*groups_arr) 
    all_pairs = np.vstack([arr.flatten() for arr in all_pairs]).T
    
    for i in range(all_pairs.shape[0]): 
        a, b = all_pairs[i] # works for 2 attributes, need to generalize 
        C0 = (group_arr[:, 0] == a)
        C1 = (group_arr[:, 1] == b)
        C2 = (group_arr[:, 0] == a) & (group_arr[:, 1] == b)
        for C, group_str in zip([C0, C1, C2], [a, b, f"{a} {b}"]): 
            g_ma = pred[C] - target[C].astype(int)
            g_ma = np.sum(g_ma)/len(group_arr) 
            if np.abs(g_ma) > alpha: 
                print(f"MA violation found for {group_str} group {g_ma:0.2e}")
                return False

    return True 

def check_ma_violation(target, pred, group_arr, group_labels, alpha): 
    ma_dict = ma_allgroups_onehot(target, pred, group_arr, group_labels)
    violation = False
    for key in ma_dict.keys(): 
        if np.abs(ma_dict[key]) > alpha: 
            print(f"MA violation found for {key} group {ma_dict[key]:0.2e}")
            violation = True
    return violation
            
    
    
def ma_allgroups_onehot(target, pred, group_arr, group_labels, single_attr=False): 
    ma_dict = {} 
    # ensure target and pred are correct dim
    if target.shape != pred.shape: 
        target = target.flatten() 
        pred = pred.flatten() 
        
    # intersectional groups
    unique_groups = np.unique(group_arr.astype(int), axis=0)
    residual = pred.flatten() - target.astype(int).flatten()
    for g in unique_groups: 

        C = (group_arr == g).all(axis=1)
        g_ma = np.sum(residual[C])/len(group_arr)

        # race encoding
        ind = np.where(g == 1)[0]
        subgroup_label = "_".join([group_labels[i] for i in ind])
        ma_dict[subgroup_label] = g_ma
        
    # single attribute groups: 
    if not single_attr:
        for i in range(len(group_labels)): 
            # find all individuals in over arching group
            C = group_arr[:, i] == 1
            g_ma = np.sum(residual[C])/len(group_arr)
            ma_dict[group_labels[i]] = g_ma
        
    # over arching groups
    return ma_dict

def accuracy_allgroups_onehot(target, pred, group_arr, group_labels, single_attr=False): 
    '''
    check accuracy for all subgroups when group membership is one-hot encoded
    this will only check for single attribute groups (i.e. whether a single property is True)
    and unique intersectional groups 
    '''
    acc_dict = {}
    # find all intersectional groups
    unique_groups = np.unique(group_arr.astype(int), axis=0)
    
    # ensure target and pred are correct dim
    if target.shape != pred.shape: 
        target = target.flatten() 
        pred = pred.flatten() 
    
    for g in unique_groups: 
        acc = group_accuracy_1group(target == pred, g, group_arr)
        
        # race encoding
        ind = np.where(g == 1)[0]
        subgroup_label = "_".join([group_labels[i] for i in ind])
        acc_dict[subgroup_label] = acc
        
    # single attribute groups: 
    if not single_attr:
        for i in range(len(group_labels)): 
            # find all individuals in over arching group
            C = group_arr[:, i] == 1
            acc = np.mean((target == pred)[C])
            acc_dict[group_labels[i]] = acc
        
    acc_dict["all"] =  np.mean((target == pred))
    return acc_dict

def auc_allgroups_onehot(target, pred, group_arr, group_labels, single_attr=False): 
    '''
    check auc for all subgroups when group membership is one-hot encoded
    this will only check for single attribute groups (i.e. whether a single property is True)
    and unique intersectional groups 
    '''
    auc_dict = {}
    
    # ensure target and pred are correct dim
    if target.shape != pred.shape: 
        target = target.flatten() 
        pred = pred.flatten() 
        
    # find all intersectional groups
    unique_groups = np.unique(group_arr.astype(int), axis=0)
    for g in unique_groups: 
        #print(g)
        C = (group_arr == g).all(axis=1)
        auc = roc_auc_score(target[C], pred[C])
        #print(np.unique(pred[C], return_counts=True))
        # race encoding
        ind = np.where(g == 1)[0]
        subgroup_label = "_".join([group_labels[i] for i in ind])
        auc_dict[subgroup_label] = auc
        
    # single attribute groups: 
    if not single_attr:
        for i in range(len(group_labels)): 
            # find all individuals in over arching group
            C = group_arr[:, i] == 1
            auc = roc_auc_score(target[C], pred[C])
            auc_dict[group_labels[i]] = auc
        
    auc_dict["all"] = roc_auc_score(target, pred)
    return auc_dict