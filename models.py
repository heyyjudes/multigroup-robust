import numpy as np
import copy

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# ignore grid search warnings  
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings

clf_dict = {
    "LR": LogisticRegression,
    "GB": GradientBoostingClassifier,
    "XGB": xgb.XGBClassifier,
    "KNN": KNeighborsClassifier, 
    "DT": DecisionTreeClassifier,
    "NN": MLPClassifier,
    "RF": RandomForestClassifier,
}

def model_choice(clf, xtrain=None, ytrain=None, scaling=True):
    param_grid_nn = {
        "mlp__alpha": [0.05, 0.1],
        "mlp__learning_rate": ["constant", "adaptive"],
        'mlp__hidden_layer_sizes': [(8, 2)] 
    }
    param_grid_knn = {
        "knn__n_neighbors": [3, 5, 7]
    }
    if scaling: 
        model = Pipeline([('scaling', StandardScaler())])
    else: 
        model = Pipeline([])
    if clf == "XBG":
        model.steps.append(("XGBoost", clf_dict[clf](objective="binary:logistic")))
    
    elif clf == "KNN": 
        temp_model = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        print("running model search")
        grid_search = GridSearchCV(temp_model, param_grid_knn, n_jobs=-1, cv=5)
        
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore",category=ConvergenceWarning) 
            grid_search.fit(xtrain, ytrain)

        # final model
        model.steps.append(("KNN", KNeighborsClassifier(grid_search.best_params_["knn__n_neighbors"])))
    elif clf == "NN":
        temp_model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=(8, 2),
                        random_state=1,
                        max_iter=500,
                    ),
                ),
            ]
        )
            
        print("running model search")
        grid_search = GridSearchCV(temp_model, param_grid_nn, n_jobs=-1, cv=5)
        grid_search.fit(xtrain, ytrain)
        print(grid_search.best_params_)
        model.steps.append(("mlp", MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=grid_search.best_params_["mlp__hidden_layer_sizes"],
                        random_state=1,
                        max_iter=500,
                        alpha=grid_search.best_params_["mlp__alpha"],
                        learning_rate=grid_search.best_params_["mlp__learning_rate"],
                    )))
        
    else:
        model.steps.append(("clf", clf_dict[clf]()))
    return model


class BoostingClassifier:
    def __init__(self, base_clf=None, coeff_arr=None, intercept_arr=None, eta=0.1):
        self.base_clf = base_clf
        self.coeff_arr = coeff_arr if coeff_arr is not None else []
        self.intercept_arr = intercept_arr if intercept_arr is not None else []
        self.eta = eta 

    def predict(self, X, thresh=0.5): 
        return self.predict_proba_1d(X) > thresh
    
    def predict_proba(self, X): 
        return np.vstack([1 - self.predict_proba_1d(X), self.predict_proba_1d(X)]).T
        

class MABoost(BoostingClassifier):
    def __init__(self, base_clf, coeff_arr=None, intercept_arr=None, eta=0.1):
        super().__init__(base_clf, coeff_arr, intercept_arr, eta)
        self.S_ind = []
        
    def predict_proba_1d(self, X):
        '''
        Implementing MABoost from Kim et al. 2019
        returns scalar probability of boosted model using self.coeff_arr and self.intercept_arr
        '''
        prob = self.base_clf.predict_proba(X)[:, 1]
        for i in range(len(self.coeff_arr)):
            z = X @ self.coeff_arr[i].T + self.intercept_arr[i]
            # partition mask x \in S*
            pmask = self.S_ind[i].astype(int)
            assert max(pmask) == 1 and min(pmask) == 0
            #print(pmask.sum())
            print("mask", pmask[:10])
            # print((z*pmask)[:10])
            print("update:", np.exp(-self.eta*z*pmask)[:10])
            prob *= np.exp(-self.eta*z*pmask) 
        return prob
    
    def multiplicative_update(self, new_coeff, new_intercept, S_ind): 
        self.coeff_arr.append(new_coeff)
        self.intercept_arr.append(new_intercept)
        self.S_ind.append(S_ind)
    
        
        
def multi_accuracy_boost(f_0, A, alpha, x_valid, y_valid, g_valid, max_T):
    """
    f_0: BoostingClassifier
    A: auditing function/algorithm
    alpha: accuracy parameter
    x_valid: validation data
    y_valid: validation labels
    """
    def res(p, y):
        """ custom smoothed resdidual function used in Kim et. al. """
        return y * ((p>=0.1)/(p + 1e-20) + (p<0.1) * (20 - 100  * p)) +\
            (1-y) * ((p < 0.9)/(1 - p + 1e-20) + (p>=0.9) * (100 * p - 80))
    
    # split validation data into 2 groups to avoid overfittiing
    main_set = np.random.choice(np.arange(len(x_valid)), int(len(x_valid)*0.5), replace=False)
    heldout_set = np.array([i for i in np.arange(len(x_valid)) if i not in main_set]) 
    
    # build S = X_0, X_1, X 
    thresh = 0.5    
    X_0 = f_0.predict_proba_1d(x_valid) > thresh
    noharm = [X_0, ~X_0, np.ones(len(X_0), dtype=bool)]
    
    # make independent copy of f_0 original predictor 
    f = copy.deepcopy(f_0)
    
    for t in range(max_T):

        print("pred", f.predict_proba_1d(x_valid)[:10])
        print("target", y_valid[:10].astype(int))
        residual = f.predict_proba_1d(x_valid) - y_valid 
        print("residual", residual[:10])
        delta = res(f.predict_proba_1d(x_valid), y_valid)
            
        # iterate over S \in X_0, ~X_0, X
        for i, s in enumerate(noharm): 
            h_clf = A(0.5)
            
            # select S data 
            samples_main = f_0.predict_proba_1d(x_valid[main_set]) > thresh
            samples_heldout = f_0.predict_proba_1d(x_valid[heldout_set]) > thresh
            
            if i == 2: 
                samples_main = np.ones(len(samples_main), dtype=bool)
                samples_heldout = np.ones(len(samples_heldout), dtype=bool)
            else: 
                samples_main = (samples_main - i).astype(bool) 
                samples_heldout = (samples_heldout - i).astype(bool)
            
            # fit h(x) for x \in S on residual = p(x) - y on main set
            # TODO: in the NN implementation, they use latent representation for x, 
            # not sure if this should be g_valid or x_valid
            h_clf.fit(g_valid[main_set][samples_main], residual[main_set][samples_main])
            
            print("score", h_clf.score(g_valid[heldout_set][samples_heldout], 
                              residual[heldout_set][samples_heldout]))
            #h_clf.fit(x_valid[main_set][main_ind], delta[main_set][main_ind])

            # evaluate residual correlation on heldout set
            clf_prediction = h_clf.predict(g_valid[heldout_set][samples_heldout])
            corr = np.mean(clf_prediction*residual[heldout_set][samples_heldout])
            
            #TODO: this condition is met very easily, does not corespond to MA for actually predefined groups
            
            # check if MA is violated
            if np.abs(corr) > alpha: 
                print(f"{i} violation found corr{corr:0.2e}, update")
                # multiplicative update on x \in S
                f.multiplicative_update(h_clf.coef_, h_clf.intercept_, s)
                break 
            else:
                print(f"{i} no violation found corr{corr:0.2e}")
        if i == 2: 
            break  
    return f

class MAEmp(BoostingClassifier):
    def __init__(self, base_clf=None, coeff_arr=None, intercept_arr=None, eta=0.1):
        super().__init__(base_clf, coeff_arr, intercept_arr, eta)
        self.v_c = [] 
        
    def predict_proba_1d(self, X):
        '''
        implementing our Algorithm 2: empirical MA 
        returns scalar probability of boosted model using self.coeff_arr and self.intercept_arr
        X: features where sensitive attributes are the last dims
        '''
        # initialize predictor to P(x) = 1/2 forall x 
        if self.base_clf == None: 
            prob = np.zeros(len(X))/2
        else: 
            prob = self.base_clf.predict_proba(X)[:, 1] 
        
        
        #print("before:", prob[:10])
        
        for i in range(len(self.coeff_arr)):
            # select x \ in C via coeff mask, subtract v_c 
            s_len = len(self.coeff_arr[i])
            
            # only compare the last s_len columns of X 
            # made sure sensitive attributes is at the end in preprocess function 
            # z : {x}^n -> [0, 1]^n
            z = (X[:, -s_len:] == self.coeff_arr[i]).all(axis=1)
            #print(f"positive updates {(z > 0).sum()}")
                   
            z = z.astype(float) * (- self.intercept_arr[i])
     
            # for x \in C, p(x) <- max{0, min{p(x), 1}}
            prob = np.clip(prob + z, a_min=0, a_max=1)
        
        #print("after:", prob[:10])
        
        return prob
    
    def update(self, coeff, intercept): 
        self.coeff_arr.append(coeff)
        # update v_c
        self.intercept_arr.append(intercept)
    

    

def multi_accuracy_boost_emp(f_0, A, alpha, x_valid, y_valid, g_valid, max_T): 
    
    if f_0.one_hot: 
        return multi_accuracy_boost_emp_onehot(f_0, A, alpha, x_valid, y_valid, g_valid, max_T)
    else: 
        # TODO: need to directly apply sensitive attribtue to group membership
        raise NotImplementedError("dense not implemented yet")
    
def multi_accuracy_boost_emp_onehot(f_0, alpha, x_valid, y_valid, g_valid, max_T):
    """
    f_0: BoostingClassifier
    A: auditing function/algorithm
    alpha: accuracy parameter
    x_valid: validation data
    y_valid: validation labels
    g_valid: validation group labels
    """
    
    # get unique groups (rows)
    unique_groups = np.unique(g_valid.astype(int), axis=0)
    
    # make independent copy of f_0 original predictor 
    f = copy.deepcopy(f_0)
    
    for t in range(max_T):
        residual = f.predict_proba_1d(x_valid) - y_valid.flatten() 
        #print(residual[:10])
        #print(f"positive updates {(residual > 0).sum()}")
        perm = np.random.permutation(len(unique_groups))

        # iterate over permutation of all subgroups
        for i in range(len(unique_groups)): 
            # TODO: check group violation -> is per group AUC supossed to be 0.5? 
            # membership_id = np.where(unique_groups[perm][i] == 1)[0]
            # for id in range(len(unique_groups[perm][i])):
            #     C = (g_valid[:, id] == 1)

            #     violation = np.abs(np.sum(residual[C])/len(x_valid))
            #     if violation > alpha: 
            #         v_c = np.sum(residual[C])/np.sum(C) 
            #         group = np.zeros((len(unique_groups[perm][i]), ))
            #         group[id] = 1
            #         f.update(group, v_c)
            #         break 
                
            # check intersectional group violation
            C = (g_valid == unique_groups[perm][i]).all(axis=1) # check this is the right axis
            # print(np.sum(residual[C]))
            # bias = 1/n * |sum_{i=1}^n p(x_i) - y_i1(x_i \in C)|
            violation = np.abs(np.sum(residual[C])/len(x_valid))
            if violation > alpha: 
                # v_c = 1/C * sum_{i=1}^n p(x_i) - y_i1(x_i \in C)
                v_c = np.sum(residual[C])/np.sum(C) 
                # print(v_c)
                # sanity check # 1
                # print(f"residual before {residual[C].sum():0.2e} vc {v_c:0.2e}")
                # update p(x)
                f.update(unique_groups[perm][i], v_c)
                # sanity check # 2
                # residual = f.predict_proba_1d(x_valid) - y_valid.flatten() 
                # print(f"residual after {residual[C].sum():0.2e}")
                break 
            # else: 
            #     print(f"MA OK for {unique_groups[perm][i]} group {violation:0.2e}")

        if i == len(unique_groups) - 1:
            print(f"iteration{t}, no violation found")
            break
    return f

def multi_accuracy_boost_emp_dense(f_0, A, alpha, x_valid, y_valid, g_valid, max_T):
    """
    f_0: BoostingClassifier
    A: auditing function/algorithm
    alpha: accuracy parameter
    x_valid: validation data
    y_valid: validation labels
    g_valid: validation group labels
    """
    
    # split validation data into 2 groups
    groups_arr = [] 
    print(g_valid.shape)
    for i in range(g_valid.shape[1]): 
        groups_arr.append(np.unique(g_valid[:, i]))
    all_pairs = np.meshgrid(*groups_arr) # works for 2 attributes, need to generalize 
    all_pairs = np.vstack([arr.flatten() for arr in all_pairs]).T
    
    # make independent copy of f_0 original predictor 
    f = copy.deepcopy(f_0)
    
    for t in range(max_T):
        residual = f.predict_proba_1d(x_valid) - y_valid 
        perm = np.random.permutation(len(all_pairs))
        
        # iterate over permutation of all subgroups
        for i in range(all_pairs.shape[0]): 
            a, b = all_pairs[perm][i]
            # check both individual and intersectional groups
            C0 = (g_valid[:, 0] == a)
            C1 = (g_valid[:, 1] == b)
            C2 = (g_valid[:, 0] == a) & (g_valid[:, 1] == b)

            for C in [C0, C1, C2]:
                violation_flag = False
                # bias = 1/n * |sum_{i=1}^n p(x_i) - y_i1(x_i \in C)|
                violation = np.abs(np.sum(residual[C])/len(x_valid))
                if violation > alpha: 
                    # v_c = 1/C * sum_{i=1}^n p(x_i) - y_i1(x_i \in C)
                    v_c = np.sum((residual)[C])/np.sum(C) 
                    # update p(x)
                    f.update(v_c, C)
                    violation_flag = True
                    break 
            if violation_flag:
                break
        if i == all_pairs.shape[0] - 1:
            print(f"iteration{t}, no violation found")
            break
    return f