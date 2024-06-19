import numpy as np
import copy
import cvxpy as cp

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

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
        
    elif clf == "DT":
        model.steps.append(("DT", DecisionTreeClassifier(max_depth=10)))
        
    elif clf == "RLR": 
        model = RobustLogRegression()
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
        

class RobustLogRegression:
    def __init__(self):
        self.beta = None
        self.scaler = StandardScaler()
    
    def fit(self, x, y):
        n = len(y)
        p = x.shape[1]
        x_fit = self.scaler.fit_transform(x)
        v = cp.Variable()
        eta = cp.Variable(len(y))
        beta = cp.Variable(p)
        constr = []
        constr += [cp.multiply(y, (x_fit @ beta)) + eta + v >= 0]
        constr += [v >= 0]
        constr += [eta >= 0]
        constr += [cp.norm(beta, 2) <= 1]
        obj = -v * n - cp.sum(eta)
        prob = cp.Problem(cp.Maximize(obj), constr)
        prob.solve()
        self.beta = beta.value
        return beta.value

    def predict(self, x, thresh):
        x = self.scaler.transform(x)
        return self.predict_proba(x) > thresh
    
    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pos_class = 1/(1 + np.exp(-x @ self.beta))
        neg_class = 1 - pos_class
        return np.vstack([neg_class, pos_class]).T
    
    def pred_prob1d(self, x):
        x = self.scaler.transform(x)
        return 1/(1 + np.exp(-x @ self.beta))

class MAEmp(BoostingClassifier):
    def __init__(self, base_clf=None, coeff_arr=None, intercept_arr=None, eta=0.1):
        super().__init__(base_clf, coeff_arr, intercept_arr, eta)
        
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
        
        for i in range(len(self.coeff_arr)):
            # select x \ in C via coeff mask, subtract v_c 
            s_len = len(self.coeff_arr[i])
            # only compare the last s_len columns of X 
            # made sure sensitive attributes is at the end in preprocess function 
            # z : {x}^n -> [0, 1]^n
            z = (X[:, -s_len:] == self.coeff_arr[i]).all(axis=1)
                   
            z = z.astype(float) * (- self.intercept_arr[i])
            prob = np.clip(prob + z, a_min=0, a_max=1)

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
        perm = np.random.permutation(len(unique_groups))

        # iterate over permutation of all subgroups
        for i in range(len(unique_groups)):
            C = (g_valid == unique_groups[perm][i]).all(axis=1) # check this is the right axis
            violation = np.abs(np.sum(residual[C])/len(x_valid))
            if violation > alpha:
                v_c = np.sum(residual[C])/np.sum(C)
                f.update(unique_groups[perm][i], v_c)
                break
        if i == len(unique_groups) - 1:
            print(f"iteration{t}, no violation found")
            break
    return f

def data_sanitation_knn(x_train, y_train, k=10, num_iter=1): 
    """
    x_train: features
    y_train: labels
    k: number of neighbors to consider
    Label Sanitization against Label Flipping
    Poisoning Attacks
    https://arxiv.org/pdf/1803.00992.pdf
    """
    # find nearest neighbors for each point in x_train
    y_sanitized = copy.deepcopy(y_train)
    for i in range(num_iter): 
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train.ravel())
        # dont actually need the distances
        neighbors = knn.kneighbors(x_train, return_distance=False) # incides of neighbors
        avg_label = np.mean(y_train[neighbors], axis=1)
        pos_class_mask = avg_label >= 0.80
        y_sanitized[pos_class_mask] = 1
        neg_class_mask = avg_label <= 0.20
        y_sanitized[neg_class_mask] = 0
    print("Total sanitized", (y_sanitized != y_train).sum(), "out of", len(y_train), "samples")
    return y_sanitized