
import pandas as pd 
import numpy as np
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage
from sklearn.model_selection import train_test_split
import acs
import copy
import pdb

class Dataset:
    def __init__(self, dataset_name, 
                one_hot=False, remove_sensitive=True) -> None:
        self.dataset_name = dataset_name
        self.one_hot = one_hot
        self.x_labels = None 
        self.s_labels = None
        self.single_attr = False 
        self.acs_dict = {'income': (ACSIncome, acs.ACSIncome_categories),
                        'employment': (ACSEmployment, acs.ACSEmployment_categories),
                        'coverage': (ACSPublicCoverage, acs.ACSPublicCoverage_categories)}
                
        self.x, self.y, self.g = self.preprocess(dataset_name, 
                                                one_hot=one_hot, 
                                                remove_sensitive=remove_sensitive)

        self.split_train_test() 

    def preprocess(self, dataset_name, one_hot, remove_sensitive):
        '''
        preprocesses the dataset based on the dataset name.
        dataset_name (str): name of the dataset to preprocess
        one_hot (bool): whether to use one hot encoding for categorical variables
        remove_sensitive (bool): whether to remove sensitive attributes from features
        '''
        if dataset_name == 'compas': 
            return process_compas('datasets/compas-scores-two-years.csv')
        elif dataset_name in ['income', 'coverage', 'employment']: 
            return self.preprocess_acs(dataset_name, one_hot, remove_sensitive)
        elif dataset_name == 'bank':
            return self.preprocess_bank(dataset_name, one_hot, remove_sensitive) 
        elif dataset_name == 'lawschool':
            return self.preprocess_lawschool(dataset_name, one_hot, remove_sensitive)
        else: 
            raise ValueError("Dataset name not recognized")
        
    def preprocess_bank(self, dataset_name, one_hot=True, remove_sensitive=False):
        '''
        custom preprocessing for bank dataset, one_hot encoding only
        '''
        
        self.subgroups_dict = {"cell-age0-17": [1, 0, 1, 0, 0, 0, 0], 
                                "cell-age18-29": [1, 0, 0, 1, 0, 0, 0], 
                                "cell-age30-44": [1, 0, 0, 0, 1, 0, 0],
                                "cell-age45-59": [1, 0, 0, 0, 0, 1, 0],
                                "cell-age60+": [1, 0, 0, 0, 0, 0, 1], 
                                "tele-age0-17": [0, 1, 1, 0, 0, 0, 0], 
                                "tele-age18-29": [0, 1, 0, 1, 0, 0, 0], 
                                "tele-age30-44": [0, 1, 0, 0, 1, 0, 0],
                                "tele-age45-59": [0, 1, 0, 0, 0, 1, 0],
                                "tele-age60+": [0, 1, 0, 0, 0, 0, 1]}
        self.single_attr = True
        if not one_hot: 
            raise ValueError("Bank dataset can only be preprocessed with one hot encoding")
        
        df = pd.read_csv('data/bank-additional/bank-additional-full.csv', delimiter=";")
        y = df['y'].map({'no': 0, 'yes': 1})
        df = df.drop('y', axis=1)
        
        bins = [0, 18, 30, 45, 60, 120]  # Define the bin edges
        labels = ['0-17', '18-29', '30-44', '45-59', '60+']  # Define the labels for each bin
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day_of_week', 'poutcome', 'age_group']
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # drop continuous age
        # keep for now
        # df_encoded = df_encoded.drop('age', axis=1)
        
        # get sensitive attributes
        sensitive_attr_keys = [key for key in df_encoded.keys() if (key.startswith('age_group') or key.startswith('contact'))]
        
        s = df_encoded[sensitive_attr_keys]
        x = df_encoded.drop(sensitive_attr_keys, axis=1)

        if not remove_sensitive:
            x = pd.concat((x, s), axis=1)

        self.x_labels = x.keys()
        self.s_labels = s.keys()
        
        return x.values, y.values, s.values

    def preprocess_lawschool(self, dataset_name, one_hot, remove_sensitive):
        self.single_attr = True
        self.exclude_attr_list = ['race1_other']
        self.subgroups_dict = {'asian-male': [1, 0, 0, 0, 0, 1],
                                'asian-feamle': [1, 0, 0, 0, 1, 0],
                                'black-male': [0, 1, 0, 0, 0, 1],
                                'black-female': [0, 1, 0, 0, 1, 0],
                                'hisp-male': [0, 0, 1, 0, 0, 1],
                                'hisp-female': [0, 0, 1, 0, 1, 0],
                                'white-male': [0, 0, 0, 1, 0, 1], 
                                'white-female': [0, 0, 0, 1, 1, 0]}
        df = pd.read_csv('data/lsac.csv', index_col=0)
        
        # get label 
        y = df['bar']
        df = df.drop('bar', axis=1)
        
        # encode categorical 
        categorical_cols = ['race1', 'gender']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        for key in self.exclude_attr_list:
            df_encoded = df_encoded.drop(key, axis=1)
        
        sensitive_attr_keys = [key for key in df_encoded.keys() if key.startswith('race1') or key.startswith('gender')]
        s = df_encoded[sensitive_attr_keys]
        x = df_encoded.drop(sensitive_attr_keys, axis=1)
        
        if not remove_sensitive:
            x = pd.concat((x, s), axis=1)
            
        self.x_labels = x.keys()
        self.s_labels = s.keys()
        
        return x.values, y.values, s.values
        

        
    def preprocess_acs(self, dataset_name, one_hot, remove_sensitive): 
        self.exclude_attr_list = ['RAC1P_Native Hawaiian and Other Pacific Islander alone', 
                     'RAC1P_American Indian alone', 
                     'RAC1P_Alaska Native alone', 
                     'RAC1P_Some Other Race alone', 
                     'RAC1P_Two or More Races', 
                     'RAC1P_American Indian or Alaska Native, not specified',]

        self.subgroups_dict = {"white-male": [0, 1, 0, 0, 1], 
                        "white-female": [1, 0, 0, 0, 1], 
                        "black-male": [0, 1, 1, 0, 0],
                        "black-female": [1, 0, 1, 0, 0],
                        "asian-male": [0, 1, 0, 1, 0],
                        "asia-female": [1, 0, 0, 1, 0]}

        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        data = data_source.get_data(states=["LA"], download=True)
        
        data_f, data_encoding = self.acs_dict[dataset_name]
        if one_hot:
            # use one hot encoding for categorical variables
            x, y, _ = data_f.df_to_pandas(data, categories=data_encoding, dummies=True)
            # remove sensitive attributes from features and labels 
            for key in self.exclude_attr_list:
                if key in x.keys():
                    print("removing: ", key)
                    x = x.drop(key, axis=1)
                
            sensitive_attr_keys = [key for key in x.keys() if key.startswith("RAC1P") or key.startswith("SEX")]
            s = x[sensitive_attr_keys]
            
            # remove sensitive attributes from features 
            x = x.drop(sensitive_attr_keys, axis=1)

            # reorder sensitive attributes
            if not remove_sensitive:
                # concatenate so sensitive attributes is at the end
                x = pd.concat((x, s), axis=1)
                
            # save labels
            self.x_labels = x.keys()
            self.s_labels = s.keys()
            print(f"{dataset_name} x shape: {x.shape}")
            # convert back to numpy 
            return x.values, y.values, s.values
            
        else: 
            x, y, g = data_f.df_to_numpy(data)
            if remove_sensitive: 
                # remove race and gender from features 
                x = np.delete(x, [-2, -1], axis=1)
            
            race = np.vectorize(adult_race_grouping.get)(g)
            gender = x[:, -2]
            # sensitive attributes
            s = np.vstack((gender, race)).T
            
            # save labels 
            self.x_labels = data_f.features() if not remove_sensitive else data_f.features()[:-2]
            self.s_labels = data_f.features()[-2: ] 
            return x, y, s
    def split_train_test(self, test_size=0.3, random_state=0):
        # split into train and non_train
        self.x_train, self.x_test, self.y_train, self.y_test, self.g_train, self.g_test = \
            train_test_split(self.x, self.y, self.g, test_size=test_size, random_state=random_state)
        
        self.x_valid, self.x_test, self.y_valid, self.y_test, self.g_valid, self.g_test = \
            train_test_split(self.x_test, self.y_test, self.g_test, test_size=0.5, random_state=random_state)
        
        return 
    
    
    def group_label_flip(self, group, group_ind=1, noise_rate=0.2, random_state=0):
        if self.one_hot: 
            self.group_label_flip_one_hot(group, 
                                          noise_rate, 
                                          random_state)
        else: 
            self.group_label_flip_dense(group,
                                        group_ind,
                                        noise_rate, 
                                        random_state)

    def group_label_flip_one_hot(self, group, noise_rate=0.2, random_state=0):
        """
        Flips the labels of a specified group within a dataset to introduce noise.
        
        Parameters:
        group (list): The group within the dataset whose labels will be flipped. e.g. 0 for white, 1 for black, etc.
        noise_rate (float, optional): The proportion of labels in the specified group that will be randomly flipped. This value should be between 0 and 1, where 0 means no label is flipped, and 1 means all labels are flipped. Defaults to 0.2.
        random_state (int, optional): A seed value for the random number generator to ensure reproducibility of the label flipping. Defaults to 0.

        Returns:
        None: This method does not return a value but instead modifies the labels of the specified group in-place.
         """
        # Safety checks
        if not (0 <= noise_rate <= 1):
            raise ValueError("Noise rate must be between 0 and 1.")

        rng = np.random.default_rng(random_state)

        # find index of given group
        if group == 'all': 
            idx = np.arange(len(self.g_train))
        else:
            idx = np.where((self.g_train == group).all(axis=1))[0]

        # flip labels at idx with probability noise_rate
        to_flip = rng.choice(idx, int(len(idx) * noise_rate), replace=False)
        self.y_train[to_flip] = 1 - self.y_train[to_flip]
            
    def group_label_flip_dense(self, group, group_ind=1, noise_rate=0.2, random_state=0):
        """
        Flips the labels of a specified group within a dataset to introduce noise.
        
        Parameters:
        group (int): The group within the dataset whose labels will be flipped. e.g. 0 for white, 1 for black, etc.
        group_ind (int, optional): The index of the label in the group to be flipped. Defaults to 1. Race
        noise_rate (float, optional): The proportion of labels in the specified group that will be randomly flipped. This value should be between 0 and 1, where 0 means no label is flipped, and 1 means all labels are flipped. Defaults to 0.2.
        random_state (int, optional): A seed value for the random number generator to ensure reproducibility of the label flipping. Defaults to 0.

        Returns:
        None: This method does not return a value but instead modifies the labels of the specified group in-place.
         """
        # Safety checks
        if not (0 <= noise_rate <= 1):
            raise ValueError("Noise rate must be between 0 and 1.")

        rng = np.random.default_rng(random_state)

        # find index of given group
        if group == 'all': 
            idx = np.arange(len(self.g_train))
        else:
            if group not in self.g_train[:, group_ind]:
                raise ValueError(f"Group {group} not found in g_train.")
            idx = np.where(self.g_train[:, group_ind] == group)[0]

        # flip labels at idx with probability noise_rate
        to_flip = rng.choice(idx, int(len(idx) * noise_rate), replace=False)
        self.y_train[to_flip] = 1 - self.y_train[to_flip]


    def group_label_shift(self, group, group_ind=1, noise_rate=0.2, random_state=0, target=0):
        if self.one_hot: 
            self.group_label_shift_one_hot(group, 
                                          noise_rate, 
                                          random_state, 
                                          target)
        else: 
            self.group_label_shift_dense(group,
                                        group_ind,
                                        noise_rate, 
                                        random_state, 
                                        target)
            
    def group_label_shift_one_hot(self, group, noise_rate=0.2, random_state=0, target=0):
        # Safety checks
        if not (0 <= noise_rate <= 1):
            raise ValueError("Noise rate must be between 0 and 1.")
        
        rng = np.random.default_rng(random_state)
        
        # find index of given group
        if group == 'all': 
            idx = np.arange(len(self.g_train))
        else:
            idx = np.where((self.g_train == group).all(axis=1))[0]
        
        if target != "*": 
            # if specific target is specified, intersect with target
            idx2 = np.where(self.y_train == target)[0]
            idx = np.intersect1d(idx, idx2)

        # flip label based on target and group 
        to_flip = rng.choice(idx, int(len(idx) * noise_rate), replace=False)
        
        # create poison examples
        self.x_poison = copy.deepcopy(self.x_train)
        self.y_poison = copy.deepcopy(self.y_train)
        self.g_poison = copy.deepcopy(self.g_train)
        if target in [0, 1]: 
            assert (self.y_poison[to_flip] == target).all() 
        
        self.y_poison[to_flip] = 1 - self.y_poison[to_flip]
        return 
            
    def group_label_shift_dense(self, group, group_ind=1, noise_rate=0.2, random_state=0, target=0):
        # Safety checks
        if not (0 <= noise_rate <= 1):
            raise ValueError("Noise rate must be between 0 and 1.")
        
        rng = np.random.default_rng(random_state)
        
        # find index of given group that is 0 
        if group == 'all': 
            idx = np.arange(len(self.g_train))
        else: 
            if group not in self.g_train[:, group_ind]:
                raise ValueError(f"Group {group} not found in g_train.")
            idx = np.where(self.g_train[:, group_ind] == group)[0]
        
        idx2 = np.where(self.y_train == target)[0]
        idx = np.intersect1d(idx, idx2)

        # flip move noise of zero labels in group g to 1
        to_flip = rng.choice(idx, int(len(idx) * noise_rate), replace=False)
        
        # create poison examples
        self.x_poison = self.x_train[:]
        self.y_poison = self.y_train[:]
        self.g_poison = self.g_train[:]
        if target in [0, 1]: 
            assert (self.y_poison[to_flip] == target).all() 
        
        self.y_poison[to_flip] = 1 - self.y_poison[to_flip]
        return 

    # def restore_training(self): 
    #     '''Restore training data to clean version'''
    #     self.x_train = self.x_train_clean.copy()
    #     self.y_train = self.y_train_clean.copy()
    #     self.g_train = self.g_train_clean.copy()


def process_compas(data_path): 
    data_path = 'data/compas-scores-two-years.csv'
    df = pd.read_csv(data_path)
    selected_columns = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
            'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    df = df[selected_columns]
    # Applying the filters specified in https://github.com/propublica/compas-analysis/
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != "O") &
        (df['score_text'] != 'N/A')]

    # convert categorical varianbles to one-hot encoding
    clean_df = pd.DataFrame()
    clean_df['priors_count'] = df['priors_count']
    for column in ['age_cat', 'c_charge_degree', 'race', 'sex']:
        one_hot_cat = pd.get_dummies(df[column], prefix=column)
        clean_df = pd.concat([clean_df, one_hot_cat], axis=1)
    clean_df['race'] = df['race'].map({'African-American': 0, 
                                            'Caucasian': 1,
                                            'Hispanic': 2,
                                            'Other': 3,
                                            'Asian': 3, 'Native American':3})

    g = clean_df['race'].values
    y = df['two_year_recid'].values
    x = clean_df.drop('race', axis=1).values
    return x, y, g
