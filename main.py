#importing libraries
import pandas as pd
import numpy as np
from striprtf.striprtf import rtf_to_text
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from pprint import pprint as pp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#load json file
with open('algoparams_from_ui.json') as f:
    dic = json.load(f)

#build a class to read the json file and process it
class ml_ui:
    def __init__(self,data):
        self.data = data
        self.design_data = data['design_state_data']
        self.session_info = self.design_data['session_info']
        self.target = self.design_data['target']['target']
        self.type = self.design_data['target']['type']
        self.df = None
    
    #write a method to read csv file
    def read_csv(self):
        self.df = pd.read_csv(self.session_info['dataset'])
        return self.df

    def feature_handling(self):
        feature_dict = self.design_data['feature_handling']
        #create an dataframe
        df = pd.DataFrame()
        #iterate over the feature_dict
        for key,value in feature_dict.items():
            if value['is_selected'] == True: 
                df[value['feature_name']] = self.df[value['feature_name']]
                feature_details = value['feature_details']

                if value['feature_variable_type']=='numerical':

                    #check for rescaling
                    if feature_details['rescaling']=='No rescaling':
                        pass
                    elif feature_details['rescaling']=='Standardization':
                        df[value['feature_name']] = (df[value['feature_name']]-df[value['feature_name']].mean())/df[value['feature_name']].std()
                    elif feature_details['rescaling']=='Normalization':
                        df[value['feature_name']] = (df[value['feature_name']]-df[value['feature_name']].min())/(df[value['feature_name']].max()-df[value['feature_name']].min())
                    elif feature_details['rescaling']=='Logarithmic':
                        df[value['feature_name']] = np.log(df[value['feature_name']])
                
                    #check for missing values
                    if feature_details['missing_values']=='Impute':
                        if feature_details['impute_with']=='Average of values':
                            #if whole column is null then fill with 0
                            if df[value['feature_name']].isnull().sum() == len(df[value['feature_name']]):
                                df[value['feature_name']] = df[value['feature_name']].fillna(feature_details['impute_value'])
                            else:
                                df[value['feature_name']] = df[value['feature_name']].fillna(df[value['feature_name']].mean())
                        elif feature_details['impute_with']=='custom':
                            df[value['feature_name']] = df[value['feature_name']].fillna(feature_details['impute_value'])

                elif value['feature_variable_type']=='text':
                    #text feature handling
                    if feature_details['text_handling']=='Tokenize and hash':
                        #encode the string values to numeric values
                        df[value['feature_name']] = df[value['feature_name']].factorize(sort=True)[0]

            else:
                pass

        self.df = df

    def feature_reduction(self):
        reduction_dict = self.design_data['feature_reduction']
        if reduction_dict['feature_reduction_method']=='Correlation with target':
            #reduction based on correlation with target
            df = self.df.copy(deep=True)
            df.loc[:,self.target] = self.df[self.target]
            corr = df.corr()
            corr = corr[self.target]
            corr = corr.sort_values(ascending=False)
            corr = corr[1:reduction_dict['Correlation with target']['num_of_features_to_keep']+1]
            df = df[corr.index]
            df.loc[:,self.target] = self.df[self.target]
            
        if reduction_dict['feature_reduction_method']=='Tree-based':
            #reduction based on tree-based
            if self.type == 'classification':
                model = ExtraTreesClassifier(n_estimators=reduction_dict['Tree-based']['num_of_trees'],max_depth=reduction_dict['Tree-based']['depth_of_trees'])
            elif self.type == 'regression':
                model = ExtraTreesRegressor(n_estimators=reduction_dict['Tree-based']['num_of_trees'],max_depth=reduction_dict['Tree-based']['depth_of_trees'])
                
            df = self.df.copy(deep=True)
            df.loc[:,self.target] = self.df[self.target]
            model = ExtraTreesClassifier()
            model.fit(df.drop(self.target,axis=1),df[self.target])
            feature_importances = pd.Series(model.feature_importances_,index=df.drop(self.target,axis=1).columns)
            feature_importances = feature_importances.sort_values(ascending=False)
            feature_importances = feature_importances[:reduction_dict['Tree-based']['num_of_features_to_keep']]
            df = df[feature_importances.index]
            df.loc[:,self.target] = self.df[self.target]

        if reduction_dict['feature_reduction_method']=='Principal Component Analysis':
            #reduction based on PCA
            df = self.df.copy(deep=True)
            df.loc[:,self.target] = self.df[self.target]
            pca = PCA(n_components=reduction_dict['Principal Component Analysis']['num_of_features_to_keep'])
            pca.fit(df.drop(self.target,axis=1))
            df = pca.transform(df.drop(self.target,axis=1))
            df = pd.DataFrame(df)
            df.loc[:,self.target] = self.df[self.target]
            
        if reduction_dict['feature_reduction_method']=='No Reduction':
            #no reduction
            df = self.df.copy(deep=True)
            df.loc[:,self.target] = self.df[self.target]
        
        self.df = df

    def get_model_object(self):
        model_dict = self.design_data['algorithms']
        model_obj = {}
        model_obj['regression'] = {}
        model_obj['classification'] = {}
        max_iter = self.design_data['hyperparameters']['max_iterations']
        model_obj['regression']['RandomForestRegressor'] = (RandomForestRegressor(),{'n_estimators':list(np.linspace(model_dict['RandomForestRegressor']['min_trees'],model_dict['RandomForestRegressor']['min_trees'],max_iter).astype(int)),
                                                                                    'max_depth':list(np.linspace(model_dict['RandomForestRegressor']['min_depth'],model_dict['RandomForestRegressor']['max_depth'],max_iter).astype(int)),
                                                                                    'min_samples_leaf':list(np.linspace(model_dict['RandomForestRegressor']['min_samples_per_leaf_min_value'],model_dict['RandomForestRegressor']['min_samples_per_leaf_max_value'],max_iter).astype(int)),
                                                                                    'n_jobs': [None if model_dict['RandomForestRegressor']['parallelism'] == 0 else model_dict['RandomForestRegressor']['parallelism']]})

        model_obj['regression']['GBTRegressor'] = (GradientBoostingRegressor(),{'n_estimators':model_dict['GBTRegressor']['num_of_BoostingStages'],
                                                                                'subsample':list(np.linspace(model_dict['GBTRegressor']['min_subsample'],model_dict['GBTRegressor']['min_subsample'],max_iter)),
                                                                                'max_depth':list(np.linspace(model_dict['GBTRegressor']['min_depth'],model_dict['GBTRegressor']['max_depth'],max_iter).astype(int)),
                                                                                'n_iter_no_change':list(np.linspace(model_dict['GBTRegressor']['min_iter'],model_dict['GBTRegressor']['max_iter'],max_iter).astype(int))})

        model_obj['regression']['LinearRegression'] = (LinearRegression(),{'n_jobs': [None if model_dict['LinearRegression']['parallelism'] == 0 else model_dict['LinearRegression']['parallelism']]})

        model_obj['regression']['RidgeRegression'] = (Ridge(),{'max_iter':list(np.linspace(model_dict['RidgeRegression']['min_iter'],model_dict['RidgeRegression']['max_iter'],max_iter).astype(int)),
                                                                'alpha':list(np.linspace(model_dict['RidgeRegression']['min_regparam'],model_dict['RidgeRegression']['max_regparam'],max_iter))})

        model_obj['regression']['LassoRegression'] = (Lasso(),{'max_iter':list(np.linspace(model_dict['LassoRegression']['min_iter'],model_dict['LassoRegression']['max_iter'],max_iter).astype(int)),
                                                                'alpha':list(np.linspace(model_dict['LassoRegression']['min_regparam'],model_dict['LassoRegression']['max_regparam'],max_iter))})

        model_obj['regression']['ElasticNetRegression'] = (ElasticNet(),{'max_iter':list(np.linspace(model_dict['ElasticNetRegression']['min_iter'],model_dict['ElasticNetRegression']['max_iter'],max_iter).astype(int)),
                                                                        'alpha':list(np.linspace(model_dict['ElasticNetRegression']['min_regparam'],model_dict['ElasticNetRegression']['max_regparam'],max_iter)),
                                                                        'l1_ratio':list(np.linspace(model_dict['ElasticNetRegression']['min_elasticnet'],model_dict['ElasticNetRegression']['min_elasticnet'],max_iter))})
        
        model_obj['regression']['xg_boost'] = (XGBRegressor(),{'n_estimators':[model_dict['xg_boost']['max_num_of_trees']], 'max_depth':model_dict['xg_boost']['max_depth_of_tree'],'learning_rate':model_dict['xg_boost']['learningRate'],
                                                                'subsample':model_dict['xg_boost']['sub_sample'],'colsample_bytree':model_dict['xg_boost']['col_sample_by_tree'],'n_jobs': [None if model_dict['xg_boost']['parallelism'] == 0 else model_dict['xg_boost']['parallelism']],
                                                                'booster': ['dart'] if model_dict['xg_boost']['dart'] == True else [None], 'gamma': model_dict['xg_boost']['gamma'], 'reg_lambda': model_dict['xg_boost']['l2_regularization'], 'reg_alpha': model_dict['xg_boost']['l1_regularization'],
                                                                'random_state': [model_dict['xg_boost']['random_state']]})   

        model_obj['regression']['DecisionTreeRegressor'] = (DecisionTreeRegressor(),{'max_depth':list(np.linspace(model_dict['DecisionTreeRegressor']['min_depth'],model_dict['DecisionTreeRegressor']['max_depth'],max_iter).astype(int)), 
                                                                                    'min_samples_split':model_dict['DecisionTreeRegressor']['min_samples_per_leaf']})
        
        svm_kernel_list = []
        if model_dict['SVM']['linear_kernel'] == True:
            svm_kernel_list.append('linear')
        if model_dict['SVM']['rep_kernel'] == True:
            svm_kernel_list.append('rbf')
        if model_dict['SVM']['polynomial_kernel'] == True:
            svm_kernel_list.append('poly')
        if model_dict['SVM']['sigmoid_kernel'] == True:
            svm_kernel_list.append('sigmoid')
        svm_gamma_list = []
        if model_dict['SVM']['auto'] == True:
            svm_gamma_list.append('auto')
        if model_dict['SVM']['scale'] == True:
            svm_gamma_list.append('scale')

        model_obj['regression']['SVM'] = (SVR(),{'kernel': svm_kernel_list,'gamma': svm_gamma_list,'tol':[model_dict['SVM']['tolerance']],
                                                'C':model_dict['SVM']['c_value'],  'max_iter': [model_dict['SVM']['max_iterations']]})

        SGD_penalty_list = []
        if model_dict['SGD']['use_l1_regularization'] == True:
            SGD_penalty_list.append('l1')
        if model_dict['SGD']['use_l2_regularization'] == True:
            SGD_penalty_list.append('l2')
        if model_dict['SGD']['use_elastic_net_regularization'] == True:
            SGD_penalty_list.append('elasticnet')

        model_obj['regression']['SGD'] = (SGDRegressor(),{'alpha':model_dict['SGD']['alpha_value'],'tol':[model_dict['SGD']['tolerance']],
                                                        'penalty':SGD_penalty_list})

        model_obj['regression']['KNN'] = (KNeighborsRegressor(),{'n_neighbors':model_dict['KNN']['k_value'],'weights':['distance'] if model_dict['KNN']['distance_weighting'] == True else ['uniform'],
                                                                'p':[model_dict['KNN']['p_value']] })

        model_obj['regression']['extra_random_trees'] = (ExtraTreesRegressor(),{'max_depth':model_dict['extra_random_trees']['max_depth'],'min_samples_leaf':model_dict['extra_random_trees']['min_samples_per_leaf'],
                                                                                'max_features':['sqrt','log2']})

        model_obj['regression']['neural_network'] = (MLPRegressor(),{'hidden_layer_sizes':[model_dict['neural_network']['hidden_layer_sizes']],'activation':['identity','logistic','tanh','relu'],
                                                                    'alpha':[model_dict['neural_network']['alpha_value']],'max_iter':[model_dict['neural_network']['max_iterations']],'tol': [model_dict['neural_network']['convergence_tolerance']],
                                                                    'early_stopping':[model_dict['neural_network']['early_stopping']],'solver':[model_dict['neural_network']['solver'].lower()],'shuffle':[model_dict['neural_network']['shuffle_data']],
                                                                    'learning_rate_init':[model_dict['neural_network']['initial_learning_rate']],'beta_1':[model_dict['neural_network']['beta_1']],'beta_2':[model_dict['neural_network']['beta_2']],
                                                                    'epsilon':[model_dict['neural_network']['epsilon']],'momentum':[model_dict['neural_network']['momentum']],'nesterovs_momentum':[model_dict['neural_network']['use_nesterov_momentum']],
                                                                    'power_t':[model_dict['neural_network']['power_t']]})


        model_obj['classification']['RandomForestClassifier'] = (RandomForestClassifier(),{'n_estimators':list(np.linspace(model_dict['RandomForestClassifier']['min_trees'],model_dict['RandomForestClassifier']['min_trees'],max_iter).astype(int)),
                                                                                    'max_depth':list(np.linspace(model_dict['RandomForestClassifier']['min_depth'],model_dict['RandomForestClassifier']['max_depth'],max_iter).astype(int)),
                                                                                    'min_samples_leaf':list(np.linspace(model_dict['RandomForestClassifier']['min_samples_per_leaf_min_value'],model_dict['RandomForestClassifier']['min_samples_per_leaf_max_value'],max_iter).astype(int)),
                                                                                    'n_jobs': [None if model_dict['RandomForestClassifier']['parallelism'] == 0 else model_dict['RandomForestClassifier']['parallelism']]})

        model_obj['classification']['GBTClassifier'] = (GradientBoostingClassifier(),{'n_estimators':model_dict['GBTClassifier']['num_of_BoostingStages'],
                                                                                'subsample':list(np.linspace(model_dict['GBTClassifier']['min_subsample'],model_dict['GBTClassifier']['min_subsample'],max_iter)),
                                                                                'max_depth':list(np.linspace(model_dict['GBTClassifier']['min_depth'],model_dict['GBTClassifier']['max_depth'],max_iter).astype(int)),
                                                                                'n_iter_no_change':list(np.linspace(model_dict['GBTClassifier']['min_iter'],model_dict['GBTClassifier']['max_iter'],max_iter).astype(int))})

        model_obj['classification']['LogisticRegression'] = (LogisticRegression(),{'n_jobs': [None if model_dict['LogisticRegression']['parallelism'] == 0 else model_dict['LogisticRegression']['parallelism']],
                                                                                'max_iter':list(np.linspace(model_dict['LogisticRegression']['min_iter'],model_dict['LogisticRegression']['max_iter'],max_iter).astype(int)),
                                                                                'l1_ratio':list(np.linspace(model_dict['LogisticRegression']['min_elasticnet'],model_dict['LogisticRegression']['min_elasticnet'],max_iter))})

        model_obj['classification']['xg_boost'] = (XGBClassifier(),{'n_estimators':[model_dict['xg_boost']['max_num_of_trees']], 'max_depth':model_dict['xg_boost']['max_depth_of_tree'],'learning_rate':model_dict['xg_boost']['learningRate'],
                                                                'subsample':model_dict['xg_boost']['sub_sample'],'colsample_bytree':model_dict['xg_boost']['col_sample_by_tree'],'n_jobs': [None if model_dict['xg_boost']['parallelism'] == 0 else model_dict['xg_boost']['parallelism']],
                                                                'booster': ['dart'] if model_dict['xg_boost']['dart'] == True else [None], 'gamma': model_dict['xg_boost']['gamma'], 'reg_lambda': model_dict['xg_boost']['l2_regularization'], 'reg_alpha': model_dict['xg_boost']['l1_regularization'],
                                                                'random_state': [model_dict['xg_boost']['random_state']]})   

        DTcls_criterion_list = []
        if model_dict['DecisionTreeClassifier']['use_gini'] == True:
            DTcls_criterion_list.append('gini')
        if model_dict['DecisionTreeClassifier']['use_entropy'] == True:
            DTcls_criterion_list.append('entropy')
        model_obj['classification']['DecisionTreeClassifier'] = (DecisionTreeClassifier(),{'criterion':DTcls_criterion_list,'max_depth':list(np.linspace(model_dict['DecisionTreeClassifier']['min_depth'],model_dict['DecisionTreeClassifier']['max_depth'],max_iter).astype(int)), 
                                                                                    'min_samples_split':model_dict['DecisionTreeClassifier']['min_samples_per_leaf']})

        model_obj['classification']['SVM'] = (SVC(),{'kernel': svm_kernel_list,'gamma': svm_gamma_list,'tol':[model_dict['SVM']['tolerance']],
                                                'C':model_dict['SVM']['c_value'],  'max_iter': [model_dict['SVM']['max_iterations']]})

    
        SGD_loss_list = []
        if model_dict['SGD']['use_logistics'] == True:
            SGD_loss_list.append('log_loss')
        if model_dict['SGD']['use_modified_hubber_loss'] == True:
            SGD_loss_list.append('modified_huber')

        model_obj['classification']['SGD'] = (SGDClassifier(),{'alpha':model_dict['SGD']['alpha_value'],'tol':[model_dict['SGD']['tolerance']],
                                                        'penalty':SGD_penalty_list,'loss':SGD_loss_list})
                                                    
        model_obj['classification']['KNN'] = (KNeighborsClassifier(),{'n_neighbors':model_dict['KNN']['k_value'],'weights':['distance'] if model_dict['KNN']['distance_weighting'] == True else ['uniform'],
                                                                'p':[model_dict['KNN']['p_value']]})
                                                            
        model_obj['classification']['extra_random_trees'] = (ExtraTreesClassifier(),{'max_depth':model_dict['extra_random_trees']['max_depth'],'min_samples_leaf':model_dict['extra_random_trees']['min_samples_per_leaf'],
                                                                                'max_features':['sqrt','log2']})

        model_obj['classification']['neural_network'] = (MLPClassifier(),{'hidden_layer_sizes':[model_dict['neural_network']['hidden_layer_sizes']],'activation':['identity','logistic','tanh','relu'],
                                                                    'alpha':[model_dict['neural_network']['alpha_value']],'max_iter':[model_dict['neural_network']['max_iterations']],'tol': [model_dict['neural_network']['convergence_tolerance']],
                                                                    'early_stopping':[model_dict['neural_network']['early_stopping']],'solver':[model_dict['neural_network']['solver'].lower()],'shuffle':[model_dict['neural_network']['shuffle_data']],
                                                                    'learning_rate_init':[model_dict['neural_network']['initial_learning_rate']],'beta_1':[model_dict['neural_network']['beta_1']],'beta_2':[model_dict['neural_network']['beta_2']],
                                                                    'epsilon':[model_dict['neural_network']['epsilon']],'momentum':[model_dict['neural_network']['momentum']],'nesterovs_momentum':[model_dict['neural_network']['use_nesterov_momentum']],
                                                                    'power_t':[model_dict['neural_network']['power_t']]})

        self.model_obj = model_obj
        return self.model_obj
    
    def print_metric(self,model,model_name):
        print('Model: ',model_name)
        
        if self.type == 'classification':
            print('Accuracy: ',accuracy_score(self.y_test,model.predict(self.x_test)))
            print('Precision: ',precision_score(self.y_test,model.predict(self.x_test),average='weighted'))
            print('Precision: ',precision_score(self.y_test,model.predict(self.x_test),average='weighted'))
            print('Recall: ',recall_score(self.y_test,model.predict(self.x_test),average='weighted'))
            print('F1: ',f1_score(self.y_test,model.predict(self.x_test),average='weighted'))
            print('-'*50)

        if self.type == 'regression':
            print('RMSE: ',np.sqrt(mean_squared_error(self.y_test,model.predict(self.x_test))))
            print('MAE: ',mean_absolute_error(self.y_test,model.predict(self.x_test)))
            print('R2: ',r2_score(self.y_test,model.predict(self.x_test)))
            print('-'*50)


    def modelling(self):
        model_dict = self.design_data['algorithms']
        req_models = ['RandomForestRegressor','GBTRegressor','LinearRegression','RidgeRegression','LassoRegression','ElasticNetRegression','xg_boost',
        'DecisionTreeRegressor','SVM','SGD','KNN','extra_random_trees','neural_network']
        cls_models = ['RandomForestClassifier','GBTClassifier','LogisticRegression','xg_boost','DecisionTreeClassifier','SVM','SGD','KNN',
        'extra_random_trees','neural_network']
        model_obj = self.get_model_object()
        hyperparameter = self.design_data['hyperparameters']

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df.drop(self.target,axis=1),self.df[self.target],test_size=0.2,random_state=42)
        if self.type == 'regression':
            for model,value in model_obj['regression'].items():
                if model_dict[model]['is_selected']==True:
                    #hyperparameter tuning using grid search
                    reg = GridSearchCV(value[0],value[1],cv=hyperparameter['num_of_folds'],scoring='neg_root_mean_squared_error',n_jobs=-1)
                    reg.fit(self.x_train,self.y_train)
                    self.print_metric(reg.best_estimator_,model_dict[model]['model_name'])


        if self.type == 'classification':
            for model,value in model_obj['classification'].items():
                if model_dict[model]['is_selected']==True:
                    #hyperparameter tuning using grid search
                    cls = GridSearchCV(value[0],value[1],cv=hyperparameter['num_of_folds'],scoring='f1',n_jobs=-1)
                    cls.fit(self.x_train,self.y_train)
                    self.print_metric(cls.best_estimator_,model_dict[model]['model_name'])
        

if __name__ == '__main__':
    session = ml_ui(dic)
    session.read_csv()
    session.feature_handling()
    session.feature_reduction()
    session.modelling()