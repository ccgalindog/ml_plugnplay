from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge,\
                                    Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout

#def create_neural_network(input_size):
#  model = Sequential()
#  model.add(Dense(128, input_dim=input_size, activation='relu'))
#  model.add(Dropout(0.4))
#  model.add(Dense(64, activation='relu'))
#  model.add(Dropout(0.4))
#  model.add(Dense(1, activation='linear'))

#  return model


def get_default_classification_models():
    """
    Get a list of default classification models.
    Returns:
        list_models: List - List of classification models.
        list_model_names: List - List of names for the models.
    """
    list_models = [XGBClassifier(),
                   LGBMClassifier(),
                   RandomForestClassifier(),
                   LogisticRegression(),
                   SVC(probability=True),
                   KNeighborsClassifier(),
                   GaussianNB()
                  ]
    list_model_names = ['xgboost',
                        'lightgbm',
                        'random_forest',
                        'logistic_regression',
                        'svc',
                        'knn',
                        'naive_bayes'
                        ]
    return list_models, list_model_names


def get_default_regression_models():
    """
    Get a list of default regression models.
    Returns:
        list_models: List - List of regression models.
        list_model_names: List - List of names for the models.
    """
    list_models = [XGBRegressor(),
                   LGBMRegressor(verbose=-1),
                   RandomForestRegressor(),
                   LinearRegression(),
                   Ridge(),
                   Lasso()#,
                   #create_neural_network(df_train_preproc.shape[1]-1),
                  ]
    list_model_names = ['xgboost',
                        'lightgbm',
                        'random_forest',
                        'linear_regression',
                        'ridge',
                        'lasso'#,
                        #'dnn'
                        ]
    return list_models, list_model_names
