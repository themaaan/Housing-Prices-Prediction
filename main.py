import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline 
from category_encoders.target_encoder import TargetEncoder
from skopt import BayesSearchCV
from skopt. space import Real, Categorical, Integer


def read_initial_data():
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")
    return data_train, data_test

def EDA(data_train):
    data_train["SalePrice"].describe()
    data_train["SalePrice"].plot.hist(bins=100)

    plt.show()

def main():
    data_train, data_test = read_initial_data()
    data_train.describe()
    data_train.info()
    # Cols to drop:
    # Alley, MasVnrType, FireplaceQu, PoolQC , Fence, MiscFeature?
    data_train.drop(columns=["Alley", "MasVnrType", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], inplace=True)
    data_test.drop(columns=["Alley", "MasVnrType", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], inplace=True)

    EDA(data_train)
        


    # THE MODEL
    X_train = data_train.drop(columns=["SalePrice"])
    y_train = data_train["SalePrice"]
    
    estimators = [('encoder', TargetEncoder()), 
                  ('clf', XGBRegressor())]
    
    pipe = Pipeline(steps=estimators)
    pipe.set_params(clf__colsample_bynode=0.8)

    print(pipe)

    search_space = {'clf__max_depth': Integer (2,8),
                    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                    'clf__subsample': Real(0.5, 1.0),
                    'clf__colsample_bytree': Real (0.5, 1.0),
                    'clf__colsample_bylevel': Real(0.5, 1.0),
                    'clf__colsample_bynode' : Real(0.5, 1.0),
                    'clf__reg_alpha': Real(0.0, 10.0),
                    'clf__reg_lambda': Real (0.0, 10.0),
                    'clf__gamma': Real(0.0, 10.0)}
    
    opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
    opt.fit(X_train, y_train)
    print(opt.best_params_)
    print(opt.best_estimator_)
    print(opt.best_score_)



if __name__ == "__main__":
    main()
