import optuna
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

def objective(trial):
    #dataset
    data, target = sklearn.datasets.fetch_covtype(return_X_y=True)
    target = list(map(lambda label: int(label)-1,target))
    train_x,valid_x,train_y,valid_y = train_test_split(
        data,
        target,
        test_size = 0.25,
        random_state = 0
    )
    dtrain = lgb.Dataset(train_x, label=train_y)

    #default parameters
    #params = {
    #    "verbosity": -1,
    #}

    #optuna suggest
    params = {
        "verbosity": -1,
        "max_bin": trial.suggest_int("max_bin",10,500),
        #"num_leaves": trial.suggest_int("num_leaves",2,500),
        "num_leaves": trial.suggest_int("num_leaves",300,700),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",2,50),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf",1e-8,10.0,log=True),
        "bagging_fraction": trial.suggest_float("begging_fraction",0.1,1.0),
        "bagging_freq": trial.suggest_int("begging_freq",1,100),
        #"feature_fraction": trial.suggest_float("feature_fraction",0.1,1.0),
        "feature_fraction":trial.suggest_float("feature_fraction",0.6,0.9),
        "lambda_l1": trial.suggest_float("lambda_l1",1e-8,10.0,log=True),
        "lambda_l2": trial.suggest_float("lambda_l2",1e-8,10.0,log=True),
        #"min_gain_to_split": trial.suggest_float("min_gain_to_split",0,10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split",0,2),
        "max_depth": trial.suggest_int("max_depth",2,100),
        #"extra_trees": trial.suggest_categorical("extra_trees",[True,False]),
        "extra_trees":False,
        "path_smooth":trial.suggest_int("path_smooth",0,10)
    }

    gbm = lgb.train(params,dtrain)

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

    return accuracy

study = optuna.create_study(direction="maximize")
#study = optuna.create_study(
#    sampler=optuna.samplers.RandomSampler(),
#    direction="maximize",
#)
study.optimize(objective, n_trials=100)

trial = study.best_trial
print("Best trial: ")
print(f"   Accuracy: {trial.value}")
print("   params: ")
for key,value in trial.params.items():
    print(f"    {key}: {value}")

#optuna.visualization.plot_param_importances(study).show()

#optuna.visualization.plot_slice(
#    study,
#    params=["num_leaves","min_gain_to_split"]
#).show()

optuna.visualization.plot_optimization_history(study).show()
