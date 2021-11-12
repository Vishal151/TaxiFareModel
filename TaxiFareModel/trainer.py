import time
import joblib
from termcolor import colored
import mlflow
import pandas as pd
import category_encoders as ce
from TaxiFareModel.data import get_data, clean_data, DIST_ARGS
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash
from TaxiFareModel.utils import compute_rmse, simple_time_tracker
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "test_experiment"


class Trainer():
    ESTIMATOR = "Linear"
    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containing all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X: pandas DataFrame
        :param y: pandas DataFrame
        :param kwargs:
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val =\
                train_test_split(self.X_train, self.y_train, test_size=0.15)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on
        # for mlflow
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        time_features = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
                                      OneHotEncoder(handle_unknown='ignore'))

        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())

        features_encoder = ColumnTransformer([
            ('distance', DistanceTransformer(), list(DIST_ARGS.values())),
            ('time_features', time_features, ['pickup_datetime']),
            ('geohash', pipe_geohash, list(DIST_ARGS.values()))
        ])

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('rgs', self.get_estimator())])

    @simple_time_tracker
    def train(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        self.mlflow_log_metric("rmse_train", rmse_train)
        if self.split:
            rmse_val = self.compute_rmse(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("rmse_val", rmse_val)
            print(colored("rmse train: {} || rmse val: {}".format(rmse_train, rmse_val), "blue"))
        else:
            print(colored("rmse train: {}".format(rmse_train), "blue"))

    def compute_rmse(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)
    
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # model types
    ESTIMATORS = ["Lasso", "Ridge", "Linear", "GBM", "RandomForest"]
    for estimator in ESTIMATORS:
        params = dict(nrows=10_000,
                    estimator=estimator,
                    split=True)
        
        df = get_data(**params)
        df = clean_data(df)
        X_train = df.drop(columns=["fare_amount"])
        y_train = df["fare_amount"]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        t = Trainer(X=X_train, y=y_train, **params)
        t.set_experiment_name('[GB] [LON] [VP] LinearV1')
        del X_train, y_train
        print(colored("############  Training model   ############", "red"))
        t.train()
        print(colored("############  Evaluating model ############", "blue"))
        t.evaluate()
        print(colored("############   Saving model    ############", "green"))
        t.save_model()
