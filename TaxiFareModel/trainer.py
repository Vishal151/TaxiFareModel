# imports
import time
import warnings

import category_encoders as ce
import mlflow
from TaxiFareModel.data import get_data, clean_data, DIST_ARGS
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash
from TaxiFareModel.utils import compute_rmse, simple_time_tracker
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "vish"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"


class Trainer(object):
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
        self.split = self.kwargs.get("split", True)  # cf doc above
        self.X_train = X
        self.y_train = y
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val =\
                train_test_split(self.X_train, self.y_train, test_size=0.15)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on

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
        model.set_params(**estimator_params)
        print(model.__class__.__name__)
        return model

    def set_pipeline(self):

        time_features = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
                                      OneHotEncoder(handle_unknown='ignore'))

        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())

        features_encoder = ColumnTransformer([
            ('distance', DistanceTransformer(**DIST_ARGS), list(DIST_ARGS.values())),
            ('time_features', time_features, ['pickup_datetime']),
            ('geohash', pipe_geohash, list(DIST_ARGS.values()))
        ])

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('rgs', self.get_estimator())])

    @simple_time_tracker
    def train(self):
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)

    ### all methods from last exercise above
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
    # Get and clean data
    N = 10000
    df = get_data(nrows=N)
    df = clean_data(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)

    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, estimator="RandomForest")
    t.train()
