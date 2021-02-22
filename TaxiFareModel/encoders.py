import pandas as pd
import numpy as np
from scipy.spatial import minkowski_distance
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
import geohash as gh

dist_args = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extract the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'
    """

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']].reset_index(drop=True)


class AddGeohash(BaseEstimator, TransformerMixin):

    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash_pickup'] = X.apply(
            lambda x: gh.encode(x.pickup_latitude, x.pickup_longitude, precision=self.precision), axis=1)
        X['geohash_dropoff'] = X.apply(
            lambda x: gh.encode(x.dropoff_latitude, x.dropoff_longitude, precision=self.precision), axis=1)
        return X[['geohash_pickup', 'geohash_dropoff']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Compute the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'
    """

    def __init__(self,
                 start_lat="start_lat",
                 start_lon="start_lon",
                 end_lat="end_lat",
                 end_lon="end_lon",
                 distance_type="haversine"):
        self.distance_type = distance_type
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "haversine":
            X["distance"] = haversine_vectorized(X, **dist_args)
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance()
        return X[["distance"]]
