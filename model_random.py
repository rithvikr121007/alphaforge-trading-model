import numpy as np
from sklearn.linear_model import LinearRegression


class Model:

    def train(self, X, y):

        X = X.sort_values(["asset_id", "timestamp"]).copy()
        X["previous_close"] = (
            X.groupby("asset_id")["close"].shift(1)
        )
        X["return"] = (
            (X["close"] - X["previous_close"])
            / X["previous_close"]
        )
        X["previous_volume"] = (
            X.groupby("asset_id")["volume"].shift(1)
        )
        X["volume_change"] = (
            (X["volume"] - X["previous_volume"])
            / X["previous_volume"]
        )
        X["volatility"] = (
            X.groupby("asset_id")["return"]
            .rolling(3)
            .std()
            .reset_index(level=0, drop=True)
        )
        X["rolling_avg"] = (
            X.groupby("asset_id")["close"]
            .rolling(3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        X["price_vs_average"] = (
            X["close"] / X["rolling_avg"] - 1
        )

        self.features = [
            "return",
            "price_vs_average",
            "volatility",
            "volume_change"
        ]
        valid = X[self.features].notna().all(axis=1)

        X_train = X.loc[valid, self.features]
        y_train = y.loc[X.index[valid]]

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)


    def predict(self, X):

     
        X = X.sort_values(["asset_id", "timestamp"]).copy()

        X["previous_close"] = (
            X.groupby("asset_id")["close"].shift(1)
        )

        X["return"] = (
            (X["close"] - X["previous_close"])
            / X["previous_close"]
        )

        X["previous_volume"] = (
            X.groupby("asset_id")["volume"].shift(1)
        )

        X["volume_change"] = (
            (X["volume"] - X["previous_volume"])
            / X["previous_volume"]
        )

        X["volatility"] = (
            X.groupby("asset_id")["return"]
            .rolling(3)
            .std()
            .reset_index(level=0, drop=True)
        )

        X["rolling_avg"] = (
            X.groupby("asset_id")["close"]
            .rolling(3)
            .mean()
            .reset_index(level=0, drop=True)
        )

        X["price_vs_average"] = (
            X["close"] / X["rolling_avg"] - 1
        )

        current_time = X["timestamp"].max()

        current = (
            X[X["timestamp"] == current_time]
            .sort_values("asset_id")
        )

        predictions = self.model.predict(
            current[self.features]
        )
        predictions = np.clip(
            predictions,
            -1.0,
            1.0
        )

        return predictions
