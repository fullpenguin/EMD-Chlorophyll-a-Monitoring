import math
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

INPUT_CSV = ""
OUTPUT_CSV = ""
FEATURE_COLS = list(range(17))
LABEL_COL = 17
BATCH_SIZE = 8
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
NCOMP = 7


def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / np.mean(y_true) * 100 if np.mean(y_true) != 0 else float('nan')
    bias = np.mean(y_pred - y_true)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'nRMSE(%)': nrmse, 'Bias': bias}


def distance_calg(x_train, y_train, x_test_i, y_pred_i, ncomp=7):
    dis_abs = np.abs(y_train - y_pred_i)
    idx = dis_abs.argsort()
    sel_idx = idx[:ncomp]
    sel = np.tile(x_test_i, (ncomp, 1))
    dis_1 = np.sum(np.square(sel - x_train[sel_idx, :]))
    dis_2 = np.sum(np.square(dis_abs[sel_idx])) / 100
    distance = np.sqrt(dis_1 + dis_2)
    return distance


def fused_distance_calg(pred_i, dis_i):
    aa = np.sum(np.square(dis_i))
    prob = (aa - np.square(dis_i)) / (np.sum(aa - np.square(dis_i)) + 1e-08)
    fused_value = np.sum(prob * pred_i)
    return fused_value


def run_emd():
    if not INPUT_CSV or not OUTPUT_CSV:
        raise ValueError("Set INPUT_CSV and OUTPUT_CSV before running.")

    data = pd.read_csv(INPUT_CSV)
    x_all = data.iloc[:, FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = data.iloc[:, LABEL_COL].to_numpy(dtype=np.float32)
    if not np.isfinite(x_all).all() or not np.isfinite(y_all).all():
        raise ValueError("Input features and targets must be finite.")

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.3
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    svm = SVR(kernel="rbf", C=800)
    svm.fit(x_train, y_train)
    svm_prediction = svm.predict(x_test)

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x_train, y_train)
    rf_prediction = rf.predict(x_test)

    adaboost = AdaBoostRegressor(n_estimators=100)
    adaboost.fit(x_train, y_train)
    adaboost_prediction = adaboost.predict(x_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        batch_size=BATCH_SIZE,
        max_iter=EPOCHS,
        n_iter_no_change=EARLY_STOPPING_PATIENCE,
        learning_rate_init=LR,
        alpha=WEIGHT_DECAY,
    )
    mlp.fit(x_train, y_train)
    mlp_prediction = mlp.predict(x_test)

    predictions = np.column_stack(
        (svm_prediction, rf_prediction, adaboost_prediction, mlp_prediction)
    )
    neighbor_count = min(NCOMP, len(x_train))
    distances = np.column_stack(
        [
            [
                distance_calg(
                    x_train, y_train, x_test[i], prediction[i], neighbor_count
                )
                for i in range(len(x_test))
            ]
            for prediction in (
                svm_prediction,
                rf_prediction,
                adaboost_prediction,
                mlp_prediction,
            )
        ]
    )
    fused_predictions = np.array(
        [
            fused_distance_calg(predictions[i], distances[i])
            for i in range(len(x_test))
        ]
    )
    metrics = regression_metrics(y_test, fused_predictions)
    pd.DataFrame([metrics]).to_csv(OUTPUT_CSV, index=False)
    print(metrics)
    return fused_predictions, metrics


if __name__ == "__main__":
    run_emd()
