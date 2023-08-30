from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout

import h5py
import numpy as np
import os
import pickle
import tensorflow as tf
import time


def nn_simple_model(X_train, X_test, y_train):
    classes = np.max(y_train) + 1

    neural_network_model = tf.keras.models.Sequential([
        Dense(units=128, activation="relu", input_shape=(X_train.shape[1],), name="FirstHiddenLayer"),
        Dropout(rate=0.2),
        Dense(units=128, activation="relu"),
        Dropout(rate=0.2),
        Dense(units=64, activation="relu"),
        Dropout(rate=0.2),
        Dense(classes, activation='softmax'),
    ])

    neural_network_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    neural_network_model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        epochs=300,
        batch_size=X_train.shape[0],
        shuffle=True,
        verbose=0,
    )

    prediciotns = neural_network_model.predict(X_test)

    return np.argmax(prediciotns, axis=1)


def mrpprobsnnlda(X_train, X_val, X_test, y_train, y_val, nn_model, rp_dim=10, rp_spaces=50, viz=False):
    classes = len(np.unique(y_train))
    dims = X_test.shape[1]

    pred_lda_val = np.zeros((X_val.shape[0], rp_spaces, classes))
    pred_lda_ts = np.zeros((X_test.shape[0], rp_spaces, classes))

    # Random spaces
    R = np.random.normal(size=(dims, rp_dim * rp_spaces))

    # Projection
    B_tr = np.matmul(X_train, R)
    B_val = np.matmul(X_val, R)
    B_ts = np.matmul(X_test, R)

    # for each subspace, find knn and compute class percentages
    for rp in range(0, rp_spaces):
        d_tr = B_tr[:, (rp * rp_dim) : ((rp + 1) * rp_dim)]
        d_val = B_val[:, (rp * rp_dim) : ((rp + 1) * rp_dim)]
        d_ts = B_ts[:, (rp * rp_dim) : ((rp + 1) * rp_dim)]

        # find k nearest neighbours
        lda = LinearDiscriminantAnalysis()
        lda.fit(d_tr, y_train)

        # add the predictions in a single matrix
        preds_val = lda.predict_proba(d_val)
        pred_lda_val[:, rp : (rp + 1), :] = preds_val[:, None, :]

        preds_ts = lda.predict_proba(d_ts)
        pred_lda_ts[:, rp : (rp + 1), :] = preds_ts[:, None, :]

    pred_lda_val = np.array([i.reshape(rp_spaces * classes) for i in pred_lda_val])
    pred_lda_ts = np.array([i.reshape(rp_spaces * classes) for i in pred_lda_ts])

    prediction = nn_model(X_train=pred_lda_val, X_test=pred_lda_ts, y_train=y_val)

    return prediction


def saver(i, y_true, y_pred, ex_time, idx, algo, results):
    if i == 0:
        results[algo] = {
            "iter": {0: y_pred},
            "labels": {0: y_true},
            "time": {0: ex_time},
            "indices": {0: idx},
            "acc": {0: metrics.accuracy_score(y_true, y_pred)},
            "f1": {0: metrics.f1_score(y_true, y_pred, average="weighted")},
        }
    else:
        results[algo]["iter"][i] = y_pred
        results[algo]["labels"][i] = y_true
        results[algo]["time"][i] = ex_time
        results[algo]["indices"][i] = idx
        results[algo]["acc"][i] = metrics.accuracy_score(y_true, y_pred)
        results[algo]["f1"][i] = metrics.f1_score(y_true, y_pred, average="weighted")


def h5file(fpath):
    name = os.path.basename(fpath).split(".")[0]
    print(name, end="\t")

    f = h5py.File(fpath, "r")
    inData = f["data"]["matrix"][:].transpose()
    inTarget = f["class"]["categories"][:]
    inTarget = np.int32(inTarget) - 1

    if inData.shape[0] != len(inTarget):
        inData = inData.transpose()
        if inData.shape[0] != len(inTarget):
            print("Data ", name, "error! Pls Check!")
            f.close()
            return
    f.close()

    print("Classes", np.unique(inTarget).shape, "(", min(np.unique(inTarget)), max(np.unique(inTarget)), ")")

    return inData, inTarget