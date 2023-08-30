from sklearn.model_selection import train_test_split

import __utilities as ut
import numpy as np
import pickle
import time
import warnings

warnings.simplefilter("ignore")


def laod_execution(X, y, iterations=5, rp_dim=10, rp_spaces=50):
    samples = len(y)
    indexies = np.arange(samples)
    results = {}

    for i in range(iterations):
        print("\nIteration: ", i + 1, "out of", iterations)
        tr, tmp = train_test_split(indexies, test_size=0.3)
        val, ts = train_test_split(indexies, test_size=0.333)

        # ============================================================
        tic = time.perf_counter()
        prediction = ut.mrpprobsnnlda(
            X_train=X[tr, :],
            X_val=X[val, :],
            X_test=X[ts, :],
            y_train=y[tr],
            y_val=y[val],
            nn_model=ut.nn_simple_model,
            rp_dim=rp_dim,
            rp_spaces=rp_spaces,
        )
        toc = time.perf_counter()

        ut.saver(i, y[ts], prediction, toc - tic, ts, "nnv_mrpt", results)
        print("NNv-MRPV", results["nnv_mrpt"]["acc"][i])

    return results


if __name__ == "__main__":
    base_path = "data/"

    filenames = [
        'scRNAseq-ZeiselBrainData.h5',
        'scRNAseq-AztekinTailData.h5',
    ]

    for data in filenames:
        data_file = base_path + data

        X, y = ut.h5file(data_file)

        results = laod_execution(X, y, iterations=50, rp_dim=30, rp_spaces=120)

        with open("res/" + data + ".pkl", "wb") as outf:
            pickle.dump(results, outf)

    print("End of experiment!\n")
