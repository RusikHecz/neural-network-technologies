# -*- coding: utf-8 -*-
# Single-run script: loads dataset CSV, trains autoencoder+regression (Keras),
# and exports encoder/decoder/regression .h5 plus encoded/decoded/regression CSVs.
# Usage (Python 3.9+ recommended):
#   pip install tensorflow keras pandas numpy h5py
#   python pr_v1_t1_train_and_export.py --dataset pr_v1_t1_dataset.csv --epochs 50 --batch_size 64

import argparse
import os
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='CSV with 7 features + target')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv(args.dataset)
    assert df.shape[1] == 8, "Expected 7 features + 1 target"
    X = df[[f"f{i}" for i in range(1,8)]].values.astype('float32')
    y = df[df.columns[-1]].values.reshape(-1,1).astype('float32')

    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    Ntr = int(args.train_ratio * N)
    tr_idx, te_idx = idx[:Ntr], idx[Ntr:]
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # Build joint model
    inp = Input(shape=(7,), name="input_7f")
    x = layers.Dense(16, activation="relu")(inp)
    x = layers.Dense(8, activation="relu")(x)
    z = layers.Dense(args.latent_dim, activation=None, name="encoded")(x)

    d = layers.Dense(8, activation="relu")(z)
    d = layers.Dense(16, activation="relu")(d)
    decoded = layers.Dense(7, activation=None, name="decoded")(d)

    r = layers.Dense(8, activation="relu")(z)
    r = layers.Dense(4, activation="relu")(r)
    reg_out = layers.Dense(1, activation=None, name="regression")(r)

    joint = Model(inp, [decoded, reg_out], name="autoencoder_regressor")
    joint.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"decoded": "mse", "regression": "mse"},
        loss_weights={"decoded": args.recon_weight, "regression": args.reg_weight}
    )

    H = joint.fit(
        X_train,
        {"decoded": X_train, "regression": y_train},
        validation_data=(X_test, {"decoded": X_test, "regression": y_test}),
        epochs=args.epochs, batch_size=args.batch_size, verbose=1
    )

    # Build split models
    encoder = Model(inp, z, name="encoder_model")

    z_in = Input(shape=(args.latent_dim,), name="z_in")
    # Recreate decoder path using the trained layers
    dec_dense1 = joint.get_layer(index=5)  # Dense(8, relu) after latent
    dec_dense2 = joint.get_layer(index=6)  # Dense(16, relu) before 'decoded'
    dec_out = joint.get_layer("decoded")(dec_dense2(dec_dense1(z_in)))
    decoder = Model(z_in, dec_out, name="decoder_model")

    regression = Model(inp, reg_out, name="regression_model")

    # Export models
    enc_path = "pr_v1_t1_encoder.h5"
    dec_path = "pr_v1_t1_decoder.h5"
    reg_path = "pr_v1_t1_regression.h5"
    encoder.save(enc_path)
    decoder.save(dec_path)
    regression.save(reg_path)

    # Export CSVs
    Z_all = encoder.predict(X, batch_size=args.batch_size, verbose=0)
    X_dec = decoder.predict(Z_all, batch_size=args.batch_size, verbose=0)
    y_pred = regression.predict(X, batch_size=args.batch_size, verbose=0)

    pd.DataFrame(Z_all, columns=[f"z{i}" for i in range(1, args.latent_dim+1)]).to_csv("pr_v1_t1_encoded.csv", index=False)
    pd.DataFrame(X_dec, columns=[f"f{i}_decoded" for i in range(1,8)]).to_csv("pr_v1_t1_decoded.csv", index=False)
    pd.DataFrame(np.hstack([y, y_pred]), columns=["y_true", "y_pred"]).to_csv("pr_v1_t1_regression.csv", index=False)

    # Save training history
    pd.DataFrame(H.history).to_csv("pr_v1_t1_history.csv", index=False)

    print("Done. Saved:")
    print(enc_path, dec_path, reg_path)
    print("pr_v1_t1_encoded.csv", "pr_v1_t1_decoded.csv", "pr_v1_t1_regression.csv")

if __name__ == "__main__":
    main()
