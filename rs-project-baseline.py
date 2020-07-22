import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
import seaborn as sns
import matplotlib.pyplot as plt
from aux import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

if __name__ == '__main__':

    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("users_text_vectors_path")
    argument_parser.add_argument("rating_table_train_path")
    argument_parser.add_argument("rating_table_test_path")
    argument_parser.add_argument("mid_dim")

    args = argument_parser.parse_args()

    sideinfo_vecs = np.load(args.users_text_vectors_path, allow_pickle=True)
    ratings_table_train = np.load(args.rating_table_train_path, allow_pickle=True)
    ratings_table_test = np.load(args.rating_table_test_path, allow_pickle=True)
    # pre process
    ratings_mean = calc_mean_ratings(ratings_table_train)
    y_true_matrix_train, train_bias_matrix = calc_y_matrix(ratings_table_train, ratings_mean)
    y_true_matrix_test, test_bias_matrix = calc_y_matrix(ratings_table_test, ratings_mean)

    in_dim = ratings_table_train.shape[1]
    side_info_dim = sideinfo_vecs.shape[1]
    mid_dim = args.mid_dim

    input_ratings = tf.keras.Input(shape=(in_dim,))
    input_side_info = tf.keras.Input(shape=(side_info_dim,))

    x = tf.keras.layers.concatenate([input_ratings, input_side_info])
    x = tf.keras.layers.Dense(mid_dim, input_shape=(in_dim + side_info_dim,), activation='tanh')(x)
    x = tf.keras.layers.concatenate([x, input_side_info], axis=1)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(in_dim)(x)
    model = tf.keras.Model(inputs=[input_ratings, input_side_info], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=rmse_loss, optimizer=optimizer, metrics=[mae_loss])
    model.summary()

    epochs = 200

    test_losses = []
    train_losses = []
    test_losses_mae = []

    for epoch in range(epochs):
        noise = np.random.choice(2, size=[train_bias_matrix.shape[0], train_bias_matrix.shape[1]], p=[0.5, 0.5])
        corrupted = np.where(noise > 0, train_bias_matrix, np.zeros_like(train_bias_matrix))
        history = model.fit([corrupted, sideinfo_vecs], y_true_matrix_train, epochs=1, batch_size=64)
        train_losses.append(history.history['loss'][0])
        non_zero_mask = y_true_matrix_test.swapaxes(0, 1)[1].sum(axis=1) > 0
        results = model.evaluate([train_bias_matrix[non_zero_mask], sideinfo_vecs[non_zero_mask]], y_true_matrix_test[non_zero_mask], verbose=0)
        print("test loss (RMSE, MAE):", results[0], results[1])
        test_losses.append(results[0])
        test_losses_mae.append(results[1])
    print("Train losses:", train_losses)
    print("Test losses (RMSE):", test_losses, "Min RMSE", min(test_losses))
    print("Test losses (MAE):", test_losses_mae)
    plt.figure(figsize=(12, 8))
    plt.title("Losses (RMSE)")
    sns.lineplot(np.arange(len(train_losses)), train_losses, label="train")
    sns.lineplot(np.arange(len(test_losses)), test_losses, label="loss")
    plt.show()