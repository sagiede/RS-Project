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


class LastLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim, side_info_vectors):
        super(LastLayer, self).__init__()
        self.w = self.add_weight(name="output_layer_FC_weights",
            shape=(input_dim, output_dim), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(output_dim,), initializer="zeros", trainable=True, name="output_layer_FC_bias_weights")
        self.exapnder_bias = self.add_weight(name="expander_bias", shape=(input_dim,), initializer="zeros", trainable=True)
        side_info_vectors = K.constant(side_info_vectors)
        self.side_info_vectors = side_info_vectors
        self.expanding_weights = self.add_weight(shape=(input_dim, side_info_vectors.shape[0]), initializer="random_normal", name="expander",
                                                 trainable=True)
        self.output_dim = output_dim

    def call(self, inputs, **kwargs):
        expander_bias_extended = K.repeat_elements(K.expand_dims(self.exapnder_bias, 1), self.output_dim, 1)
        side_info_expanded = tf.matmul(self.expanding_weights, self.side_info_vectors) + expander_bias_extended
        return tf.matmul(inputs, self.w + side_info_expanded) + self.b


if __name__ == '__main__':

    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("main_side_info")
    argument_parser.add_argument("secondary_side_info")
    argument_parser.add_argument("rating_table_train_path")
    argument_parser.add_argument("rating_table_test_path")
    argument_parser.add_argument("mid_dim", type=int)

    args = argument_parser.parse_args()

    main_sideinfo_vecs = np.load(args.main_side_info, allow_pickle=True)
    secondary_side_info = np.load(args.secondary_side_info, allow_pickle=True)
    ratings_table_train = np.load(args.rating_table_train_path, allow_pickle=True)
    ratings_table_test = np.load(args.rating_table_test_path, allow_pickle=True)
    # pre process
    ratings_mean = calc_mean_ratings(ratings_table_train)
    y_true_matrix_train, train_bias_matrix = calc_y_matrix(ratings_table_train, ratings_mean)
    y_true_matrix_test, test_bias_matrix = calc_y_matrix(ratings_table_test, ratings_mean)

    in_dim = ratings_table_train.shape[1]
    side_info_dim = main_sideinfo_vecs.shape[1]
    mid_dim = args.mid_dim

    input_ratings = tf.keras.Input(shape=(in_dim,))
    input_side_info = tf.keras.Input(shape=(side_info_dim,))

    x = tf.keras.layers.concatenate([input_ratings, input_side_info])
    x = tf.keras.layers.Dense(mid_dim, input_shape=(in_dim + side_info_dim,), activation='tanh')(x)
    x = tf.keras.layers.concatenate([x, input_side_info], axis=1)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = LastLayer(in_dim, mid_dim + side_info_dim, secondary_side_info.transpose())(x)
    model = tf.keras.Model(inputs=[input_ratings, input_side_info], outputs=x)
    lr = 0.000125
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=rmse_loss, optimizer=optimizer, metrics=[mae_loss])
    model.summary()

    epochs = 200

    test_losses = []
    train_losses = []
    test_losses_mae = []

    for epoch in range(epochs):
        print(epoch)
        if epoch == 140:
            K.set_value(model.optimizer.learning_rate, lr / 2)
        noise = np.random.choice(2, size=[train_bias_matrix.shape[0], train_bias_matrix.shape[1]], p=[0.5, 0.5])
        corrupted = np.where(noise > 0, train_bias_matrix, np.zeros_like(train_bias_matrix))
        history = model.fit([corrupted, main_sideinfo_vecs], y_true_matrix_train, epochs=1, batch_size=64)
        train_losses.append(history.history['loss'][0])
        non_zero_mask = y_true_matrix_test.swapaxes(0, 1)[1].sum(axis=1) > 0
        results = model.evaluate([train_bias_matrix[non_zero_mask], main_sideinfo_vecs[non_zero_mask]], y_true_matrix_test[non_zero_mask], verbose=0)
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
