import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import TensorBoard

def make_tensorboard(set_dir_name=""):
    JST = timezone(timedelta(hours = 9))
    str_now = datetime.now(JST).strftime("%a_%d_%b_%Y_%H_%M_%S")
    directory_name = str_now
    log_dir = "{}_{}".format(set_dir_name, directory_name)
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    
    return tensorboard


def make_trained_model(x_train, y_train):
    # Model
    model = Sequential([
        Conv2D(
            input_shape = (28, 28, 1),
            filters = 32,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        Conv2D(
            filters = 32,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        MaxPooling2D(pool_size = (2, 2)),
        Dropout(0.25),

        Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        MaxPooling2D(pool_size = (2, 2)),
        Dropout(0.25),

        Conv2D(
            filters = 128,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        Conv2D(
            filters = 128,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        MaxPooling2D(pool_size = (2, 2)),
        Dropout(0.25),

        Conv2D(
            filters = 256,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        Conv2D(
            filters = 256,
            kernel_size = (3, 3),
            strides = (1, 1),
            padding = "same",
            activation = "relu"
        ),
        MaxPooling2D(pool_size = (2, 2)),
        Dropout(0.25),

        Flatten(),

        Dense(units = 512, activation = "relu"),
        Dropout(0.5),
        Dense(units = 10, activation = "softmax")
    ])

    model.compile(
        optimizer = SGD(),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        batch_size = 32,
        epochs = 100,
        validation_split = 0.2,
        callbacks = [
            make_tensorboard(set_dir_name = "tensorboards/Keras_Kuzushiji_V1")
        ]
    )

    return model


def write_results(predicts, dirname="results"):
    JST = timezone(timedelta(hours = 9))
    str_now = datetime.now(JST).strftime("%Y%m%d%H%M%S")
    filename = "predicts_{}.txt".format(str_now)

    pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predicts) + 1),
            "Label": predicts
        }
    ).to_csv(os.path.join(dirname, filename), index=None)

    return None


def store_model(model):
    JST = timezone(timedelta(hours = 9))
    str_now = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join("models", str_now)

    # モデル格納用ディレクトリの作成
    os.makedirs(dir_name)

    # モデルを json としてファイルに出力
    json_model = model.to_json()
    json_file_name = os.path.join(dir_name, "model.json")
    with open(json_file_name, "w") as f:
        f.write(json_model)

    # モデルパラメータを h5 形式でファイルに出力
    h5_file_name = os.path.join(dir_name, "model.h5")
    model.save(h5_file_name)

    return None


# Data Load
X_train = np.load("data/kmnist-train-imgs.npz")["arr_0"].astype(np.int32)
y_train = np.load("data/kmnist-train-labels.npz")["arr_0"].astype(np.int32)
X_test = np.load("data/kmnist-test-imgs.npz")["arr_0"].astype(np.int32)

# Cleansing
X_train_scaled  = X_train.reshape(60000, 28, 28, 1) / 255
X_test_scaled   = X_test.reshape(10000, 28, 28, 1)  / 255
Y_train_dummied = np_utils.to_categorical(y_train, 10)

# モデル定義＆学習
model = make_trained_model(X_train_scaled, Y_train_dummied) 

# 訓練データでの精度を出力
train_predicts = np.array([ np.argmax(probs) for probs in model.predict(X_train_scaled) ])
train_accuracy = np.mean(train_predicts == y_train)
print("Train Accuracy: {:.4}".format(train_accuracy))

# 結果を CSV に出力
predicts  = np.array([ np.argmax(probs) for probs in model.predict(X_test_scaled) ])
write_results(predicts = predicts)

# モデルを保存
store_model(model)
