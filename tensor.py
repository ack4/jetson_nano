#!/usr/bin/env python3

print('start')

from tensorflow.keras.datasets import mnist
#x:手書き数字画像(28×28)、y:（xの画像が表す数字）

#(x_train, y_train):モデルの学習用、(x_test, y_test):モデルの評価用
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plt
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

# 各MNIST画像の上に（タイトルとして）対応するラベルを表示
for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.set_title(str(y_train[i]))
    ax.imshow(x_train[i], cmap='gray')
#plt.show()

from tensorflow.keras.utils import to_categorical
# 入力画像を行列(28x28)からベクトル(長さ784)に変換
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 名義尺度の値をone-hot表現へ変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# モデルの「容器」を作成
model = Sequential()

# 「容器」へ各layer（Dense, Activation）を積み重ねていく（追加した順に配置されるので注意）
# 最初のlayerはinput_shapeを指定して、入力するデータの次元を与える必要がある
model.add(Dense(units=256, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# モデルの学習方法について指定しておく
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

"""モデルの学習¶
構築したモデルで実際に学習を行うには、Sequential.fit関数を用います。この関数は固定長のバッチで学習を行います。
主な引数は次の通りです。
x：学習に使用する入力データ
y：学習に使用する出力データ
batch_size：学習中のパラメータ更新を1回行うにあたって用いるサンプル数（ミニバッチのサイズ）
epochs：学習のエポック数
verbose：学習のログを出力するか（0:しない、1：バーで出力、2:エポックごとに出力）
validation_split/validation_data：検証用に用いるデータの割合（0～１の実数）、または検証用データそのもの（いずれかのみ指定可能）
shuffle：各エポックごとにデータをシャッフルするか
callbacks：訓練中のモデルの挙動を監視できるcallback関数を指定できます
"""
model.fit(x_train, y_train,
          batch_size=1000, epochs=10, verbose=1,
          validation_data=(x_test, y_test))


