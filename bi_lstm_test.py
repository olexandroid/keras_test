# бібліотеки
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, GRU, SimpleRNN, Convolution1D, MaxPooling1D, Bidirectional
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint

# конфіги
max_features = 20000
maxlen = 100
embedding_size = 128

dropout = 0.25

filter_length = 5
nb_filter = 64
pool_size = 4

lstm_output_size = 70
batch_size=30
epochs = 3

#  тестові дані з бібліотеки тестових даних
print('Loading data ...')
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# нормалізація даних (доповнення нулями щоб вектори були однакової розмірності)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# проектування архітектури нейронної мережі, шарів алгоритмічної обробки даних чи шарів нейронів
print('Build model ...')
model = Sequential([
    Embedding(max_features, embedding_size),
    Convolution1D(nb_filter, filter_length, padding = 'valid', activation = 'relu', strides = 1 ),
    MaxPooling1D(pool_size = pool_size),
    Bidirectional(LSTM(lstm_output_size)),
    Dense(1),
    Dropout(dropout),
    Activation('sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# навчання
print('Train ...')
model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test, Y_test), callbacks=[ModelCheckpoint('weights/bi_lstm.keras', save_best_only = True)])

# тестування та статистика
score, acc = model.evaluate(X_test, Y_test, batch_size = batch_size)
print('Test score:', score)
print('Test acc:', acc)
