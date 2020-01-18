from functions import load_data, train, evaluate, plot_result
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU

(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
model.add(Dense(784, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd', 
    metrics=['accuracy']
)

history = train(model, x_train, y_train)
plot_result(history)
evaluate(model, x_test, y_test)
