from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def train(model, x_train, y_train):
    return model.fit(x_train, y_train,
            batch_size=100,
            epochs=3,
            verbose=1,
            validation_split=0.3)

def evaluate(model, x_test, y_test):
    print("[INFO] evaluating network...")
    predictions = model.predict(x_test)
    predictions = (predictions > 0.5) 
    print(classification_report(y_test,
        predictions, target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))

def plot_result(history):
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='testing accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

