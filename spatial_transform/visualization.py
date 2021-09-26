import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import History


def show_train_progress(history: History) -> None:
    """
    Shows train progres for simple classification task.
    :param history: History instance returned by tf.keras.model.fit
    :return: None
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.grid()
    plt.ylim(0.9,1)
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.ylim(0, 0.3)
    plt.grid()
    plt.show()