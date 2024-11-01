from neural_net import *
from dataloader import *
from matplotlib import pyplot as plt

def plot_image(x, net):
    plt.gray()
    plt.imshow(x.reshape((28, 28)) * 255)
    plt.show()
    y = net.predict(x)
    pred = y.argmax()
    print(f"Model prediction: {pred} ({y[pred]:2.2%} sure)")

def plot_history(history):
    x_points = np.arange(1, len(history["loss"]) + 1)
    p1 = plt.subplot(1, 2, 1)
    p1.set_title("Loss")
    p1.plot(x_points, history["loss"], label="Training")
    p1.plot(x_points, history["val_loss"], label="Validation")
    p1.legend()
    p2 = plt.subplot(1, 2, 2)
    p2.set_title("Accuracy")
    p2.plot(x_points, history["acc"], label="Training")
    p2.plot(x_points, history["val_acc"], label="Validation")
    p2.legend()
    plt.show()

def main():
    net = NeuralNetwork([
        Layer(16, relu, 784),
        Layer(16, relu),
        Layer(16, relu),
        Layer(10, softmax)
    ], loss=categorical_crossentropy)
    x, y = loadtrain()
    print(f"Loaded {x.shape[0]} images")
    history = net.train(x, y, epochs=20, validation_split=0.25)
    plot_history(history)
    print("Finished training")
    for i in range(10):
        plot_image(x[i], net)
    print("Done")

if __name__ == "__main__":
    main()
