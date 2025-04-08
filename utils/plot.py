import matplotlib.pyplot as plt

def plot_loss(losses, title="Training Loss Curve"):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
