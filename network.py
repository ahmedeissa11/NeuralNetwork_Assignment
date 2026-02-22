import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


class SimpleFeedforwardNN:
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        # weights randomly in [-0.5, 0.5]
        self.W1 = np.random.rand(hidden_size, input_size) - 0.5
        self.b1 = np.array([0.5, 0.7])  # as requested
        self.W2 = np.random.rand(output_size, hidden_size) - 0.5
        self.b2 = 0.7

    def forward(self, x):
        # x: shape (input_size,)
        self.z1 = self.W1.dot(x) + self.b1
        self.a1 = tanh(self.z1)
        self.z2 = self.W2.dot(self.a1) + self.b2
        self.a2 = tanh(self.z2)
        return self.a2

    def one_step_backprop(self, x, y, lr=0.1):
        # Forward pass
        y_pred = self.forward(x)

        # Compute loss (MSE / 2)
        loss = 0.5 * (y - y_pred) ** 2

        # Backpropagation (one step)
        # Output layer
        dL_da2 = (y_pred - y)  # derivative of 0.5*(y-y_pred)^2 w.r.t a2
        da2_dz2 = tanh_derivative(self.z2)
        dz2 = dL_da2 * da2_dz2  # shape (1,)

        dW2 = dz2.reshape(-1, 1).dot(self.a1.reshape(1, -1))
        db2 = dz2

        # Hidden layer
        dL_da1 = self.W2.T.dot(dz2).flatten()
        da1_dz1 = tanh_derivative(self.z1)
        dz1 = dL_da1 * da1_dz1

        dW1 = dz1.reshape(-1, 1).dot(x.reshape(1, -1))
        db1 = dz1

        # Update weights and biases
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return float(loss)


def demo():
    # Simple demo with a single input-target pair
    np.random.seed(42)
    nn = SimpleFeedforwardNN()

    x = np.array([0.1, 0.2])
    y = np.array([0.3])

    print("Initial W1:\n", nn.W1)
    print("Initial b1:\n", nn.b1)
    print("Initial W2:\n", nn.W2)
    print("Initial b2:\n", nn.b2)

    loss_before = nn.one_step_backprop(x, y, lr=0.1)
    print("Loss before update:", loss_before)

    # Do another forward to show effect
    y_pred_after = nn.forward(x)
    loss_after = 0.5 * (y - y_pred_after) ** 2

    print("Updated W1:\n", nn.W1)
    print("Updated b1:\n", nn.b1)
    print("Updated W2:\n", nn.W2)
    print("Updated b2:\n", nn.b2)
    print("Loss after update:", float(loss_after))


if __name__ == '__main__':
    demo()
