import random
from typing import List
from autodiff.scalar import Scalar

class Module:
    def zero_grad(self) -> None:
        """
        Reset the gradients of all parameters to zero.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Scalar]:
        """
        Return a list of parameters of the module.
        """
        return []

class Neuron(Module):
    def __init__(self, num_inputs: int, use_relu=True):
        """
        Initialize the Neuron with the given number of inputs.

        :param num_inputs: Number of inputs that the neuron will receive
        :param use_relu: Whether to use ReLU activation function or no activation function
        """
        # We randomly initialize the weights of the neuron `self.w`
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(num_inputs)]
        # We initialize the bias `self.b` to 0
        self.b = Scalar(0)
        self.use_relu = use_relu

    def __call__(self, x: List[Scalar]) -> Scalar:
        """
        Forward pass through the neuron. Return a Scalar value, representing the output of the neuron.
        Apply the ReLU activation function if `self.use_relu` is True. Otherwise, use no activation function.
        Hint: Given a Scalar object `s`, you can compute the ReLU of `s` by calling `s.relu()`.

        :param x: List of Scalar values, representing the inputs to the neuron
        """
        # DOne: Implement the forward pass through the neuron.
        # Step 1: Compute the weighted sum of inputs.
        # We perform the dot product of the weights and inputs and add the bias.
        # The expression `wi * xi` performs element-wise multiplication of weights and inputs.
        # The `sum` function accumulates the results and adds the bias `self.b`.

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Step 2: Apply the ReLU activation function if `self.use_relu` is True.
        # The ReLU function returns the input value if it is positive, otherwise it returns 0.
        # We use the `relu` method of the Scalar class to apply ReLU.
        # If `self.use_relu` is False, we simply return the computed weighted sum without any activation.
        
        if self.use_relu:
            out = act.relu()
        else:
            out = act

        # Return the final output which is a Scalar object.
        return out
        

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.use_relu else 'Linear'}Neuron({len(self.w)})"

class FeedForwardLayer(Module):
    def __init__(self, num_inputs: int, num_outputs: int, use_relu: bool):
        """
        Initialize the FeedForwardLayer with the given number of inputs and outputs.

        :param num_inputs: Number of inputs that each neuron in that layer will receive
        :param num_outputs: Number of neurons in that layer
        """
        # Done: Initialize the neurons in the layer. `self.neurons` should be a List of Neuron objects.
        self.neurons = [Neuron(num_inputs, use_relu) for _ in range(num_outputs)]
        

    def __call__(self, x: List[Scalar]) -> List[Scalar]:
        """
        Forward pass through the layer. Return a list of Scalars, where each Scalar is the output of a neuron.

        :return: List of Scalar values, representing the outputs of the layer
        """
        # Compute the output of each neuron in the layer for the given inputs
        return [neuron(x) for neuron in self.neurons]
        return None

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"FeedForwardLayer of [{', '.join(str(n) for n in self.neurons)}]"

class MultiLayerPerceptron(Module):
    def __init__(self, num_inputs: int, num_hidden: List[int], num_outputs: int):
        """
        Initialize the MultiLayerPerceptron with the given architecture.
        Note that num_inputs and num_outputs are integers, while num_hidden is a list of integers.

        :param num_inputs: Number of input features
        :param num_hidden: List of integers, where each integer represents the number of neurons in that hidden layer
        :param num_outputs: Number of output neurons
        """
        # TODO: `self.layers` should be a List of FeedForwardLayer objects.
        self.layers = None
        raise NotImplementedError('Task 2.3 not implemented')

    def __call__(self, x: List[Scalar]) -> List[Scalar]:
        """
        Forward pass through the network.
        Call the first layer with the input x.
        Call each layer after that with the output of the previous layer.

        :param x: List of Scalar values, representing the input features
        """
        raise NotImplementedError('Task 2.3 not implemented')
        return None

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MultiLayerPerceptron of [{', '.join(str(layer) for layer in self.layers)}]"
