import numpy as np

def ReLU(X):
    """
    ReLU function activation
    Params:
        X as matrix
    Return:
        X if X > 0 else 0
    """
    return(np.maximum(0, X))

def Sigmoid(X):
    """
    Sigmoid function activation
    Params:
        X as matrix
    Return:
        1 / (1 + np.exp(-X))
    """
    return 1 / (1 + np.exp(-X))

def Tanh(X):
    """
    Hyperbolic tangent function activation
    Params:
        X as matrix
    Return:
        (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    """
    return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))


# Test cases
if __name__ == "__main__":
    X = np.array([-1, 2, 0, -3, 4])

    # Test case for ReLU
    expected_relu = np.array([0, 2, 0, 0, 4])
    result_relu = ReLU(X)
    assert np.array_equal(result_relu, expected_relu), "ReLU test case failed"

    # Test case for Sigmoid
    expected_sigmoid = np.array([0.26894142, 0.88079708, 0.5, 0.04742587, 0.98201379])
    result_sigmoid = Sigmoid(X)
    assert np.allclose(result_sigmoid, expected_sigmoid), "Sigmoid test case failed"

    # Test case for Tanh
    expected_tanh = np.array([-0.76159416, 0.96402758, 0., -0.99505475, 0.9993293])
    result_tanh = Tanh(X)
    assert np.allclose(result_tanh, expected_tanh), "Tanh test case failed"

    print("All test cases passed!")