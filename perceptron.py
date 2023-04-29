import matplotlib.pyplot as plt
import numpy as np
import random

def draw_output(w1, w2, input_output, epochs, errors):

    for values in input_output:

        x1 = values[0]
        x2 = values[1]

        print(f'Inputs: {x1} {x2} \t Predicted output: {predict(x1, x2, w1, w2)}\n')

    print(f'There were {epochs} epochs over the dataset\n')
    print(f'Final calculated weights {w1, w2}')

    ys = np.array(errors)
    xs = [x for x in range(1, epochs+1)]

    plt.plot(xs, ys)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Total error per epoch")
    
    plt.show()

def predict(x1, x2, w1, w2):

    threshold = 1

    sum = x1*w1 + x2*w2

    if sum >= threshold:
        predicted_output = 1
    else:
        predicted_output = 0

    return predicted_output

def train_perceptron(input_output):

    w1, w2 = 0.4, 0.05
    learning_rate = 0.1
    epoch = 0
    max_epochs = 1000
    error = random.randint(1, 2)
    converged = False
    errors = []

    while (not converged and epoch < max_epochs):

        converged = True
        
        total_epoch_error = 0

        for input in input_output:

            x1 = input[0]
            x2 = input[1]
            actual_output = input[2]

            predicted_output = predict(x1, x2, w1, w2)

            error = (actual_output - predicted_output)
            total_epoch_error += error
            
            if error != 0:

                w1 += learning_rate * x1 * error
                w2 += learning_rate * x2 * error

                converged = False

        errors.append(total_epoch_error)
        epoch +=1
        
    if converged or epoch < max_epochs:
        draw_output(w1, w2, input_output, epoch, errors)
    
    else:
        print("These set of inputs are not linearly separable so perceptron can not converge.")

# AND
# train_perceptron([[0,0,0], [0,1,0], [1,0,0], [1,1,1]])

# OR
train_perceptron([[0,0,0], [0,1,1], [1,0,1], [1,1,1]])