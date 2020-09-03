import numpy as np 
import matplotlib.pyplot as plt

def function(value):
    if value >= 0:
        return 1
    else:
        return -1

class Perceptron_neuron():
    def __init__(self, weights, a, bias, epoch, function):
        self.weights = weights
        self.bias = bias
        self.a = a
        self.out = 0
        self.epoch = epoch
        self.update_counter = 0
        self.function = function
    
    def update_rule(self, inputs, t):
        pass

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def get_net_value(self, inputs):
        out = 0
        for i in range(len(inputs)):
            out += self.weights[0] * inputs[i]
            out += self.weights[1] * inputs[i]
        out += self.bias
        self.out = out
        return out

    def get_h_value(self, instance):
        return self.function(self.get_net_value(instance))

    def fit(self, inputs, target):
        if self.update_counter >= self.epoch:
            return 1
        else:
            outputs = []
            for instance in inputs:
                output = self.get_h_value(instance)
                outputs.append(output)
            return np.array_equal(output, target)
            

class Linear_perceptron(Perceptron_neuron):
    def __init__(self, weights, a, bias, epoch, function):
        super().__init__(weights, a, bias, epoch, function)

    def update_rule(self, inputs, t):
        self.update_counter += 1
        for i in range(len(inputs)):
            h = self.get_h_value(inputs[i])
            if h - t[i] != 0:
                self.bias = self.bias + self.a * t[i]
                self.weights = self.weights + self.a * inputs[i] *t[i]

    

class Adeline_perceptron(Perceptron_neuron):
    def __init__(self, weights, a, bias, epoch, function):
        super().__init__(weights, a, bias, epoch, function)
    
    def update_rule(self, inputs, t):
        self.update_counter += 1
        for i in range(len(inputs)):
            net = self.get_net_value(inputs[i])
            h = self.get_h_value(inputs[i])
            if h - t[i] != 0:
                self.bias = self.bias + self.a * (t[i] - net)
                self.weights = self.weights + self.a * inputs[i] * (t[i] - net)


    

# generate first sampling
first_data_sample1 =  np.random.normal(0, 0.5, (100, 2))
second_data_sample1 =  np.random.normal(0, 0.5, (100, 2))

class1_sample1    = first_data_sample1 + 1
class2_sample1    = second_data_sample1 - 1

plt.scatter(class1_sample1[: , 0]  ,class1_sample1[:, 1], color = 'blue')
plt.scatter(class2_sample1[: , 0]  ,class2_sample1[:, 1], color = 'red')
plt.show()

# Linear for first sample
sample1_weights = [0, 0]
class1_target1 = []
class2_target1 = []
for i in range(100):
    class1_target1.append(1)
    class2_target1.append(-1)

sample1 = []
target1 = []
for i in range(100):
    sample1.append(class1_sample1[i])
    target1.append(class1_target1[i])
for i in range(100):
    sample1.append(class2_sample1[i])
    target1.append(class2_target1[i])

# print(Data)

n = Linear_perceptron(sample1_weights, 1, 0.01, 10000, function)
while(n.fit(sample1, target1) != 1):
    n.update_rule(sample1, target1)

result_weights = n.get_weights()
result_bias = n.get_bias()
x1 = 3
y1 = -1 * (x1 * result_weights[0] + result_bias) / result_weights[1]
x2 = -3
y2 = -1 * (x2 * result_weights[0] + result_bias) / result_weights[1]

plt.scatter(class1_sample1[: , 0]  ,class1_sample1[:, 1], label = 'first class', color = 'blue')
plt.scatter(class2_sample1[: , 0]  ,class2_sample1[:, 1], label = 'second class', color= 'red')
plt.plot([x1 , x2], [y1, y2] , label = 'line' , color= 'green')
plt.legend()
plt.title('Linear perceptron')
plt.show()


# Adeline for first sample

sample1_weights_A = [0, 0]
class1_target1_A = []
class2_target1_A = []
for i in range(100):
    class1_target1_A.append(1)
    class2_target1_A.append(-1)

sample1_A = []
target1_A = []
for i in range(100):
    sample1_A.append(class1_sample1[i])
    target1_A.append(class1_target1[i])
for i in range(100):
    sample1_A.append(class2_sample1[i])
    target1_A.append(class2_target1[i])

# print(Data)

neuron_A = Adeline_perceptron(sample1_weights_A, 0.9, 0.01, 100, function)
while(neuron_A.fit(sample1_A, target1_A) != 1):
    neuron_A.update_rule(sample1_A, target1_A)

result_weights_A = neuron_A.get_weights()
result_bias_A = neuron_A.get_bias()
x1_A = 3
y1_A = -1 * (x1_A * result_weights_A[0] + result_bias_A) / result_weights_A[1]
x2_A = -3
y2_A = -1 * (x2_A * result_weights_A[0] + result_bias_A) / result_weights_A[1]

plt.scatter(class1_sample1[: , 0]  ,class1_sample1[:, 1], label = 'first class', color = 'blue')
plt.scatter(class2_sample1[: , 0]  ,class2_sample1[:, 1], label = 'second class', color= 'red')
plt.plot([x1_A , x2_A], [y1_A, y2_A] , label = 'line' , color= 'green')
plt.legend()
plt.title('Adeline perceptron')
plt.show()

# generate second sampling
first_data_sample2 = np.random.normal(0, 0.5, (100, 2))
second_data_sample2 = np.random.normal(0, 0.5, (10, 2))

class1_sample2 = first_data_sample2 + 1
class2_sample2 = second_data_sample2 - 1

plt.scatter(class1_sample2[: , 0]  ,class1_sample2[:, 1], color = 'blue')
plt.scatter(class2_sample2[: , 0]  ,class2_sample2[:, 1], color = 'red')
plt.show()




# print(len(class1_sample2))
# print(len(class2_sample2))

sample2_weights = [0, 0]
class1_target2 = []
class2_target2 = []
for i in range(100):
    class1_target2.append(1)
    class2_target2.append(-1)

sample2 = []
target2 = []
for i in range(100):
    sample2.append(class1_sample2[i])
    target2.append(class1_target2[i])
for i in range(10):
    sample2.append(class2_sample2[i])
    target2.append(class2_target2[i])

neuron = Linear_perceptron(sample2_weights, 0.5, 0.0, 10000, function)
while(neuron.fit(sample2, target2) != 1):
    neuron.update_rule(sample2, target2)

result_weights2 = neuron.get_weights()
print(result_weights2)
result_bias2 = neuron.get_bias()
x1 = 3
y1 = -1 * (x1 * result_weights2[0] + result_bias2) / result_weights2[1]
x2 = -3
y2 = -1 * (x2 * result_weights2[0] + result_bias2) / result_weights2[1]

plt.scatter(class1_sample2[: , 0]  ,class1_sample2[:, 1], label = 'first class', color = 'blue')
plt.scatter(class2_sample2[: , 0]  ,class2_sample2[:, 1], label = 'second class', color= 'red')
plt.plot([x1 , x2], [y1, y2] , label = 'line' , color= 'green')
plt.legend()
plt.title('Linear perceptron')
plt.show()


sample2_weights_A = [0, 0]
class1_target2_A = []
class2_target2_A = []
for i in range(100):
    class1_target2_A.append(1)
    class2_target2_A.append(-1)

sample2_A = []
target2_A = []
for i in range(100):
    sample2_A.append(class1_sample2[i])
    target2_A.append(class1_target2_A[i])
for i in range(10):
    sample2_A.append(class2_sample2[i])
    target2_A.append(class2_target2_A[i])

neuron2_A = Adeline_perceptron(sample2_weights_A, 0.5, 0.0, 500, function)
while(neuron2_A.fit(sample2_A, target2_A) != 1):
    neuron2_A.update_rule(sample2_A, target2_A)

result_weights2_A = neuron2_A.get_weights()
result_bias2_A = neuron2_A.get_bias()
x1_A = 3
y1_A = -1 * (x1_A * result_weights2_A[0] + result_bias2_A) / result_weights2_A[1]
x2_A = -3
y2_A = -1 * (x2_A * result_weights2_A[0] + result_bias2_A) / result_weights2_A[1]

plt.scatter(class1_sample2[: , 0]  ,class1_sample2[:, 1], label = 'first class', color = 'blue')
plt.scatter(class2_sample2[: , 0]  ,class2_sample2[:, 1], label = 'second class', color= 'red')
plt.plot([x1_A , x2_A], [y1_A, y2_A] , label = 'line' , color= 'green')
plt.legend()
plt.title('Adeline perceptron')
plt.show()







