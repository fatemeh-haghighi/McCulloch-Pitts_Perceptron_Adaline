class Mac_pit_neuron():
    def __init__(self, input_value, weight_value):
        self.inputs = input_value
        self.weights = weight_value
            
    def calculate_output(self):
        out = 0
        for i in range(len(self.inputs)):
            out = out + self.inputs[i] * self.weights[i]
        if out >= 0 :
            return 1
        else:
            return 0


class Adder():
    def __init__(self, input1, input2, carry_in, bias):
        self.input1 = input1
        self.input2 = input2
        self.carry_in = carry_in
        self.bias = bias
    
    def set_input_value(self, in1, in2, carry):
        self.input1 = in1
        self.input2 = in2
        self.carry_in = carry
    
    def calculate_output(self):
        carry_neuron = Mac_pit_neuron([self.input1, self.input2, self.carry_in, self.bias], [1, 1, 1, -2])
        carry_neuron_output = carry_neuron.calculate_output()
        output_neuron = Mac_pit_neuron([self.input1, self.input2, self.carry_in, carry_neuron_output, self.bias]
                                       ,[1, 1, 1, -2, -1])        
        return output_neuron.calculate_output(), carry_neuron_output        



class Full_adder():
    def __init__(self, bit_number):
        self.bit_number = bit_number
        self.neurons = []
        
    def calculation(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        for i in range(self.bit_number):
            self.neurons.append(Adder(0, 0, 0, 1))
        carry = 0
        out = []
        for i in range(self.bit_number):
            out.append(0)
        for i in range(self.bit_number):
            self.neurons[i].set_input_value(input1[i], input2[i], carry)
            out[i] , carry = self.neurons[i].calculate_output()
        return out, carry



def make_full_adder(full_adder, n):
    for in1_2 in range(n):
        for in1_1 in range(n):
            for in2_2 in range(n):
                for in2_1 in range(n):
                    output, carry_out = full_adder.calculation([in1_1, in1_2], [in2_1, in2_2]) 
                    in1 = [in1_1, in1_2]
                    in2 = [in2_1, in2_2]
                    print(in1[::-1], '+', in2[::-1] ," = ", output[::-1], carry_out)


full_adder = Full_adder(2)
make_full_adder(full_adder, 2)