import pandas as pd
import time
import os
import random

class InputNeuron:

    def __init__(self) -> None:
        self.neuron_value: float = 0.5

    def getValue(self) -> float:
        return self.neuron_value
    
    def setValue(self, new_value) -> None:
        self.neuron_value = new_value


class OutputNeuron:

    def __init__(self, previous_layer) -> None:
        self.neuron_value: float = 0.5
        self.previous_layer: list = previous_layer
        self.weights: list = [random.random() for _ in range(len(previous_layer))]
        self.bias: float = random.random()

    def updateValue(self) -> None:
         # Is Different from intermediate neuron, since there is no activation function (nmv there is its just a relu)

        accumulated_value: int = 0
        relu = lambda x : max(0,x)
                
        for weightIndex, neuron in enumerate(self.previous_layer):
            accumulated_value += (neuron.getValue() * self.weights[weightIndex])

        self.neuron_value = relu(accumulated_value + self.bias) 

    def getNeuronCost(self, expected_value) -> float:
        error: float = self.neuron_value - expected_value
        return error * error

    def getNeuronCostDerivative(self, expected_value):
        return 2 * (self.neuron_value - expected_value)
    
    def getNeuronActivationDerivative(self):
        # this logic is specifically meant for relu, since its derivative is 0 when x < 0 and 1 when x > 0
        if self.neuron_value > 0:
            return 1
        return 0

    def getValue(self) -> float:
        return self.neuron_value
    
    def setValue(self, new_value) -> None:
        self.neuron_value = new_value

    def getWeights(self) -> list:
        return self.weights
    
    def setWeight(self, index, value) -> None:
        self.weights[index] = value

    def getBias(self) -> float:
        return self.bias
        
    def setBias(self, new_bias) -> None:
        self.bias = new_bias


class IntermediateNeuron:

    def __init__(self, previous_layer) -> None:
        self.neuron_value: float = 0.5
        self.previous_layer: list = previous_layer
        self.weights: list = [random.random() for _ in range(len(previous_layer))]
        self.bias: float = random.random()

    def updateValue(self) -> None:

        accumulated_value: int = 0
        relu = lambda x : max(0,x)
                
        for weightIndex, neuron in enumerate(self.previous_layer):
            accumulated_value += (neuron.getValue() * self.weights[weightIndex])

        self.neuron_value = relu(accumulated_value + self.bias) 

    def getValue(self) -> float:
        return self.neuron_value
    
    def setValue(self, new_value) -> None:
        self.neuron_value = new_value

    def getWeights(self) -> list:
        return self.weights
    
    def setWeight(self, index, value) -> None:
        self.weights[index] = value

    def getBias(self) -> float:
        return self.bias
        
    def setBias(self, new_bias) -> None:
        self.bias = new_bias


class Network:

    def __init__(self, num_inputs, intermediate_layer_count, intermediate_layer_density, num_outputs) -> None:
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs
        self.intermediate_layer_count: int = intermediate_layer_count
        self.intermediate_layer_density: int = intermediate_layer_density

        self.dataset_index = 0

        self.print_iteration: int = 0

        # fills an empty network with neuron objects and creates gradients (cost function position)
        self.initializeNetwork()

    def initializeNetwork(self) -> None:

        # the goober
        self.network: list = []


        # create input layer        
        input_layer: list = [InputNeuron() for _ in range(self.num_inputs)]
        self.network.append(input_layer)

        # create intermediate layers
        for layerIndex in range(self.intermediate_layer_count):
            # layer index starts at 0 instead of 1 because layer 0 will always be the input layer (when recalling previous layer)
            previous_layer: list = self.network[layerIndex]
            intermediate_layer: list = [IntermediateNeuron(previous_layer) for _ in range(self.intermediate_layer_density)]
            self.network.append(intermediate_layer)

        # create output layer
        output_layer: list = [OutputNeuron(self.network[-2]) for _ in range(self.num_outputs)]
        self.network.append(output_layer)



        #create an empty gradient for weights and biases
        self.weight_cost_gradient: list = [[0] * self.num_inputs for _ in range(self.num_inputs)] + [[0] * self.intermediate_layer_count for _ in range(self.intermediate_layer_count-1)] + [[0] * self.intermediate_layer_density for _ in range(self.num_outputs)]
        self.bias_cost_gradient: list = [[[0] * self.intermediate_layer_density for _ in range(self.intermediate_layer_density)] for _ in range(self.intermediate_layer_count)] + [[[0] * self.num_outputs for _ in range(self.intermediate_layer_density)]]
        
    def load(self, inputs, outputs) -> None:
        self.input_data: list = inputs
        self.output_data: list = outputs

    def run(self, inputs) -> None:
        if len(inputs) == len(self.network[0]):
            
            # sets all input neurons to their input value
            for i, value in enumerate(inputs):
                self.network[0][i].setValue(value)

            # calls each neuron after the input layers to read their previous layer and form a new value
            weighted_layers: list = self.network.copy()
            weighted_layers = self.network[1:]

            for layer in weighted_layers:
                for neuron in layer:
                    neuron.updateValue()

        else:
            raise("Incorrect amount of inputs")

    def learn(self, learn_rate, batch_size) -> None:
        # IMPORTANT: the first weight in the neuron will always be associated with the first neuron in the previous layer
        # backpropigation starts at the output, hence the backwards loop
        for i in range(batch_size):
            self.runBackpropigation()
            self.applyGradient(learn_rate)
            self.nPrint()

    def runBackpropigation(self):
        #since output is relu, getting the derivative should be either 1 or 0
        #any mention of previous or next should be noted since the network moving right to left
        #this code is awful cause the way the network is set up is stupid, gl


        singe_input: list = self.input_data[self.dataset_index % len(self.input_data)-1]
        single_expected_output: list = self.output_data[self.dataset_index % len(self.output_data)-1]
        self.run(singe_input)
        output_layer: list = self.network[-1]
        penultimate_layer: list = self.network[-2]
        self.dataset_index += 1
        self.previous_partial_derivatives: list = []


        # OUTPUT WEIGHTS
        for i, neuron in enumerate(output_layer):

            #each list will represent one neurons previous partial derivatives
            self.previous_partial_derivatives.append([])
            for j in range(len(neuron.getWeights())):

                partial_derivative: float = 1 * (2*(neuron.getValue() - single_expected_output[i]))
                self.previous_partial_derivatives[i].append(partial_derivative)

                weight_cost: float = penultimate_layer[j].getValue() * partial_derivative

                # last layer of the ith neuron of jth weight
                self.weight_cost_gradient[-1][i][j] = weight_cost


        # INTERMEDIATE WEIGHTS
        #each layer
        for layer_index in range(len(self.network)-2,0,-1):
            #each neuron
            for j, neuron in enumerate(self.network[layer_index]):
                self.previous_partial_derivatives.append([])
                #each weight
                for k, weight in enumerate(neuron.getWeights()):
                    next_neuron = self.network[layer_index-1][k]

                    # hell is real and its inside this method
                    weight_cost: float = next_neuron.getValue() * self.getNeuronCostHistory(self.previous_partial_derivatives, j, self.network[layer_index+1])
                    self.weight_cost_gradient[layer_index][j][k] = weight_cost

    def applyGradient(self, learn_rate):
        working_layers: list = self.network[1:]

        for i, layer in enumerate(working_layers):
            for j, neuron in enumerate(layer):
                for k, weight in enumerate(neuron.getWeights()):

                    new_value: float = weight - self.weight_cost_gradient[i][j][k] * learn_rate
                    neuron.setWeight(k, new_value)

    def getNeuronCostHistory(self, previous_derivatives, neuron_index, previous_layer):
        # made this cause of the "awful" structure

        previous_derivative_cost: list = []
        previous_weight_values: list = []

        for neuron in previous_layer:
            for i, weight in enumerate(neuron.getWeights()):
                if i == neuron_index:
                    previous_weight_values.append(weight)

        for item in previous_derivatives:
            for j, value in enumerate(item):    
                if j == neuron_index:
                    previous_derivative_cost.append(value)

        to_be_summed: list = []

        for i, weight in enumerate(previous_weight_values):
            value: float = weight * previous_derivative_cost[i]
            to_be_summed.append(value)

        history: float = 0

        for item in to_be_summed:
            history += item

        return history

    def getNetworkValues(self) -> list:
        out_list: list = []
        for layerIndex in range(len(self.network)):
            out_list.append([])
            for neuron in self.network[layerIndex]:
                out_list[layerIndex].append(neuron.getValue())
        return out_list

    def nPrint(self, wait=None) -> None:
        # print a easy to read version of the network

        # Convert object to their respective values
        network_values: list = self.getNetworkValues()

        max_length: int = max(len(layer) for layer in network_values)
        out_string: str = "\n"
        
        for i in range(max_length):
            row: list = []
            for layer in network_values:
                if i < len(layer):
                    row.append(str(round(layer[i],3)))
                else:
                    row.append("")
            out_string += "\t\t".join(row) + "\n"

        out_string += f"\nIteration {self.print_iteration}\n"
        self.print_iteration += 1

        # pure magic for all i know
        os.system('cls' if os.name == 'nt' else 'clear')
        print(out_string)

        if wait:
           time.sleep(wait)

    def getOutputValues(self) -> list:
        output_values: list = []
        output_neurons: list = self.network[-1]

        for neuron in output_neurons:
            output_values.append(neuron.getValue())

        return output_values
    


def main():

    df = pd.read_csv('c:/Users/Grant Shimkaveg/Documents/vscodeProjects/my-ai/Assets/student_data.csv')

    #inputs
    student_input_data: list = []
    student_study_time: list = df['StudyTimeWeekly'].tolist()
    student_age: list = df['Age'].tolist()
    student_absences: list = df['Absences'].tolist()
    # the range doest matter since there should be an equal amount of data between the 3
    for i in range(len(student_study_time)):
        student_input_data.append([student_age[i], student_study_time[i], student_absences[i]])

    #expected outputs
    student_output_data: list = df['GPA'].tolist()
    student_output_data = [[i] for i in student_output_data]

    # example takes in the age and study time and returns possible gpa
    myNetwork = Network(3,2,3,1)
    myNetwork.nPrint()
    myNetwork.load(student_input_data, student_output_data)
    myNetwork.learn(0.01,1)
    



if __name__ == "__main__":
    main()