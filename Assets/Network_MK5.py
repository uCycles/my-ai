import os

class Neuron:

    def __init__(self, neuron_type, neuron_activation, previous_layer=None) -> None:

        self.neuron_type: str = neuron_type
        self.neuron_activation: str = neuron_activation

        # Initialise neuron type
        if self.neuron_type == "input":
            self.neuron_output: float = 0.5
        elif self.neuron_type == "output":
            self.previous_layer = previous_layer
            self.neuron_weight: float = 0.5
            self.neuron_inputs: float = []
        elif self.neuron_type == "intermediate":
            self.previous_layer = previous_layer
            self.neuron_weight: float = 0.5
            self.neuron_output: float = 0.5
            self.neuron_input: list = []
            self.bias: float = 0.5
        else:
            raise TypeError("Invalid layer/neuron type!")
        

        # Initialise neuron activation
        if neuron_activation == "relu":
            self.applyActivation = lambda x : max(0,x)
            self.applyActivationDerivative = lambda x: 1 if x > 0 else 0
        else:
            raise TypeError("Invalid neuron activation!")

    def getNeuronInput(self) -> None:
        self.neuron_input: float = 0
        for neuron in self.previous_layer:
            self.neuron_input += neuron.getNeuronOutput()
    
    def getNeuronValue(self):
        return self.neuron_output

    def setNeuronValue(self, value):
        self.neuron_output = value   

class Layer:

    def __init__(self, layer_type, layer_activation, layer_size=1, previous_layer=None) -> None:

        self.layer_type: str = layer_type
        self.layer_size: int = layer_size
        self.layer_activation: str = layer_activation
        self.previousLayer: object = previous_layer
        self.layer: list = []

        if layer_type == "input" and previous_layer is not None:
            raise ValueError("Input layer cannot have a previous_layer argument!")

        for _ in range(self.layer_size):
            self.layer.append(Neuron(self.layer_type, self.layer_activation, self.previousLayer))
    
    def getLayer(self):
        return self.layer        

    def getSize(self):
        return self.layer_size

    def updateLayer(self) -> None:
        pass

    def readPreviousLayer(self):
        pass



class Network:

    def __init__(self) -> None:
        self.network: list = []
        self.network_size: int = 0
        self.built: bool = False

    def addLayer(self, size, activation):
        if self.network_size < 0:
        single_layer: object = Layer(size, activation, type)
        self.network.append()

    def c
    def learn(self, batch_size, epochs):
        def getBatches():
            print("e")
        
        getBatches()

    def getCost(self):
        pass

    def nPrint(self):
        # "nice print"


        numbered_network: list = []
        max_length: int = max(layer.getSize() for layer in self.network)
        out_string: str = "\n"
        
        for i in range(max_length):
            row = list()


    '''
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
    '''


def main():
    myNetwork = Network()
    myNetwork.addLayer(3, "relu", "input")

if __name__ == "__main__":
    main()