class Neuron:
    def __init__(self, neuron_type="", neuron_activation="") -> None:

        self.neuron_type: str = neuron_type
        self.neuron_activation: str = neuron_activation

        # Initialise neuron type
        if self.neuron_type == "input":
            self.neuron_output: float = 0.5

        elif self.neuron_type == "output":
            self.neuron_weight: float = 0.5
            self.neuron_inputs: float = []
            self.bias: float = 0.5

        elif self.neuron_type == "intermediate":
            self.neuron_weight: float = 0.5
            self.neuron_output: float = 0.5
            self.neuron_inputs: list = []
            self.bias: float = 0.5

        else:
            raise TypeError("Invalid neuron_type!")
        

        # Initialise neuron activation
        if neuron_activation == "relu":
            self.applyActivation = lambda x : max(0,x)
            self.applyActivationDerivative = lambda x: 1 if x > 0 else 0
        
        else:
            raise TypeError("Invalid neuron_activation!")

    def setNeuronInput(self, previous_layer):
        self.neuron_input = external_neuron.getNeuronOutput()

    def getNeuronOutput(self) -> float:
        return self.neuron_output

class Layer:
    def __init__(self, layer_type="", layer_activation="", layer_size=1) -> None:

        self.layer_type: str = layer_type
        self.layer_size: int = layer_size
        self.layer_activation: str = layer_activation
        self.layer: list = []


        if layer_size < 1:
            raise ValueError("Invalid layer_size!")


        if self.layer_type == "input":
            for _ in range(self.layer_size):
                self.layer.append(Neuron(self.layer_type, self.layer_activation))


        elif self.layer_type == "output":
            pass
        
        elif self.layer_type == "intermediate":
            pass
        
        else:
            raise TypeError("Invalid layer_type!")




    def updateLayer(self) -> None:
        pass



    def readPreviousLayer(self):

    def getCurrentLayer(self) -> list:


class Network:
    def __init__(self) -> None:
        pass

    def learn(self, batch_size, epochs):
        def getBatches():
            print("e")
        
        getBatches()


def main():
    pass

if __name__ == "__main__":
    main()