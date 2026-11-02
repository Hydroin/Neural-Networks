#include <iostream>
#include <vector>

// Returns 1.0 if the sum is above 0, otherwise 0.0 (the step activation).
double activate(double sum)
{
    if (sum > 0)
    {
        return 1.0;
    }
    return 0.0;
}

// One neuron: weights scale the inputs, bias shifts the total, activate decides 0 or 1.
struct Neuron
{
    // Each weight scales its own input, and the bias shifts the total. Using the sum formula.
    std::vector<double> weights = {0.0, 0.0};
    double bias         = 0.0;
    double learningRate = 1.0;
    
    // Weighted sum of the inputs plus the bias, before activation.
    double sum (const std::vector<double>& inputs) const
    {
        double total = bias;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            total += weights[i] * inputs[i];
        }
        return total;
    }

    // The neuron's answer (0 or 1) for these inputs.
    double fire (const std::vector<double>& inputs) const
    {
        return activate(sum(inputs));
    }

    // Perceptron rule: nudge each weight and the bias by learningRate * error, if the answer was wrong.
    void learn (const std::vector<double>& inputs, double wanted)
    {
        double error = wanted - fire(inputs);
        if (error == 0.0) return;

        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] += learningRate * error * inputs[i];
        }
        bias += learningRate * error;
    }
};

// A row of neurons that all receive the same inputs and each give one answer.
struct Layer
{
    std::vector<Neuron> neurons;

    // Makes neuronCount neurons, each with inputCount weights set to 0.
    Layer(size_t neuronCount, size_t inputCount)
    {
        neurons.resize(neuronCount);
        for (size_t i = 0; i < neurons.size(); ++i)
        {
            neurons[i].weights.assign(inputCount, 0.0);
        }
        
    }
    
    // Every neuron fires on the inputs and the answers come back as one list, in neuron order.
    std::vector<double> forward (const std::vector<double>& inputs) const
    {
        std::vector<double> results;

        for (size_t i = 0; i < neurons.size(); ++i)
        {
            double result = neurons[i].fire(inputs);

            results.push_back(result);
        }

        return results;
    }
};

int main()
{
// ==================== New Neuron Layer test ==========================

Layer layer(2, 2);

layer.neurons[0].bias       = -1.0;
layer.neurons[0].weights[0] =  2.0;
layer.neurons[0].weights[1] = -1.0;

layer.neurons[1].bias       =  0.0;
layer.neurons[1].weights[0] = -1.0;
layer.neurons[1].weights[1] =  1.0;

// loops with 4 test inputs
std::vector<std::vector<double>> layerInputs = { {0,0}, {0,1}, {1,0}, {1,1} }; 

for (size_t i = 0; i < layerInputs.size(); ++i)
{
    std::vector<double> results = layer.forward(layerInputs[i]);

    std::cout << layerInputs[i][0] << " " << layerInputs[i][1] << " -> " <<"Neuron 0:" << results[0] << " " << "Neuron 1:" << results[1] << "\n";
}

std::cout << "Neuron 0: w1=" << layer.neurons[0].weights[0] << " w2=" << layer.neurons[0].weights[1] << " bias=" << layer.neurons[0].bias << "\n";
std::cout << "Neuron 1: w1=" << layer.neurons[1].weights[0] << " w2=" << layer.neurons[1].weights[1] << " bias=" << layer.neurons[1].bias << "\n";

// ==================== XOR network test ==========================

    // 1. hidden layer: 2 neurons, 2 inputs each
    Layer hidden(2, 2);
  
    // 2. output layer: 1 neuron, 2 inputs (the hidden layer's two answers)
    Layer output(1, 2);

    // 3. set hidden neuron 0 (A): weights and bias
    hidden.neurons[0].bias       = -0.5;
    hidden.neurons[0].weights[0] =  1.0;
    hidden.neurons[0].weights[1] =  1.0;

    // 4. set hidden neuron 1 (B): weights and bias
    hidden.neurons[1].bias       = -1.5;
    hidden.neurons[1].weights[0] =  1.1;
    hidden.neurons[1].weights[1] =  1.1;

    // 5. set the output neuron: weights and bias
    output.neurons[0].bias       = -0.5;
    output.neurons[0].weights[0] =  1.0;
    output.neurons[0].weights[1] = -1.0;
    // 6. what XOR wants for each of the four inputs: {0, 1, 1, 0}
    std::vector<double> xorwants = {0, 1, 1, 0};

    // 7. loop over the four inputs. For each one:
    for (size_t i = 0; i < xorwants.size(); ++i)
    {
        //      hiddenAnswers = hidden.forward(input)
        std::vector<double> hiddenAnswers = hidden.forward(layerInputs[i]);
        std::cout << layerInputs[i][0] << " " << layerInputs[i][1] << "->" << "Neuron 0: " << hiddenAnswers[0] << " " << "Neuron 1: " << hiddenAnswers[1] << " " << "\n";

        //      print the input, the final answer and the wanted answer
        std::vector<double> finalAnswers = output.forward(hiddenAnswers);
        std::cout << "Answer:" << finalAnswers[0] << "\n";
    }
    //      finalAnswers  = output.forward(hiddenAnswers)
    


// ================ Old Single Neuron training test =========================
    std::vector<std::vector<double>> inputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    std::vector<double> wanted = {0, 0, 0, 1}; // AND
    // std::vector<double> wanted = {0, 1, 1, 1}; // OR
    // std::vector<double> wanted = {0, 1, 1, 0}; // XOR
      
    Neuron neuron;
    int rounds = 0;

    for (int round = 0; round < 1000; ++round)
    {
        ++rounds;
        bool learned = true;

        for (int i = 0; i < 4; ++i)
        {
            neuron.learn(inputs[i], wanted[i]);
        }

        for (int i = 0; i < 4; ++i)
        {
            if (neuron.fire(inputs[i]) != wanted[i])
            {
                learned = false;
            }
        }

        if (learned) break;
    }

    for (int i = 0; i < 4; ++i)
    {
        std::cout << inputs[i][0] << " " << inputs[i][1] << " -> " << neuron.fire(inputs[i]) << " (want " << wanted[i] << ")\n"; 
    }
    std::cout << "Rounds taken: " << rounds << ": w1=" << neuron.weights[0] << " w2=" << neuron.weights[1] << " bias=" << neuron.bias << "\n";
    
}