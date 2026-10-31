#include <iostream>
#include <array>
#include <vector>

struct Neuron
{
    // Each weight scales its own input, and the bias shifts the total. Using the sum formula.
    std::vector<double> weights = {0.0, 0.0};
    double bias         = 0.0;
    double learningRate = 1.0;
    
    double sum (const std::vector<double>& inputs) const
    {
        double total = bias;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            total += weights[i] * inputs[i];
        }
        return total;
    }

    int fire (const std::vector<double>& inputs) const
    {
        return (sum(inputs) > 0 ? 1 : 0);
    }

    void learn (const std::vector<double>& inputs, int wanted)
    {
        int error = wanted - fire(inputs);
        if (error == 0) return;

        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] += learningRate * error * inputs[i];
        }
        bias += learningRate * error;
    }
};

int main()
{
    std::vector<std::vector<double>> inputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    std::vector<int> wanted = {0, 0, 0, 1}; // AND
    // std::vector<int> wanted = {0, 1, 1, 1}; // OR
    // std::vector<int> wanted = {0, 1, 1, 0}; // XOR
    
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