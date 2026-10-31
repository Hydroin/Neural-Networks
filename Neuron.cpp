#include <iostream>
#include <array>
#include <vector>

struct Neuron
{
    // Each weight scales its own input, and the bias shifts the total. Using the sum formula.
    std::vector<double> weights = {0.0, 0.0};
    double bias         = 0.0;
    double learningRate = 1.0;
    
    double sum (const std::array<double, 2>& inputs) const
    {
        return weights[0] * inputs[0] + weights[1] * inputs[1] + bias;
    }

    int fire (const std::array<double, 2>& inputs) const
    {
        return (sum(inputs) > 0 ? 1 : 0);
    }

    void learn (const std::array<double, 2>& inputs, int wanted)
    {
        int error = wanted - fire(inputs);
        weights[0] += learningRate * error * inputs[0];
        weights[1] += learningRate * error * inputs[1];
        bias     += learningRate * error;
    }
};

int main()
{
    std::array<std::array<double, 2>, 4> inputs = {{ {0,0}, {0,1}, {1,0}, {1,1} }};
    std::array<int, 4> wanted = {0, 0, 0, 1}; // AND
    // std::array<int, 4> wanted = {0, 1, 1, 1}; // OR
    // std::array<int, 4> wanted = {0, 1, 1, 0}; // XOR
    
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