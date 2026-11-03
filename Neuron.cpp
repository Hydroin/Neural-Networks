#include <cmath>
#include <iostream>
#include <vector>
#include <random>

// Returns 1.0 if the sum is above 0, otherwise 0.0 (the step activation).
double activate(double sum)
{
    if (sum > 0)
    {
        return 1.0;
    }
    return 0.0;
}

// Which activation a layer uses. Only lookup() knows what each name means.
enum class Activation { Step, Sigmoid, ReLU };

// One activation's two jobs: apply turns a sum into an answer, slope is worked out from that answer.
struct ActivationFns
{
    double (*apply)(double sum);
    double (*slope)(double answer);
};

double stepSlope(double)         { return 0.0; }
double sigmoidApply(double sum)  { return 1.0 / (1.0 + std::exp(-sum)); }
double sigmoidSlope(double a)    { return a * (1.0 - a); }
double reluApply(double sum)     { return sum > 0 ? sum : 0.0; }
double reluSlope(double a)       { return a > 0 ? 1.0 : 0.0; }

// Turns an Activation name into its pair of functions.
const ActivationFns& lookup(Activation a)
{
    static const ActivationFns step   {activate,     stepSlope};
    static const ActivationFns sigmoid{sigmoidApply, sigmoidSlope};
    static const ActivationFns relu   {reluApply,    reluSlope};

    switch (a)
    {
        case Activation::Step:    return step;
        case Activation::Sigmoid: return sigmoid;
        case Activation::ReLU:    return relu;
    }
    return step;
}

// (Skeleton from the stage 3 page, commented out until each piece is written)
// // The loss
// // Cross-entropy for one example: how wrong the answers are, as one number.
// double crossEntropy(const std::vector<double>& answers, const std::vector<double>& wanted);

// // Output blame for one example: answer - wanted, one value per output neuron.
// std::vector<double> outputBlame(const std::vector<double>& answers, const std::vector<double>& wanted);

// // What a layer remembers, and its backward step
// // Layer gets: lastInputs, lastAnswers (saved by forward, which is no longer const),
// //             weightNudges (one per weight) and biasNudges (one per neuron).
// // Given the blame at this layer's sums, works out and STORES the nudges (does not apply them),
// // and returns the blame arriving at this layer's inputs (the previous layer's answers).
// std::vector<double> Layer::backward(const std::vector<double>& blameAtSums);
// // This layer's activation slope for each of its last answers (the hidden-blame step multiplies by it).
// std::vector<double> Layer::slopes() const;

// // Applying, and the Network
// // Adds learningRate * nudge to every weight and bias, then clears the stored nudges.
// void Layer::applyNudges(double learningRate);

// struct Network
// {
// std::vector<Layer> layers;                                   // in order, input side first
// std::vector<double> forward(const std::vector<double>& in);   // chains the layers
// // One example: forward, outputBlame, walk the layers backward (multiply the blame going
// // into a hidden layer by that layer's slopes()), then applyNudges on every layer (online).
// void trainExample(const std::vector<double>& in, const std::vector<double>& wanted, double learningRate);
// };

// // ---- tests, each a void function called from main ----

// void testLoss();           // times-10 XOR weights: close to 0. Random start: about 0.7
// void testGradientCheck();  // nudge one weight by 0.0001 and compare the loss change with your stored nudge
// void trainXor();           // fixed seed, 4 hidden sigmoid neurons, prints rounds and final loss
// One neuron: weights scale the inputs, bias shifts the total, activate decides 0 or 1.

struct Neuron
{
//  Each weight scales its own input, and the bias shifts the total. Using the sum formula.
    std::vector<double> weights = {0.0, 0.0};
    double bias         = 0.0;
    double learningRate = 1.0;
    
//  Weighted sum of the inputs plus the bias, before activation.
    double sum (const std::vector<double>& inputs) const
    {
        double total = bias;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            total += weights[i] * inputs[i];
        }
        return total;
    }

//  The neuron's answer (0 or 1) for these inputs.
    double fire (const std::vector<double>& inputs) const
    {
        return activate(sum(inputs));
    }

//  Perceptron rule: nudge each weight and the bias by learningRate * error, if the answer was wrong.
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
    Activation activation = Activation::Sigmoid;

//  Makes neuronCount neurons, each with inputCount weights set to 0, using this activation.
    Layer(size_t neuronCount, size_t inputCount, Activation act = Activation::Sigmoid)
    {
        activation = act;
        neurons.resize(neuronCount);
        for (size_t i = 0; i < neurons.size(); ++i)
        {
            neurons[i].weights.assign(inputCount, 0.0);
        }
        
    }
    
//  Every neuron's sum goes through the layer's activation; answers come back as one list, in neuron order.
    std::vector<double> forward (const std::vector<double>& inputs) const
    {
        std::vector<double> results;
        const ActivationFns& fns = lookup(activation);

        for (size_t a = 0; a < neurons.size(); ++a)
        {
            double result = fns.apply(neurons[a].sum(inputs));

            results.push_back(result);
        }

        return results;
    }

//  Random starting weights
//  Fills every weight in the layer with a random number in [-range, range]; biases stay 0.
    void randomise(std::mt19937& dice, double range)
    {
        std::uniform_real_distribution<double> search(-range, range);         
        for (size_t b = 0; b < neurons.size(); ++b)
        {
            for (size_t x = 0; x < neurons[b].weights.size(); ++x)
            {
                neurons[b].weights[x] = search(dice);
            }
            neurons[b].bias = 0.0;
        }
    }
};

// Same seed must give identical weights; a different seed must not.
void testRandomise()
{
//  1. three layers of the same shape: layerA, layerB (same seed as A), layerC (different seed)
    Layer A(2, 2);
    Layer B(2, 2);
    Layer C(2, 2);
//  2. one generator per layer, each built from its own seed (not shared, or they would not start at the same place)
    std::mt19937 firstRoll(42);
    std::mt19937 secondRoll(42);
    std::mt19937 thirdRoll(21);
//  3. randomise each layer with its own generator and the same range
    A.randomise(firstRoll, 1.0);
    B.randomise(secondRoll, 1.0);
    C.randomise(thirdRoll, 1.0);
//     4. compare layerA's weights with layerB's: should be the same
//    bool match   =  (A.neurons[0].weights == B.neurons[0].weights)
//                 && (A.neurons[1].weights == B.neurons[1].weights);
//     5. compare layerA's weights with layerC's: should be different
//    bool compare =  (A.neurons[0].weights == C.neurons[0].weights)
//                 && (A.neurons[1].weights == C.neurons[1].weights);
//     6. print what was compared and a clear same / different line for each
    
//    std::cout   << "A vs B: " << (match ? "same" : "different") << "\n"
//                << "A vs C: " << (compare ? "same" : "different") << "\n"
//                << " Weight [0]        " << "  |  Weight [1]" << "\n"
//                << "Neuron A[0]:" << A.neurons[0].weights[0] << " | Neuron A[1]:" << A.neurons[1].weights[1] << "\n" 
//                << "Neuron B[0]:" << B.neurons[0].weights[0] << " | Neuron B[1]:" << B.neurons[1].weights[1] << "\n"
//                << "Neuron C[0]:" << C.neurons[0].weights[0] << " | Neuron C[1]:" << C.neurons[1].weights[1] << "\n";
}

// Hand-set XOR network: a hidden layer of 2 feeds an output layer of 1.
void testXor()
{
//  1. hidden layer: 2 neurons, 2 inputs each
    Layer hidden(2, 2, Activation::Step);
  
//  2. output layer: 1 neuron, 2 inputs (the hidden layer's two answers)
    Layer output(1, 2, Activation::Step);

//  3. set hidden neuron 0 (A): weights and bias
    hidden.neurons[0].bias       = -0.5;
    hidden.neurons[0].weights[0] =  1.0;
    hidden.neurons[0].weights[1] =  1.0;

//  4. set hidden neuron 1 (B): weights and bias
    hidden.neurons[1].bias       = -1.5;
    hidden.neurons[1].weights[0] =  1.1;
    hidden.neurons[1].weights[1] =  1.1;

//  5. set the output neuron: weights and bias
    output.neurons[0].bias       = -0.5;
    output.neurons[0].weights[0] =  1.0;
    output.neurons[0].weights[1] = -1.0;
//  6. what XOR wants for each of the four inputs: {0, 1, 1, 0}
    std::vector<double> want = {0, 1, 1, 0};
    std::vector<std::vector<double>> layerInputs = { {0,0}, {0,1}, {1,0}, {1,1} };

//  7. loop over the four inputs. For each one:
    for (size_t i = 0; i < want.size(); ++i)
    {
//      hiddenAnswers = hidden.forward(input)
        std::vector<double> hiddenAnswers = hidden.forward(layerInputs[i]);
//      std::cout << layerInputs[i][0] << " " << layerInputs[i][1] << "->" << "Neuron 0: " << hiddenAnswers[0] << " " << "Neuron 1: " << hiddenAnswers[1] << " " << "|" << " Wants " << want[i] <<"\n";

//      print the input, the final answer and the wanted answer
        std::vector<double> finalAnswers = output.forward(hiddenAnswers);
//      std::cout << "Answer:" << finalAnswers[0] << "\n";
    }
//      finalAnswers  = output.forward(hiddenAnswers)
}

// Sigmoid by hand: its value and slope at 0, then the stage 2 XOR weights with sigmoid layers.
void testSigmoid()
{
//  1. sigmoid of 0: print sigmoidApply(0.0), then sigmoidSlope of THAT answer (it takes the answer, not the sum)
//  expected: 0.5 and 0.25
//  double sigApply = sigmoidApply(0.0);
//  std::cout << "Sigmoid:" << sigApply << " | Slope " << sigmoidSlope(sigApply) << "\n";

//  2. the same for a big positive and a big negative sum, e.g. 10 and -10: the slope nearly vanishes
//  double negSig = sigmoidApply(-10.0);
//  double posSig = sigmoidApply(10.0);
//  std::cout << "-10: " << negSig << " | Slope " << sigmoidSlope(negSig) <<"\n"<< "+10: " << posSig << " | Slope " << sigmoidSlope(posSig) <<"\n";

//  3. a list of scales to try: 1 then 10
    std::vector<double> scales = {1, 10};
    std::vector<double> aim   = {0.0, 1.0, 1.0, 0.0};
    std::vector<std::vector<double>> layerInputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    for (size_t a = 0; a < scales.size(); ++a)
    {
        double scale = scales[a];
//      4. for each scale: build hidden (2, 2) and output (1, 2) with Activation::Sigmoid
        Layer hidden(2, 2, Activation::Sigmoid);
        Layer outputs(1, 2, Activation::Sigmoid);

//      5. set the nine stage 2 numbers (six weights and three biases), each multiplied by the scale

//      Hidden Neurons
        hidden.neurons[0].bias        = -0.5 * scale;
        hidden.neurons[0].weights[0]  =  1.0 * scale;
        hidden.neurons[0].weights[1]  =  1.0 * scale;

        hidden.neurons[1].bias        = -1.5 * scale;
        hidden.neurons[1].weights[0]  =  1.1 * scale;
        hidden.neurons[1].weights[1]  =  1.1 * scale;

//      outputs
        outputs.neurons[0].bias       = -0.5 * scale;
        outputs.neurons[0].weights[0] =  1.0 * scale;
        outputs.neurons[0].weights[1] = -1.0 * scale;
//      6. run the four XOR inputs through hidden then output, as in testXor, and print each final answer with what is wanted
//      expected at scale 1: all four between 0.41 and 0.43 (two rows land on the wrong side of 0.5)
//      expected at scale 10: about 0.007, 0.991, 0.991, 0.007

        for (size_t i = 0; i < aim.size(); ++i)
        {
            std::vector<double> hiddenAnswers = hidden.forward(layerInputs[i]);
            std::cout << layerInputs[i][0] << " " << layerInputs[i][1] << "->" << " Neuron 0: " << hiddenAnswers[0] << " | " 
                      << " Neuron 1: " << hiddenAnswers[1] << " |" << " Aim " << aim[i];

            //      print the input, the final answer and the wanted answer
            std::vector<double> finalAnswers = outputs.forward(hiddenAnswers);
            std::cout << " | Output:" << finalAnswers[0]<< "\n";
        }
    }
}

// Single neuron learns AND from zero weights; prints how many rounds it took.
void trainSingleNeuron()
{
    std::vector<std::vector<double>> inputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    std::vector<double> wanted = {0, 0, 0, 1}; // AND
//  std::vector<double> wanted = {0, 1, 1, 1}; // OR
//  std::vector<double> wanted = {0, 1, 1, 0}; // XOR
      
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
//  std::cout << inputs[i][0] << " " << inputs[i][1] << " -> " << neuron.fire(inputs[i]) << " (want " << wanted[i] << ")\n"; 
    }
//  std::cout << "Rounds taken: " << rounds << ": w1=" << neuron.weights[0] << " w2=" << neuron.weights[1] << " bias=" << neuron.bias << "\n";
}

int main()
{
    testXor();
    trainSingleNeuron();
    testRandomise();
    testSigmoid();
}
