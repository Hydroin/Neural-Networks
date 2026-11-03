#include <cmath>
#include <iostream>
#include <vector>
#include <random>

// Returns 1.0 if the sum is above 0, otherwise 0.0 (the step activation).
double Activate(double Sum)
{
    if (Sum > 0)
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
    double (*Apply)(double Sum);
    double (*Slope)(double Answer);
};

double StepSlope(double)         { return 0.0; }
double SigmoidApply(double Sum)  { return 1.0 / (1.0 + std::exp(-Sum)); }
double SigmoidSlope(double A)    { return A * (1.0 - A); }
double ReluApply(double Sum)     { return Sum > 0 ? Sum : 0.0; }
double ReluSlope(double A)       { return A > 0 ? 1.0 : 0.0; }

// Turns an Activation name into its pair of functions.
const ActivationFns& lookup(Activation A)
{
    static const ActivationFns Step   {Activate,     StepSlope};
    static const ActivationFns Sigmoid{SigmoidApply, SigmoidSlope};
    static const ActivationFns Relu   {ReluApply,    ReluSlope};

    switch (A)
    {
        case Activation::Step:    return Step;
        case Activation::Sigmoid: return Sigmoid;
        case Activation::ReLU:    return Relu;
    }
    return Step;
}

struct Neuron
{
//  Each weight scales its own input, and the bias shifts the total. Using the sum formula.
    std::vector<double> Weights = {0.0, 0.0};
    double Bias         = 0.0;
    double LearningRate = 1.0;
    
//  Weighted sum of the inputs plus the bias, before activation.
    double Sum (const std::vector<double>& Inputs) const
    {
        double Total = Bias;
        for (size_t I = 0; I < Weights.size(); ++I)
        {
            Total += Weights[I] * Inputs[I];
        }
        return Total;
    }

//  The neuron's answer (0 or 1) for these inputs.
    double Fire (const std::vector<double>& Inputs) const
    {
        return Activate(Sum(Inputs));
    }

//  Perceptron rule: nudge each weight and the bias by learningRate * error, if the answer was wrong.
    void Learn (const std::vector<double>& Inputs, double Wanted)
    {
        double Error = Wanted - Fire(Inputs);
        if (Error == 0.0) return;

        for (size_t I = 0; I < Weights.size(); ++I)
        {
            Weights[I] += LearningRate * Error * Inputs[I];
        }
        Bias += LearningRate * Error;
    }
};

// A row of neurons that all receive the same inputs and each give one answer.
struct Layer
{
    std::vector<Neuron> Neurons;
    Activation Trigger = Activation::Sigmoid;

//  Makes neuronCount neurons, each with inputCount weights set to 0, using this activation.
    Layer(size_t NeuronCount, size_t InputCount, Activation Act = Activation::Sigmoid)
    {
        Trigger = Act;
        Neurons.resize(NeuronCount);
        for (size_t I = 0; I < Neurons.size(); ++I)
        {
            Neurons[I].Weights.assign(InputCount, 0.0);
        }
        
    }
    
//  Every neuron's sum goes through the layer's activation; answers come back as one list, in neuron order.
    std::vector<double> forward (const std::vector<double>& Inputs) const
    {
        std::vector<double> Results;
        const ActivationFns& Fns = lookup(Trigger);

        for (size_t A = 0; A < Neurons.size(); ++A)
        {
            double Result = Fns.Apply(Neurons[A].Sum(Inputs));

            Results.push_back(Result);
        }

        return Results;
    }

//  Random starting weights
//  Fills every weight in the layer with a random number in [-range, range]; biases stay 0.
    void Randomise(std::mt19937& Dice, double Range)
    {
        std::uniform_real_distribution<double> Search(-Range, Range);         
        for (size_t B = 0; B < Neurons.size(); ++B)
        {
            for (size_t X = 0; X < Neurons[B].Weights.size(); ++X)
            {
                Neurons[B].Weights[X] = Search(Dice);
            }
            Neurons[B].Bias = 0.0;
        }
    }
};

// ---- step 4: the loss (write these two; they sit above the tests that use them) ----
// Cross-entropy for ONE example: how wrong the answers are, as one number.
//   for each output neuron, with o = its answer and t = what was wanted: -( t*ln(o) + (1-t)*ln(1-o) ), added up.
//   ln(0) is infinite, so keep o a tiny bit away from exactly 0 and 1.
// double crossEntropy(const std::vector<double>& answers, const std::vector<double>& wanted);

// Output blame for ONE example: answer - wanted, one value per output neuron (the sigmoid slope cancels out with cross-entropy).
// std::vector<double> outputBlame(const std::vector<double>& answers, const std::vector<double>& wanted);

// Decision for you: the nine XOR numbers are now needed a third time (testXor, testSigmoid, testLoss).
// Pull them into one helper that sets them on two layers, multiplied by a scale? Or copy them again?
// void setXorWeights(Layer& hidden, Layer& output, double scale);

// Same seed must give identical weights; a different seed must not.
void TestRandomise()
{
//  1. three layers of the same shape: layerA, layerB (same seed as A), layerC (different seed)
    Layer A(2, 2);
    Layer B(2, 2);
    Layer C(2, 2);
//  2. one generator per layer, each built from its own seed (not shared, or they would not start at the same place)
    std::mt19937 FirstRoll(42);
    std::mt19937 SecondRoll(42);
    std::mt19937 ThirdRoll(21);
//  3. randomise each layer with its own generator and the same range
    A.Randomise(FirstRoll, 1.0);
    B.Randomise(SecondRoll, 1.0);
    C.Randomise(ThirdRoll, 1.0);
//     4. compare layerA's weights with layerB's: should be the same
//    bool Match   =  (A.Neurons[0].Weights == B.Neurons[0].Weights)
//                 && (A.Neurons[1].Weights == B.Neurons[1].Weights);
//     5. compare LayerA's Weights with layerC's: should be different
//    bool Compare =  (A.Neurons[0].Weights == C.Neurons[0].Weights)
//                 && (A.Neurons[1].Weights == C.Neurons[1].Weights);
//     6. print what was compared and a clear same / different line for each
    
//    std::cout   << "A vs B: " << (Match ? "Same" : "Different") << "\n"
//                << "A vs C: " << (Compare ? "Same" : "Different") << "\n"
//                << " Weight [0]        " << "  |  Weight [1]" << "\n"
//                << "Neuron A[0]:" << A.Neurons[0].Weights[0] << " | Neuron A[1]:" << A.Neurons[1].Weights[1] << "\n" 
//                << "Neuron B[0]:" << B.Neurons[0].Weights[0] << " | Neuron B[1]:" << B.Neurons[1].Weights[1] << "\n"
//                << "Neuron C[0]:" << C.Neurons[0].Weights[0] << " | Neuron C[1]:" << C.Neurons[1].Weights[1] << "\n";
}

// Hand-set XOR network: a hidden layer of 2 feeds an output layer of 1.
void TestXor()
{
//  1. hidden layer: 2 neurons, 2 inputs each
    Layer Hidden(2, 2, Activation::Step);
  
//  2. output layer: 1 neuron, 2 inputs (the hidden layer's two answers)
    Layer Output(1, 2, Activation::Step);

//  3. set hidden neuron 0 (A): weights and bias
    Hidden.Neurons[0].Bias       = -0.5;
    Hidden.Neurons[0].Weights[0] =  1.0;
    Hidden.Neurons[0].Weights[1] =  1.0;

//  4. set hidden neuron 1 (B): weights and bias
    Hidden.Neurons[1].Bias       = -1.5;
    Hidden.Neurons[1].Weights[0] =  1.1;
    Hidden.Neurons[1].Weights[1] =  1.1;

//  5. set the output neuron: weights and bias
    Output.Neurons[0].Bias       = -0.5;
    Output.Neurons[0].Weights[0] =  1.0;
    Output.Neurons[0].Weights[1] = -1.0;
//  6. what XOR wants for each of the four inputs: {0, 1, 1, 0}
    std::vector<double> Want = {0, 1, 1, 0};
    std::vector<std::vector<double>> LayerInputs = { {0,0}, {0,1}, {1,0}, {1,1} };

//  7. loop over the four inputs. For each one:
    for (size_t I = 0; I < Want.size(); ++I)
    {
//      hiddenAnswers = hidden.forward(input)
        std::vector<double> HiddenAnswers = Hidden.forward(LayerInputs[I]);
//      std::cout << LayerInputs[I][0] << " " << LayerInputs[I][1] << "->" << "Neuron 0: " << HiddenAnswers[0] << " " << "Neuron 1: " << HiddenAnswers[1] << " " << "|" << " Wants " << Want[I] <<"\n";

//      print the input, the final answer and the wanted answer
        std::vector<double> FinalAnswers = Output.forward(HiddenAnswers);
//      std::cout << "Answer:" << FinalAnswers[0] << "\n";
    }
//      FinalAnswers  = Output.Forward(HiddenAnswers)
}

// Sigmoid by hand: its value and slope at 0, then the stage 2 XOR weights with sigmoid layers.
void TestSigmoid()
{
//  1. sigmoid of 0: print sigmoidApply(0.0), then sigmoidSlope of THAT answer (it takes the answer, not the sum)
//  expected: 0.5 and 0.25
//  double SigApply = SigmoidApply(0.0);
//  std::cout << "Sigmoid:" << SigApply << " | Slope " << sigmoidSlope(SigApply) << "\n";

//  2. the same for a big positive and a big negative sum, e.g. 10 and -10: the slope nearly vanishes
//  double NegSig = SigmoidApply(-10.0);
//  double PosSig = SigmoidApply(10.0);
//  std::cout << "-10: " << NegSig << " | Slope " << SigmoidSlope(NegSig) <<"\n"<< "+10: " << PosSig << " | Slope " << SigmoidSlope(PosSig) <<"\n";

//  3. a list of scales to try: 1 then 10
    std::vector<double> Scales = {1, 10};
    std::vector<double> Aim   = {0.0, 1.0, 1.0, 0.0};
    std::vector<std::vector<double>> LayerInputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    for (size_t A = 0; A < Scales.size(); ++A)
    {
        double Scale = Scales[A];
//      4. for each scale: build hidden (2, 2) and output (1, 2) with Activation::Sigmoid
        Layer Hidden(2, 2, Activation::Sigmoid);
        Layer Outputs(1, 2, Activation::Sigmoid);

//      5. set the nine stage 2 numbers (six weights and three biases), each multiplied by the scale

//      Hidden Neurons
        Hidden.Neurons[0].Bias        = -0.5 * Scale;
        Hidden.Neurons[0].Weights[0]  =  1.0 * Scale;
        Hidden.Neurons[0].Weights[1]  =  1.0 * Scale;

        Hidden.Neurons[1].Bias        = -1.5 * Scale;
        Hidden.Neurons[1].Weights[0]  =  1.1 * Scale;
        Hidden.Neurons[1].Weights[1]  =  1.1 * Scale;

//      outputs
        Outputs.Neurons[0].Bias       = -0.5 * Scale;
        Outputs.Neurons[0].Weights[0] =  1.0 * Scale;
        Outputs.Neurons[0].Weights[1] = -1.0 * Scale;
//      6. run the four XOR inputs through hidden then output, as in testXor, and print each final answer with what is wanted
//      expected at scale 1: all four between 0.41 and 0.43 (two rows land on the wrong side of 0.5)
//      expected at scale 10: about 0.007, 0.991, 0.991, 0.007

        for (size_t I = 0; I < Aim.size(); ++I)
        {
            std::vector<double> HiddenAnswers = Hidden.forward(LayerInputs[I]);
            std::cout << LayerInputs[I][0] << " " << LayerInputs[I][1] << "->" << " Neuron 0: " << HiddenAnswers[0] << " | " 
                      << " Neuron 1: " << HiddenAnswers[1] << " |" << " Aim " << Aim[I];

            //      print the input, the final answer and the wanted answer
            std::vector<double> FinalAnswers = Outputs.forward(HiddenAnswers);
            std::cout << " | Output:" << FinalAnswers[0]<< "\n";
        }
    }
}

// The loss on paper numbers, then on the XOR network.
void TestLoss()
{
    // 1. work these out on paper FIRST, then check the code against them (answer o, wanted t -> loss, blame):
    //      o 0.5, t 1 -> 0.693, -0.5     o 0.9, t 1 -> 0.105, -0.1     o 0.1, t 1 -> 2.303, -0.9
    //      o 0.9, t 0 -> 2.303,  0.9     o 0.1, t 0 -> 0.105,  0.1

    // 2. call crossEntropy and outputBlame for each of those five cases and print the results beside the paper numbers

    // 3. the XOR network from testSigmoid at scale 10: run the four inputs and print the average loss over the four. expected: close to 0 (about 0.008)

    // 4. the same network at scale 1: the four answers sit near 0.42, so the average loss is about 0.69, barely better than all answers 0.5

    // 5. (optional) hidden and output built with zero weights: every answer 0.5, so the loss is exactly 0.693 for every row
}

// Single neuron learns AND from zero weights; prints how many rounds it took.
void TrainSingleNeuron()
{
    std::vector<std::vector<double>> Inputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    std::vector<double> Wanted = {0, 0, 0, 1}; // AND
//  std::vector<double> Wanted = {0, 1, 1, 1}; // OR
//  std::vector<double> Wanted = {0, 1, 1, 0}; // XOR
      
    Neuron Neuron;
    int Rounds = 0;

    for (int Round = 0; Round < 1000; ++Round)
    {
        ++Rounds;
        bool Learned = true;

        for (int I = 0; I < 4; ++I)
        {
            Neuron.Learn(Inputs[I], Wanted[I]);
        }

        for (int I = 0; I < 4; ++I)
        {
            if (Neuron.Fire(Inputs[I]) != Wanted[I])
            {
                Learned = false;
            }
        }

        if (Learned) break;
    }

    for (int I = 0; I < 4; ++I)
    {
//  std::cout << Inputs[I][0] << " " << Inputs[I][1] << " -> " << Neuron.Fire(Inputs[I]) << " (Want " << Wanted[I] << ")\n"; 
    }
//  std::cout << "Rounds Taken: " << Rounds << ": W1=" << Neuron.Weights[0] << " W2=" << Neuron.Weights[1] << " Bias=" << Neuron.Bias << "\n";
}

int main()
{
    TestXor();
    TrainSingleNeuron();
    TestRandomise();
    TestSigmoid();
}
