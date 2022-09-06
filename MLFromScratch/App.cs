namespace MLFromScratch;

public class App {
    public static void Main() {
        const float LearnRate = 0.1f;

        var network = new NeuralNetwork(new() { 4, 8, 8, 1}); // 4 inputs, 8 -> 8 (hidden layers), 1 output

        // Training inputs:
        var xs = new List<List<float>> {
            new() { 1.0f, 0.0f,  1.0f, 0.2f },
            new() { 1.0f, 0.2f,  1.0f, 0.8f },
            new() { 1.0f, 0.3f,  1.0f, 0.7f },
            new() { 1.0f, 0.5f,  1.0f, 0.3f },
            new() { 1.0f, -1.0f, 1.0f, 0.1f },

            new() { 0.0f, 0.0f,  1.0f, 0.2f },
            new() { 0.0f, 0.2f,  1.0f, 0.8f },
            new() { 0.0f, 0.3f,  1.0f, 0.7f },
            new() { 0.0f, 0.5f,  1.0f, 0.3f },
            new() { 0.0f, -1.0f, 1.0f, 0.1f },
        };

        // Training outputs:
        // We are training it to basically give us the first value from the input.
        var ys = new List<float> {
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,

            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        };

        Console.WriteLine("Training...");

        for (var i = 0; i < 1000; i++) { 
            for (var j = 0; j < xs.Count; j++) { 
                var inputs  = xs[j].Select(x => new Node(x)).ToList();
                var outputs = new Node(ys[j]);

                var prediction = network.F(inputs);
                var loss = MeanErrorSquared(prediction, outputs);

                network.ZeroGradients();
                loss.Backward();

                foreach (var parameter in network.Parameters()) {
                    parameter.Value += -LearnRate * parameter.Gradient;
                }

                if (i % 100 == 0 && j == 0) {
                    Console.WriteLine($"Loss: {loss.Value}");
                }
            }
        }

        Console.WriteLine("Training done!");
        Console.WriteLine();

        Predict(network, xs[0], ys[0]);
        Predict(network, xs[5], ys[5]);

        static void Predict(NeuralNetwork network, List<float> arguments, float expected)
        { 
            Console.WriteLine("PREDICTION:");
            Console.WriteLine("---");

            var inputs     = arguments.Select(x => new Node(x)).ToList();
            var prediction = network.F(inputs);

            Console.WriteLine($"Inputs: ({inputs[0]}, {inputs[1]}, {inputs[2]}, {inputs[3]})");
            Console.WriteLine($"Output (from neural net): {prediction.Value}");
            Console.WriteLine($"Expected (from training data): {expected}");
            Console.WriteLine($"Accuracy: {(1.0f - MathF.Abs(prediction.Value - expected)) * 100}%");
            Console.WriteLine();
        }

        Console.ReadKey();
    }

    public static Node MeanErrorSquared(Node prediction, Node expected) {
        return (prediction - expected) * (prediction - expected);
    }
}
