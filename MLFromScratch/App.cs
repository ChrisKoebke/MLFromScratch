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

        PredictAndPrint(network, xs[0], ys[0]);
        PredictAndPrint(network, xs[5], ys[5]);;

        static void PredictAndPrint(NeuralNetwork network, List<float> arguments, float expected)
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

        Console.WriteLine("COMPILED PREDICTION:");
        Console.WriteLine("---");

        var y = CompiledPredict(xs[0][0], xs[0][1], xs[0][2], xs[0][3]);
        Console.WriteLine($"Inputs: ({xs[0][0]}, {xs[0][1]}, {xs[0][2]}, {xs[0][3]})");
        Console.WriteLine($"Output (from neural net): {y}");
        Console.WriteLine($"Expected (from training data): {ys[0]}");
        Console.WriteLine($"Accuracy: {(1.0f - MathF.Abs(y - ys[0])) * 100}%");

        // var source = CodeGenerator.Compile(network);
        // Console.WriteLine();
        // Console.WriteLine(source);;

        Console.ReadKey();
    }

    public static float Tanh(float x) {
        return (MathF.Exp(2 * x) - 1) / (MathF.Exp(2 * x) + 1);
    }

    public static float CompiledPredict(float x0, float x1, float x2, float x3) {
        //
        // This is an example of code that was generated with CodeGenerator.Compile(network), to basically "bake in"
        // a network into native code so it can get executed faster and without allocating memory:
        //
        var x_0_0 = x0;
        var x_0_1 = x1;
        var x_0_2 = x2;
        var x_0_3 = x3;

        var x_1_0 = Tanh(0.17976332f * x_0_0 + -0.07247622f * x_0_1 + 0.25883532f * x_0_2 + -0.22645265f * x_0_3 + -0.35350746f);
        var x_1_1 = Tanh(0.9443617f * x_0_0 + 0.052588664f * x_0_1 + -0.4831369f * x_0_2 + -0.19101757f * x_0_3 + 0.098440625f);
        var x_1_2 = Tanh(0.82679355f * x_0_0 + 0.5911239f * x_0_1 + -0.639091f * x_0_2 + 0.11381389f * x_0_3 + 0.07297094f);
        var x_1_3 = Tanh(-1.0324512f * x_0_0 + 0.9858262f * x_0_1 + -0.31542856f * x_0_2 + -0.3934335f * x_0_3 + -0.045094162f);
        var x_1_4 = Tanh(0.43831238f * x_0_0 + -0.9546213f * x_0_1 + 0.7116353f * x_0_2 + -0.93591666f * x_0_3 + 0.018686743f);
        var x_1_5 = Tanh(-0.84479505f * x_0_0 + 0.17166275f * x_0_1 + 0.9817917f * x_0_2 + -0.37356707f * x_0_3 + 0.14791852f);
        var x_1_6 = Tanh(-0.7710782f * x_0_0 + -0.6037873f * x_0_1 + -0.22961557f * x_0_2 + 0.6951524f * x_0_3 + 0.22719917f);
        var x_1_7 = Tanh(-0.5624935f * x_0_0 + 0.105284505f * x_0_1 + -0.8487693f * x_0_2 + -0.12506394f * x_0_3 + 0.12266797f);
        var x_2_0 = Tanh(-0.56510156f * x_1_0 + 0.75620246f * x_1_1 + 0.11094339f * x_1_2 + -0.4622466f * x_1_3 + -0.5202279f * x_1_4 + -0.34843767f * x_1_5 + -0.041356146f * x_1_6 + -0.30611172f * x_1_7 + 0.07542046f);
        var x_2_1 = Tanh(0.0021360004f * x_1_0 + -0.14919424f * x_1_1 + -0.46087122f * x_1_2 + 0.43554133f * x_1_3 + -1.1045192f * x_1_4 + -0.69981194f * x_1_5 + -0.6310606f * x_1_6 + -0.5021968f * x_1_7 + -0.11064363f);
        var x_2_2 = Tanh(-0.59416693f * x_1_0 + 0.2208693f * x_1_1 + -0.47427106f * x_1_2 + 0.53049296f * x_1_3 + -0.040462524f * x_1_4 + 1.0405141f * x_1_5 + 0.55855244f * x_1_6 + 0.5756745f * x_1_7 + 0.099781826f);
        var x_2_3 = Tanh(-0.951668f * x_1_0 + -0.9311682f * x_1_1 + -0.30623075f * x_1_2 + -0.31770527f * x_1_3 + 0.11325169f * x_1_4 + 0.9438283f * x_1_5 + -0.54166377f * x_1_6 + -1.0405385f * x_1_7 + 0.12271716f);
        var x_2_4 = Tanh(-0.828794f * x_1_0 + -0.48712236f * x_1_1 + -0.09485937f * x_1_2 + -0.07117941f * x_1_3 + -0.019411897f * x_1_4 + 1.0137384f * x_1_5 + 0.73940283f * x_1_6 + 0.8019832f * x_1_7 + 0.09396536f);
        var x_2_5 = Tanh(0.25925305f * x_1_0 + -0.09705377f * x_1_1 + -0.14283955f * x_1_2 + -0.37317112f * x_1_3 + -0.7236235f * x_1_4 + -0.3086805f * x_1_5 + -0.38050216f * x_1_6 + 0.33082482f * x_1_7 + -0.021914743f);
        var x_2_6 = Tanh(-0.56004995f * x_1_0 + 0.76233476f * x_1_1 + 0.42753443f * x_1_2 + -0.81513333f * x_1_3 + -0.591621f * x_1_4 + 0.26527834f * x_1_5 + 0.83634496f * x_1_6 + -0.2023264f * x_1_7 + 0.15699628f);
        var x_2_7 = Tanh(0.27252933f * x_1_0 + 0.42326695f * x_1_1 + -0.9767607f * x_1_2 + -0.46765447f * x_1_3 + 0.54572946f * x_1_4 + -0.48381475f * x_1_5 + 0.8670065f * x_1_6 + -0.3801679f * x_1_7 + 0.016294638f);
        var x_3_0 = Tanh(0.31819886f * x_2_0 + -0.4290093f * x_2_1 + -0.7381542f * x_2_2 + 0.61780703f * x_2_3 + -1.1323014f * x_2_4 + -0.1742616f * x_2_5 + 0.3880024f * x_2_6 + -0.27467477f * x_2_7 + 0.3723702f);

        return x_3_0;
    }

    public static Node MeanErrorSquared(Node prediction, Node expected) {
        return (prediction - expected) * (prediction - expected);
    }
}
