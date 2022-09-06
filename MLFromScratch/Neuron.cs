namespace MLFromScratch;

public class Neuron {
    public Neuron(int weights) {
        Weights = new();
        for (var i = 0; i < weights; i++) {
            var randomValue = RandomRange(-1.0f, 1.0f);
            var randomNode  = new Node(randomValue, op: "weight");
            Weights.Add(randomNode);
        }

        Bias = new(0, op: "bias");
    }

    public List<Node> Weights;
    public Node       Bias;

    public static Random Random = new();
    public static float RandomRange(float min, float max) {
        return min + (float)Random.NextDouble() * (max - min);
    }

    public List<Node> Parameters() {
        var result = new List<Node>();
        result.AddRange(Weights);
        result.Add(Bias);
        return result;
    }

    public Node F(List<Node> x) {
        Node dot = new(0);
        for (var i = 0; i < Weights.Count; i++) {
            dot += Weights[i] * x[i];
        }
        dot += Bias;
        
        var activation = dot.Tanh();
        return activation;
    }
}