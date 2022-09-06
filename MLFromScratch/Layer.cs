namespace MLFromScratch;

public class Layer {
    public Layer(int inputs, int outputs) {
        Neurons = new();
        for (var i = 0; i < outputs; i++) {
            var neuron = new Neuron(inputs);
            Neurons.Add(neuron);
        }
    }

    public List<Neuron> Neurons;

    public List<Node> Parameters() {
        var result = new List<Node>();
        foreach (var neuron in Neurons) {
            result.AddRange(neuron.Parameters());
        }
        return result;
    }

    public List<Node> F(List<Node> x) {
        var result = new List<Node>();
        foreach (var neuron in Neurons) {
            var y = neuron.F(x);
            result.Add(y);
        }
        return result;
    }
}