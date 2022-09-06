using System.Linq;

namespace MLFromScratch;

public class NeuralNetwork {
    public NeuralNetwork(List<int> outputs) {
        Layers = new();
        for (var i = 0; i < outputs.Count-1; i++) {
            var layer = new Layer(outputs[i], outputs[i+1]);
            Layers.Add(layer);
        }
    }

    public List<Layer> Layers;

    public void ZeroGradients() {
        foreach (var node in Parameters()) {
            node.Gradient = 0;
        }
    }

    public List<Node> Parameters() {
        var result = new List<Node>();
        foreach (var layer in Layers) {
            result.AddRange(layer.Parameters());
        }
        return result;
    }

    public Node F(List<Node> x) {
        foreach (var layer in Layers) {
            x = layer.F(x);
        }

        // We are assuming the network always has a single value as output. If we would want
        // to support multiple values, we would need to return a List<Node> here...
        return x[0];
    }
}
