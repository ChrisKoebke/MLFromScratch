using System.Text;

namespace MLFromScratch;

public class CodeGenerator {
    public static string Compile(NeuralNetwork network) {
        var builder = new StringBuilder();

        for (var i = 0; i < network.Layers[0].InputCount; i++) {
            builder.AppendLine($"var x_0_{i} = x{i};");
        }

        builder.AppendLine();

        for (var layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++) { 
            var layer = network.Layers[layerIndex];
            for (var neuronIndex = 0; neuronIndex < layer.Neurons.Count; neuronIndex++) { 
                var neuron = layer.Neurons[neuronIndex];

                builder.Append($"var x_{layerIndex+1}_{neuronIndex} = Tanh(");

                for (var weightIndex = 0; weightIndex < neuron.Weights.Count; weightIndex++) { 
                    var weight = neuron.Weights[weightIndex];
                    if (weightIndex > 0) {
                        builder.Append(" + ");
                    }
                    builder.Append($"{weight.Value}f * x_{layerIndex}_{weightIndex}");
                }
                builder.AppendLine($" + {neuron.Bias}f);");
            }
        }

        builder.AppendLine();
        builder.AppendLine($"return x_{network.Layers.Count}_0;");

        return builder.ToString();
    }
}
