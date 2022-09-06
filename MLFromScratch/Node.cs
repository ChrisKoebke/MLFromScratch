namespace MLFromScratch;

public class Node {
    public Node(float value, List<Node> previous = null, string op = "const") {
        Value        = value;
        Gradient     = 0;
        BackwardFunc = () => { };
        Previous     = previous ?? new();
        Op           = op;
    }

    public float      Value;
    public float      Gradient;
    public Action     BackwardFunc;
    public List<Node> Previous;
    public string     Op; // For debugging purposes & visualization

    public List<Node> Parameters() {
        var result = new List<Node>();
        foreach (var previous in Previous) {
            result.AddRange(previous.Parameters());
        }
        result.Add(this);
        return result;
    }

    public void Backward() {
        Gradient = 1;

        var nodes = BuildTopologicalOrder(new(), new());
        nodes.Reverse();

        foreach (var node in nodes) {
            node.BackwardFunc();
        }
    }

    public List<Node> BuildTopologicalOrder(List<Node> nodes, HashSet<Node> visited) {
        if (visited.Contains(this)) {
            return nodes; 
        } else {
            visited.Add(this);
        }

        foreach (var node in Previous) {
            node.BuildTopologicalOrder(nodes, visited);
        }
        nodes.Add(this);

        return nodes;
    }

    public Node Tanh() {
        var t = (MathF.Exp(2 * Value) - 1) / (MathF.Exp(2 * Value) + 1);
        
        var result = new Node(t, new() { this }, "tanh");
        result.BackwardFunc = () => {
            Gradient += (1 - t*t) * result.Gradient;
        };

        return result;
    }

    public static Node operator +(Node left, Node right) {
        var result = new Node(left.Value + right.Value, new() { left, right }, "+");
        result.BackwardFunc = () => {
            left .Gradient += result.Gradient;
            right.Gradient += result.Gradient;
        };
        return result;
    }

    public static Node operator -(Node left, Node right) {
        var result = new Node(left.Value - right.Value, new() { left, right }, "-");
        result.BackwardFunc = () => {
            left .Gradient += result.Gradient;
            right.Gradient += result.Gradient;
        };
        return result;
    }

    public static Node operator *(Node left, Node right) {
        var result = new Node(left.Value * right.Value, new() { left, right }, "*");
        result.BackwardFunc = () => {
            left .Gradient += result.Gradient * right.Value;
            right.Gradient += result.Gradient * left .Value;
        };
        return result;
    }

    public override string ToString() {
        return Value.ToString();
    }
}