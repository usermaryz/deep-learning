class Module:
    # обнуление градиента
    def zero_grad(self):
        for p in self.parameters():
          p.grad = 0

    def parameters(self):
      #передаем параметры
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(0) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
      # применяем веса  к каждому из данных
        act = sum(self.w[i] * x[i] for i in range(len(self.w))) + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
      # выводим параметры
        return self.w  +[self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    #  нейроны, сколько их в слое
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
      # применение у каждого слоя параметров к данным
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
      # параметры
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    # создаем слойку, каждый слой рассматриеваем отдельно
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
      # вызываем слой, применяем к нему его параметпреы
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        repr = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [{repr}]"