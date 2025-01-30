from dataset.function import TensorDict


class BaseActor:
    def __init__(self, net, objective):

        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        raise NotImplementedError

    def to(self, device):
        self.net.to(device)

    def train(self, mode=True):
        self.net.train(mode)

    def eval(self):
        self.train(False)