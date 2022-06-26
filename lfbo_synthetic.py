from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import botorch

from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from botorch.optim.optimize import optimize_acqf
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors import DeterministicPosterior


class Forrester(SyntheticTestFunction):
    dim = 1
    _bounds = [(0.0, 1.0)]
    _optimal_value: 0.0
    _optimizers = [(0.0,)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (6. * X - 2.) ** 2 * torch.sin(12 * X - 4.)


class Network(Model):
    def __init__(self, input_dim, output_dim, num_layers, num_units):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if not i:
                self.layers.append(nn.Linear(input_dim, num_units))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(num_units, num_units))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(num_units, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def posterior(self, X: Tensor, **kwargs: Any) -> DeterministicPosterior:
        y = self.forward(X).view(-1)
        return DeterministicPosterior(y)


class LFAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model) -> None:
        super().__init__(model)

    def forward(self, X):
        return self.model.posterior(X).mean


def train_model(model, X, Y, W):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    batch_size = 64

    dataset = TensorDataset(X, Y, W)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    for i in range(500):
        for x, y, w in loader:
            optimizer.zero_grad()
            y_ = model(x)
            loss = nn.BCEWithLogitsLoss(weight=w)(y_, y)
            loss.backward()
            optimizer.step()
        # with torch.no_grad():
        #     Y_ = model(X)
        #     loss = nn.BCEWithLogitsLoss(weight=W)(Y_, Y)

    return model


def prepare_data(X, eta=1.0):
    fx = f(X).view(-1)
    tau = torch.quantile(fx, 0.33)

    y = torch.less(fx, tau)
    x1, y1 = X[y], y[y]
    x0, y0 = X, torch.zeros_like(y)
    w1 = (tau - fx)[y]
    w1 = w1 ** eta / torch.mean(w1)
    w0 = 1 - y0.float()
    s1 = x1.size(0)
    s0 = x0.size(0)

    X = torch.cat([x1, x0], dim=0)
    Y = torch.cat([y1, y0], dim=0).float().view(-1, 1)
    W = torch.cat([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], dim=0).view(-1, 1)
    W = W / W.mean()
    return X, Y, W


f = Forrester()

soboleng = torch.quasirandom.SobolEngine(dimension=1)

X_obs = soboleng.draw(4)

for i in range(100):
    if i < 4:
        print(f'{X_obs[i, 0].item():.4f}, {f(X_obs[i, 0]).item():.4f}')
        continue

    model = Network(1, 1, 2, 32)

    X, Y, W = prepare_data(X_obs)
    model = train_model(model, X, Y, W)

    acqf = LFAcquisitionFunction(model)

    a = optimize_acqf(acqf, bounds=torch.tensor([[0.0], [1.0]]), q=1, num_restarts=5, raw_samples=100)[0]

    print(f'{a.item():.4f}, {f(a).item():.4f}, {model(a).item():.4f}, \
            {nn.BCEWithLogitsLoss(weight=W)(model(X), Y).item():.4f}')

    X_obs = torch.cat([X_obs, a], dim=0)
