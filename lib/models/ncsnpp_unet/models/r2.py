import torch
from torch import nn
from torch.nn import functional as F


class PosLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(PosLinear, self).__init__(*args, **kwargs)

        self.sp = nn.Softplus()

    def forward(self, x):
        weight = self.sp(self.weight)
        bias = self.bias
        return F.linear(x, weight, bias)


class MonotIncF(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=1024, x_0=0, x_1=1, y_0=-10, y_1=10):
        super(MonotIncF, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = PosLinear(in_dim, out_dim)
        self.fc2 = PosLinear(out_dim, h_dim)
        self.fc3 = PosLinear(h_dim, out_dim)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.x_0 = nn.Parameter(torch.ones(1, in_dim) * x_0, requires_grad=False)
        self.x_1 = nn.Parameter(torch.ones(1, in_dim) * x_1, requires_grad=False)
        self.y_0 = nn.Parameter(torch.ones(1, out_dim) * y_0, requires_grad=False)

        y_gap = torch.ones(1, out_dim) * (y_1 - y_0)
        un_y_gap = torch.log(torch.expm1(y_gap))
        self.un_y_gap = nn.Parameter(un_y_gap, requires_grad=False)

    def unnorm_func(self, x):
        x = self.fc1(x)

        y = self.fc2(x)
        y = self.sigmoid(y)
        y = self.fc3(y)

        return x + y

    def get_gap(self):
        return self.softplus(self.un_y_gap)

    def get_y_0(self):
        return self.y_0

    def get_y_1(self):
        return self.y_0 + self.get_gap()

    def forward(self, x):
        bs = x.shape[0]
        x_adj = torch.cat([self.x_0, self.x_1, x], dim=0)

        y_adj = self.unnorm_func(x_adj)

        yo_0, yo_1, yo = y_adj[:1], y_adj[1:2], y_adj[2:]

        y_0 = self.get_y_0()
        y_gap = self.get_gap()

        y = y_0 + y_gap * (yo - yo_0) / (yo_1 - yo_0)

        return y


class R2(nn.Module):
    def __init__(self, tau, h_dim=1024):
        super(R2, self).__init__()

        self.mif = MonotIncF(
            in_dim=1, out_dim=1, h_dim=h_dim, x_0=0, x_1=tau, y_0=0, y_1=20
        )

    def forward(self, x):
        x = x.clone()[:, None]
        create_graph = torch.is_grad_enabled()

        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True

            y = self.mif(x)

            (grad,) = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=create_graph
            )

        if not create_graph:
            y.requires_grad = False
            grad.requires_grad = False

        r_2 = -grad
        e_int_r_2 = torch.exp(-y)

        return {"r2": r_2[:, 0], "e_int_r_2": e_int_r_2[:, 0]}
