import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, None, :] * x)


class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        hidden_dim (int): hidden dimension
    """

    def __init__(self, c_dim, in_dim, hidden_dim=None, out_dim=None, 
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        if out_dim is None:
            out_dim = in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.bn_0 = norm(in_dim)
        self.bn_1 = norm(hidden_dim)

        self.fc_0 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.fc_c = nn.Conv1d(c_dim, out_dim, 1)
        self.actvn = nn.ReLU()

        if in_dim == out_dim:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(in_dim, out_dim, 1, bias=False)

        # Initialization
        # nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)

        return out


class Criticnet(nn.Module):
    """ Decoder conditioned on sigma.

    Example configuration:
        hidden_size: 256
        n_blocks: 5
        in_dim: 3  equals to out_dim
    """
    def __init__(self, in_dim=3, hidden_dim=256, n_blocks=5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # Input = Conditional = dim (xyz) + 1 (sigma)
        c_dim = in_dim + 1
        self.conv_p = nn.Conv1d(c_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([ResnetBlockConv1d(c_dim, hidden_dim) for _ in range(n_blocks)])
        self.bn_out = nn.BatchNorm1d(hidden_dim)
        self.conv_out = nn.Conv1d(hidden_dim, self.out_dim, 1)
        self.actvn_out = nn.ReLU()

    # This should have the same signature as the sig condition one
    def forward(self, x, c):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, 1) sigma
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        p = x.transpose(1, 2)  # (bs, dim, n_points)
        batch_size, D, num_points = p.size()

        c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
        c_xyz = torch.cat([p, c_expand], dim=1)
        net = self.actvn_out(self.conv_p(c_xyz))
        for block in self.blocks:
            net = block(net, c_xyz)
        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
        return out


class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, layer=nn.Linear, dropout=False):
        super(SmallMLP, self).__init__()
        out_dim = in_dim
        c_dim = in_dim + 1
        if dropout:
            self.net = nn.Sequential(
                layer(c_dim, hidden_dim),
                Swish(hidden_dim),
                nn.Dropout(.5),
                layer(hidden_dim, hidden_dim),
                Swish(hidden_dim),
                nn.Dropout(.5),
                layer(hidden_dim, out_dim)
            )
        else:
            self.net = nn.Sequential(
                layer(c_dim, hidden_dim),
                Swish(hidden_dim),
                layer(hidden_dim, hidden_dim),
                Swish(hidden_dim),
                layer(hidden_dim, out_dim)
            )
        # self.normalized = False

    def forward(self, x, c):
        batch_size, num_points, D  = x.size()
        c_expand = c.unsqueeze(1).expand(-1, num_points, -1)
        c_xyz = torch.cat([x, c_expand], dim=2)
        out = self.net(c_xyz)
        return out
        # if self.normalized:
            # return out / (out.norm(dim=1, keepdim=True) + 1e-6)
        # else:
            # return out
