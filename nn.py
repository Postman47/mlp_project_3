import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
        dim_in=1, dim_out=1, dim_hidden=256, hidden_depth=2, batch_norm=False, last_act=None, act='relu'):
        super(MLP, self).__init__()
        self.encoding, self.network = None, None

        if not self.network:
            layers = []
            self.activation = act

            for i in range(hidden_depth):
                if i==0:
                    w_in, w_out = dim_in, dim_hidden
                    layers.append(nn.Flatten())
                else:
                    w_in, w_out = dim_hidden, dim_hidden

                match act:
                    # fill with additional cases if needed
                    case _:
                        layers.append(nn.Linear(in_features=w_in, out_features=w_out))
                        if batch_norm:
                            layers.append(nn.BatchNorm1d(w_out))
                        layers.append(nn.ReLU())

            # Add last layer with its activation
            layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_out))
            if last_act=='tanh':
                    layers.append(nn.Tanh())
            elif last_act=='sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Identity())

            self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        regression_output = output[:,0].unsqueeze(1)
        classification_output = output[:,1:]
        return regression_output, classification_output
