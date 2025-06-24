import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn

def subnet_fc(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, c_out, kernel_size=3, padding=1)
    )

def build_inn(channels=1, height=256, width=256, num_blocks=6, clamp=1.5):
    nodes = [Ff.InputNode(channels, height, width, name='input')]

    for k in range(num_blocks):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GLOWCouplingBlock,
                {
                    'subnet_constructor': subnet_fc,
                    'clamp': clamp
                },
                name=f'coupling_{k}'
            )
        )
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.PermuteRandom,
                {'seed': k},
                name=f'permute_{k}'
            )
        )

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes, verbose=False)
