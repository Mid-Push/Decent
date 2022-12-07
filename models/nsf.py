from nflows import transforms, distributions, flows
import torch
import torch.nn as nn

def create_linear_transform(linear_transform_type, features):
    if linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_base_transform(base_transform_type, features, hidden_features, num_transform_blocks, dropout_probability,
                          use_batch_norm, num_bins, tail_bound):
    if base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
    elif base_transform_type == 'quadratic-autoregressive':
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
    elif base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
    else:
        raise ValueError


def create_transform(features, num_flow_steps, linear_transform_type='lu', base_transform_type='rq-autoregressive', hidden_features=256,
                     num_transform_blocks=2, dropout_probability=0.25, use_batch_norm=0, num_bins=8, tail_bound=3):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            transforms.BatchNorm(features),
            create_linear_transform(linear_transform_type, features),
            create_base_transform(base_transform_type, features, hidden_features, num_transform_blocks,
                                  dropout_probability, use_batch_norm, num_bins, tail_bound)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform(linear_transform_type, features)
    ])
    return transform

class NSF(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(NSF, self).__init__()
        base_dist = distributions.StandardNormal(shape=[input_dim])
        transform = create_transform(input_dim, num_blocks, base_transform_type='rq-autoregressive')
        self.flow = flows.Flow(transform, base_dist)

    def log_probs(self, x):
        return self.flow.log_prob(x)



if __name__ == '__main__':
    import random
    import numpy as np
    import os
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    seed=123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    flow = NSF(10, 1)
    x = np.random.randn(10000, 10)
    x = torch.from_numpy(x.astype(np.float32))
    print(x[:10])
    weight = torch.from_numpy(np.random.randn(10, 10).astype(np.float32))
    x = torch.sigmoid(torch.matmul(x, weight)).detach()
    #+ torch.sigmoid(x).detach()
    #x = x*2.
    print(x.view(-1)[:10])
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3, amsgrad=True)
    train_data = x[:8000]
    test_data = x[8000:]
    for it in range(60000):
        ids = np.random.choice(8000, 256)
        loss = -flow.log_probs(train_data[ids]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.1)
        opt.step()
        if it % 100 == 0:
            test_log_prob = flow.log_probs(test_data).mean()
            print('[%d] log_prob: %.3f' % (it, test_log_prob.item()))


