import sys

sys.path.append('../examples/vision/plnn')

from beta_CROWN_solver import LiRPAConvNet
from batch_branch_and_bound import relu_bab_parallel

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def bab(model_ori, data, target, norm, eps, data_max=None, data_min=None, branching='fsb'):
    if norm == np.inf:
        if data_max is None:
            # data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            # data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = data + eps  # torch.min(data + eps, data_max)  # eps is already normalized
            data_lb = data - eps  # torch.max(data - eps, data_min)
        else:
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data

    pred = torch.argmax(model_ori(data), dim=1)
    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, pred, target, solve_slope=True, device='cpu', in_size=(1, 3, 32, 32))

    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    args = lambda : None
    args.batch_size = 200
    args.decision_thresh = 100000
    args.no_beta = True
    args.max_subproblems_list = 100000
    args.iteration = 20
    args.timeout = 300
    args.lr_alpha = 0.1
    args.lr_beta = 0.1
    args.new_branching = False
    args.lr_init_alpha = 0.5
    args.branching_method = 'fsb'
    args.beta_warmup = True
    args.opt_coeffs = False
    args.opt_bias = False
    args.lp_test = False
    args.opt_intermediate_beta = False
    args.intermediate_refinement_layers = []
    args.share_slopes = False
    args.branching_candidates = 2
    args.branching_reduceop = 'min'
    args.optimizer = 'adam'
    min_lb, min_ub, glb_record, nb_states = relu_bab_parallel(model, domain, x, no_LP=True, args=args)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()

    print(min_lb)
    assert 0.585 < min_lb < 0.62972  # PGD bound = 0.62972, best certified bound so far = 0.60 (using GPU)


def test_fsb():
    model_ori = cifar_model_wide()
    data = torch.load('data/beta_crown_test_data')
    model_ori.load_state_dict(data['state_dict'])
    x = data['x']
    pidx = data['pidx']
    eps_temp = data['eps_temp']
    data_max = data['data_max']
    data_min = data['data_min']

    bab(model_ori, x, pidx, float('inf'), eps_temp, data_max=data_max, data_min=data_min, branching='fsb')


if __name__ == "__main__":
    # test_sb('sb-min')
    test_fsb()

