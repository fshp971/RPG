import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--arch', type=str, default='wrn')

    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd','adam'])
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-steps', type=int, default=80001)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    parser.add_argument('--lr-decay-freq', type=int, default=30000)

    ''' for adversarial training '''
    parser.add_argument('--pgd-radius', type=float, default=0)
    parser.add_argument('--pgd-steps', type=int, default=0)
    parser.add_argument('--pgd-step-size', type=float, default=0)
    parser.add_argument('--pgd-random-start', action='store_true')
    parser.add_argument('--pgd-norm-type', type=str, default='l-infty',
                        choices=['l-infty', 'l2', 'l1'])

    ''' random seed '''
    parser.add_argument('--fix-random-seed', action='store_true')
    parser.add_argument('--np-seed', type=int, default=19260817)
    parser.add_argument('--torch-seed', type=int, default=998244353)

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calc-mg-freq', type=int, default=100)
    parser.add_argument('--report-freq', type=int, default=100)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--save-dir', type=str, default='./temp/')

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-train-step', type=int, default=None)

    parser.add_argument('--eval-save-name', type=str, default=None)

    parser.add_argument('--grad-opt', type=str, default=None,
                        choices=['get-mu', 'get-noise'])
    parser.add_argument('--resume-grad-mu-path', type=str, default=None)
    parser.add_argument('--grad-rep-T', type=int, default=0)
    parser.add_argument('--grad-samp-T', type=int, default=0)

    return parser.parse_args()
