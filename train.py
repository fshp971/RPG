import pickle
import logging
import os
import sys
import torch
import numpy as np

import utils


def save_checkpoint(name, path, model, optim, log):
    if os.path.exists(path) == False:
        os.makedirs(path)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
    }, '{}/{}-model.pkl'.format(path, name))

    with open('{}/{}-log.pkl'.format(path, name), 'wb') as f:
        pickle.dump(log, f)


def get_grad_norm(model, criterion, x, y):
    model.eval()
    params = [pp for pp in model.parameters()]

    max_grad_norm = 0
    grad_norm_list = []
    for xx, yy in zip(x, y):
        xx = xx.reshape(1, *xx.shape)
        yy = yy.reshape(1, *yy.shape)
        _yy = model(xx)
        lo = criterion(_yy, yy)
        grad = torch.autograd.grad(lo, params)
        grad_norm = 0
        for gg in grad: grad_norm += (gg**2).sum().item()
        grad_norm = np.sqrt(grad_norm)
        max_grad_norm = max(max_grad_norm, grad_norm)
        grad_norm_list.append(grad_norm)
    
    return max_grad_norm, grad_norm_list


def main(args):
    model = utils.get_arch(arch=args.arch, dataset=args.dataset)

    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader = utils.get_loader(
        args.dataset, args.batch_size, train=True, training=True)

    attacker = utils.PGDAttacker(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        random_start = args.pgd_random_start,
        norm_type = args.pgd_norm_type,
    )

    log = dict()

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    if args.resume:
        # raise NotImplementedError
        state_dict = torch.load( '{}-model.pkl'.format(args.resume_path) )
        model.load_state_dict(state_dict['model_state_dict'])
        optim.load_state_dict(state_dict['optim_state_dict'])

        with open('{}-log.pkl'.format(args.resume_path), 'rb') as f:
            log = pickle.load(f)

    # x, y = next(train_loader)
    # if not args.cpu: x, y = x.cuda(), y.cuda()
    # adv_x = attacker.perturb(model, criterion, x, y)
    # max_grad_norm, grad_norm_list = get_grad_norm(model,criterion,adv_x,y)
    # utils.add_log(log, 'max_grad_norm', max_grad_norm)
    # utils.add_log(log, 'grad_norm_list', grad_norm_list)
    # logger.info('step [{}/{}]: max_grad_norm {:.3e}'
    #             .format(0, args.train_steps, max_grad_norm))
    # logger.info('')

    if not args.resume:
        ''' save the initial model parameter '''
        save_checkpoint('ckpt-{}'.format(0), '{}/ckpts/'.format(args.save_dir), model, optim, log)

    start_step = args.resume_train_step if args.resume else 0
    for step in range(start_step, args.train_steps, 1):
        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group['lr'] = lr

        x, y = next(train_loader)
        if not args.cpu:
            x, y = x.cuda(), y.cuda()
        adv_x = attacker.perturb(model, criterion, x, y)

        if (step+1) % args.calc_mg_freq == 0:
            max_grad_norm, grad_norm_list = get_grad_norm(model, criterion, adv_x, y)
            utils.add_log(log, 'max_grad_norm', max_grad_norm)
            utils.add_log(log, 'grad_norm_list', grad_norm_list)
            logger.info('step [{}/{}]: max_grad_norm {:.3e}'
                        .format(step+1, args.train_steps, max_grad_norm))
            logger.info('')

        with torch.no_grad():
            model.eval()
            _y = model(x)
            nat_loss = criterion(_y, y)
            nat_acc = (_y.argmax(dim=1) == y).sum().item() / len(y)
            utils.add_log(log, 'nat_loss', nat_loss.item())
            utils.add_log(log, 'nat_acc', nat_acc)

        # ''' ERM begin '''
        # model.train()
        # _y = model(x)
        # nat_loss = criterion(_y, y)
        # nat_acc = (_y.argmax(dim=1) == y).sum().item() / len(y)
        # utils.add_log(log, 'nat_loss', nat_loss.item())
        # utils.add_log(log, 'nat_acc', nat_acc)

        # model.zero_grad()
        # nat_loss.backward()

        # nat_grad_norm = 0
        # for pp in model.parameters():
        #     nat_grad_norm += (pp.grad.data**2).sum().item()
        # nat_grad_norm = np.sqrt(nat_grad_norm)
        # utils.add_log(log, 'nat_grad_norm', nat_grad_norm)
        # ''' ERM end '''

        ''' adv begin (includes gradient descent) '''
        model.train()
        _y = model(adv_x)
        adv_loss = criterion(_y, y)
        adv_acc = (_y.argmax(dim=1) == y).sum().item() / len(y)
        utils.add_log(log, 'adv_loss', adv_loss.item())
        utils.add_log(log, 'adv_acc', adv_acc)

        optim.zero_grad()
        adv_loss.backward()
        optim.step()

        adv_grad_norm = 0
        for pp in model.parameters():
            adv_grad_norm += (pp.grad.data**2).sum().item()
        adv_grad_norm = np.sqrt(adv_grad_norm)
        utils.add_log(log, 'adv_grad_norm', adv_grad_norm)
        ''' adv end '''

        # xjb_rate = batch_grad_norm / old_batch_grad_norm
        # logger.info('RI??? {:.3e}'.format(xjb_rate))

        if (step+1) % args.report_freq == 0:
            logger.info('step [{}/{}]:'.format(step+1, args.train_steps))
            logger.info('nat_acc {:.2%} \t nat_loss {:.3e}'
                        .format( nat_acc, nat_loss.item() ))
            logger.info('adv_acc {:.2%} \t adv_loss {:.3e}'
                        .format( adv_acc, adv_loss.item() ))
            # logger.info('nat_grad_norm {:.3e} \t adv_grad_norm {:.3e} \t rate {:.3e}'
            #             .format( nat_grad_norm, adv_grad_norm, adv_grad_norm/nat_grad_norm ))
            logger.info('')

        if (step+1) % args.save_freq == 0 \
            or (step+1) == args.train_steps:
            save_checkpoint('ckpt-{}'.format(step+1), '{}/ckpts/'.format(args.save_dir), model, optim, log)


if __name__ == '__main__':
    args = utils.get_args()

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/log.txt'.format(args.save_dir), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<18}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    ''' fix random seed '''
    if args.fix_random_seed:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.manual_seed( args.torch_seed )
        torch.cuda.manual_seed( args.torch_seed )
        np.random.seed( args.np_seed )

    try:
        main(args)
    except Exception as e:
        logger.exception("Unexpected exception! %s", e)
