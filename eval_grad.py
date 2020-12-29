import pickle
import logging
import os
import sys
import torch
import numpy as np

import utils


def get_grad_mu(model, criterion, loader, attacker, cpu=False):
    model.train()
    grad_mu = None
    cnt = 0

    for x, y in loader:
    # for idx, (x, y) in enumerate(loader):
        # print(idx)
        cnt += len(y)

        if not args.cpu:
            x, y = x.cuda(), y.cuda()
        adv_x = attacker.perturb(model, criterion, x, y)

        adv_loss = criterion(model(adv_x), y) * len(y)
        model.zero_grad()
        adv_loss.backward()

        if grad_mu is None:
            grad_mu = []
            for pp in model.parameters():
                grad_mu.append( np.array( pp.grad.data.cpu() ) )
        else:
            for g_mu, pp in zip(grad_mu, model.parameters()):
                g_mu += np.array( pp.grad.data.cpu() )

    # print(cnt)
    for g_mu in grad_mu:
        g_mu /= cnt

    return grad_mu


def get_grad_noise(model, criterion, loader, attacker, grad_mu, rep_T, samp_T, cpu=False):
    model.train()
    grad_noise = []

    for i in range(rep_T):
        x, y = next(loader)

        if not args.cpu:
            x, y = x.cuda(), y.cuda()
        adv_x = attacker.perturb(model, criterion, x, y)

        model.zero_grad()
        adv_loss = criterion(model(adv_x), y)
        adv_loss.backward()

        grad_list = []
        for g_mu, pp in zip(grad_mu, model.parameters()):
            grad_list.append(
                (np.array( pp.grad.data.cpu() ) - g_mu).reshape(-1) )

        grad_list = np.concatenate(grad_list)

        rand_idx = np.random.permutation( len(grad_list) )
        grad_noise.append( grad_list[rand_idx[:samp_T]] )

    grad_noise = np.concatenate(grad_noise)
    # print(len(grad_noise))
    # print(grad_noise.mean() / grad_noise.std())
    return grad_noise


def main(args):
    model = utils.get_arch(arch=args.arch, dataset=args.dataset)
    criterion = torch.nn.CrossEntropyLoss()

    state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
    model.load_state_dict( state_dict['model_state_dict'] )
    del state_dict

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    attacker = utils.PGDAttacker(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        random_start = True,
        norm_type = args.pgd_norm_type,
    )

    if args.grad_opt == 'get-mu':
        ''' calculate grad_mu '''
        train_loader = utils.get_loader(
            args.dataset, args.batch_size, train=True, training=False)

        grad_mu = get_grad_mu(
            model, criterion, train_loader, attacker, args.cpu)

        with open('{}/{}-mu.pkl'.format(args.save_dir, args.eval_save_name), 'wb') as f:
            pickle.dump(grad_mu, f)

    elif args.grad_opt == 'get-noise':
        ''' calculate grad noise '''
        with open(args.resume_grad_mu_path, 'rb') as f:
            grad_mu = pickle.load(f)
        train_loader = utils.get_loader(
            args.dataset, args.batch_size, train=True, training=True)

        grad_noise = get_grad_noise(
            model, criterion, train_loader, attacker, grad_mu,
            args.grad_rep_T, args.grad_samp_T, args.cpu)

        with open('{}/{}-noise.pkl'.format(args.save_dir, args.eval_save_name), 'wb') as f:
            pickle.dump(grad_noise, f)


if __name__ == '__main__':
    args = utils.get_args()

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    main(args)
