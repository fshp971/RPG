import pickle
import logging
import os
import sys
import torch
import numpy as np

import utils


def evaluate(model, criterion, loader, attacker=None, cpu=False):
    acc = utils.AverageMeter()
    loss = utils.AverageMeter()

    for x, y in loader:
        if not cpu:
            x, y = x.cuda(), y.cuda()

        if attacker is not None:
            adv_x = attacker.perturb(model, criterion, x, y)
        else:
            adv_x = x

        with torch.no_grad():
            model.eval()
            _y = model(adv_x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)
            lo = criterion(_y, y)

        acc.update(ac, len(y))
        loss.update(lo.item(), len(y))

    return acc.average(), loss.average()


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
        random_start = False,
        norm_type = args.pgd_norm_type,
    )

    train_loader = utils.get_loader(
        args.dataset, args.batch_size, train=True, training=False)
    test_loader = utils.get_loader(
        args.dataset, args.batch_size, train=False, training=False)

    log = dict()

    atk_acc = utils.membership_inference_attack(model, train_loader, test_loader, args.cpu)
    log['atk_acc'] = atk_acc

    nat_train_acc, nat_train_loss = evaluate(
        model, criterion, train_loader, cpu=args.cpu)
    log['nat_train_acc'] = nat_train_acc
    log['nat_train_loss'] = nat_train_loss

    adv_train_acc, adv_train_loss = evaluate(
        model, criterion, train_loader, attacker=attacker, cpu=args.cpu)
    log['adv_train_acc'] = adv_train_acc
    log['adv_train_loss'] = adv_train_loss

    nat_test_acc, nat_test_loss = evaluate(
        model, criterion, test_loader, cpu=args.cpu)
    log['nat_test_acc'] = nat_test_acc
    log['nat_test_loss'] = nat_test_loss

    adv_test_acc, adv_test_loss = evaluate(
        model, criterion, test_loader, attacker=attacker, cpu=args.cpu)
    log['adv_test_acc'] = adv_test_acc
    log['adv_test_loss'] = adv_test_loss

    with open('{}/{}.pkl'.format(args.save_dir, args.eval_save_name), 'wb') as f:
        pickle.dump(log, f)


if __name__ == '__main__':
    args = utils.get_args()

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    main(args)
