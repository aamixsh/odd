import torch
from torch import nn
import numpy as np

from .critic import get_critic_source_loss, get_critic_target_loss, train_multiple_critics, eval_multiple_critics
from .utils import create_dataloader
from torch.utils.data import DataLoader, TensorDataset


def validation_scores_and_bounds_from_loaders(
        source_loader, source_val_loader,
        target_loader, target_val_loader,
        args, num_repeats=30, num_epochs=100, critic=None, delta=0.01, loss_type='disagreement'
):

    feats, logits, _, _ = next(iter(source_loader))
    dim = feats.shape[1]
    num_classes = logits.shape[1]
    device = logits.device
    critic_stats = {}
    n_source, n_target = len(source_loader.dataset), len(target_loader.dataset)
    n_val_source, n_val_target = len(source_val_loader.dataset), len(target_val_loader.dataset)

    if critic is None:
        critic_weights, critic_biases, critic_stats = \
            train_multiple_critics(source_loader, target_loader, source_val_loader, target_val_loader, args,
                                   num_repeats=num_repeats, num_epochs=num_epochs, loss_type=loss_type)
        train_agree_diffs = critic_stats['tr_src_agree'] - critic_stats['tr_trg_agree']
        train_loss_diffs = critic_stats['tr_trg_loss'] - critic_stats['tr_src_loss']
        train_weighted_agree_diffs = critic_stats['tr_src_weighted_agree'] - critic_stats['tr_trg_weighted_agree']
        train_weighted_loss_diffs = critic_stats['tr_trg_weighted_loss'] - critic_stats['tr_src_weighted_loss']
        # train_weighted_agree_weighted_nums = critic_stats['tr_src_weighted_agree_weighted_nums'] - critic_stats['tr_trg_weighted_agree_weighted_nums']
        # train_weighted_loss_weighted_nums = critic_stats['tr_src_weighted_loss_weighted_nums'] - critic_stats['tr_trg_weighted_loss_weighted_nums']
        test_agree_diffs = critic_stats['ts_src_agree'] - critic_stats['ts_trg_agree']
        test_loss_diffs = critic_stats['ts_trg_loss'] - critic_stats['ts_src_loss']
        test_weighted_agree_diffs = critic_stats['ts_src_nonoverlap_weighted_agree'] - critic_stats['ts_trg_nonoverlap_weighted_agree']
        test_weighted_loss_diffs = critic_stats['ts_trg_weighted_loss'] - critic_stats['ts_src_weighted_loss']
        # test_weighted_agree_weighted_nums = critic_stats['ts_src_weighted_agree_weighted_nums'] - critic_stats['ts_trg_weighted_agree_weighted_nums']
        # test_weighted_loss_weighted_nums = critic_stats['ts_src_weighted_loss_weighted_nums'] - critic_stats['ts_trg_weighted_loss_weighted_nums']
        
        if args.best_critic_strategy == 'dis2':
            best_critic_ind = test_agree_diffs.argmax() // num_epochs
        elif args.best_critic_strategy == 'odd':
            best_critic_ind = test_weighted_agree_diffs.argmax() // num_epochs
        elif args.best_critic_strategy == 'new_bound':
            best_critic_ind = critic_stats['new_bound'].argmin() // num_epochs

        critic = nn.Linear(dim, num_classes, device=device)
        critic.weight.data = critic_weights[best_critic_ind].T
        critic.bias.data = critic_biases[best_critic_ind]

    else:
        source_agrees, target_agrees, source_overlap_weighted_agrees, target_overlap_weighted_agrees, \
        source_nonoverlap_weighted_agrees, target_nonoverlap_weighted_agrees, \
        source_losses, target_losses, source_weighted_losses, target_weighted_losses, \
        weighted_source_nums, weighted_target_nums, n_source, n_target = eval_multiple_critics(
            critic.weight.T.unsqueeze(0), critic.bias.unsqueeze(0),
            source_loader, target_loader, loss_type=loss_type)

        source_val_agrees, target_val_agrees, source_val_overlap_weighted_agrees, target_val_overlap_weighted_agrees, \
        source_val_nonoverlap_weighted_agrees, target_val_nonoverlap_weighted_agrees, \
        source_val_losses, target_val_losses, source_val_weighted_losses, target_val_weighted_losses, \
        weighted_source_val_nums, weighted_target_val_nums, n_val_source, n_val_target = eval_multiple_critics(
            critic.weight.T.unsqueeze(0), critic.bias.unsqueeze(0),
            source_val_loader, target_val_loader, loss_type=loss_type)
        
        critic_stats['tr_src_agree'], critic_stats['tr_trg_agree'] = source_agrees / n_source, target_agrees / n_target
        critic_stats['tr_src_weighted_agree'], critic_stats['tr_trg_weighted_agree'] = source_weighted_agrees / n_source, target_weighted_agrees / n_target
        # critic_stats['tr_src_weighted_agree_weighted_nums'], critic_stats['tr_trg_weighted_agree_weighted_nums'] = source_weighted_agree_weighted_nums / weighted_source_nums, target_weighted_agree_weighted_nums / weighted_target_nums
        # critic_stats['tr_src_weighted_loss_weighted_nums'], critic_stats['tr_trg_weighted_loss_weighted_nums'] = source_weighted_loss_weighted_nums / weighted_source_nums, target_weighted_loss_weighted_nums / weighted_target_nums

        critic_stats['ts_src_agree'], critic_stats['ts_trg_agree'] = source_val_agrees / n_val_source, target_val_agrees / n_val_target
        critic_stats['ts_src_overlap_weighted_agree'], critic_stats['ts_trg_overlap_weighted_agree'] = source_val_overlap_weighted_agrees / n_val_source, target_val_overlap_weighted_agrees / n_val_target 
        critic_stats['ts_src_nonoverlap_weighted_agree'], critic_stats['ts_trg_nonoverlap_weighted_agree'] = source_val_nonoverlap_weighted_agrees / n_val_source, target_val_nonoverlap_weighted_agrees / n_val_target
        # critic_stats['ts_src_weighted_agree_weighted_nums'], critic_stats['ts_trg_weighted_agree_weighted_nums'] = source_val_weighted_agree_weighted_nums / weighted_source_val_nums, target_val_weighted_agree_weighted_nums / weighted_target_val_nums   
        # critic_stats['ts_src_weighted_loss_weighted_nums'], critic_stats['ts_trg_weighted_loss_weighted_nums'] = source_val_weighted_loss_weighted_nums / weighted_source_val_nums, target_val_weighted_loss_weighted_nums / weighted_target_val_nums
        critic_stats['new_bound'] = source_val_overlap_weighted_agrees / n_val_source + target_val_nonoverlap_weighted_agrees / n_val_target

        critic_stats['weighted_source_val_nums'] = weighted_source_val_nums
        critic_stats['weighted_target_val_nums'] = weighted_target_val_nums
        critic_stats['n_val_source'] = n_val_source
        critic_stats['n_val_target'] = n_val_target

        train_agree_diffs = ((source_agrees - target_agrees) / n_source).unsqueeze(0)
        train_weighted_agree_diffs = ((source_weighted_agrees - target_weighted_agrees) / n_source).unsqueeze(0)
        train_weighted_loss_diffs = ((target_weighted_losses - source_weighted_losses) / n_source).unsqueeze(0)
        # train_weighted_agree_weighted_nums = ((source_weighted_agree_weighted_nums - target_weighted_agree_weighted_nums) / weighted_source_nums).unsqueeze(0)
        # train_weighted_loss_weighted_nums = ((target_weighted_loss_weighted_nums - source_weighted_loss_weighted_nums) / weighted_source_nums).unsqueeze(0)
        test_agree_diffs = ((source_val_agrees - target_val_agrees) / n_val_source).unsqueeze(0)
        test_weighted_agree_diffs = ((source_val_nonoverlap_weighted_agrees - target_val_nonoverlap_weighted_agrees) / n_val_source).unsqueeze(0)
        # test_weighted_agree_weighted_nums = ((source_val_weighted_agree_weighted_nums - target_val_weighted_agree_weighted_nums) / weighted_source_val_nums).unsqueeze(0)
        test_weighted_loss_diffs = ((target_val_weighted_losses - source_val_weighted_losses) / n_val_source).unsqueeze(0)
        # test_weighted_loss_weighted_nums = ((target_val_weighted_loss_weighted_nums - source_val_weighted_loss_weighted_nums) / weighted_source_val_nums).unsqueeze(0)

    with torch.no_grad():
        h_acc = h_val_acc = 0.
        weighted_h_acc = weighted_h_val_acc = 0.
        critic_source_val_loss = weighted_critic_source_val_loss = 0.
        for feats, h_logits, labels, source_odd_weights in source_val_loader:
            acc = (h_logits.argmax(1) == labels)
            h_val_acc += acc.sum().item()
            weighted_h_val_acc += (acc * source_odd_weights.clamp(max=1)).sum().item()

            if not critic_stats:
                critic_source_val_loss += get_critic_source_loss(critic(feats), h_logits.argmax(1)).sum().item()
                weighted_critic_source_val_loss += get_critic_source_loss(critic(feats), h_logits.argmax(1), source_odd_weights).sum().item()
       
        h_val_acc /= critic_stats['n_val_source']
        weighted_h_val_acc /= critic_stats['n_val_source']
        # weighted_h_val_acc_weighted_nums = weighted_h_val_acc / critic_stats['weighted_source_val_nums']
        critic_source_val_loss = critic_source_val_loss / critic_stats['n_val_source']
        weighted_critic_source_val_loss = weighted_critic_source_val_loss / critic_stats['n_val_source']
        # weighted_critic_source_val_loss_weighted_nums = weighted_critic_source_val_loss / critic_stats['weighted_source_val_nums']

        if 'ts_src_loss' in critic_stats:
            critic_source_val_loss = critic_stats['ts_src_loss'][best_critic_ind, -1].item()
            weighted_critic_source_val_loss = critic_stats['ts_src_weighted_loss'][best_critic_ind, -1].item()
            # weighted_critic_source_val_loss_weighted_nums = critic_stats['ts_src_weighted_loss_weighted_nums'][best_critic_ind, -1].item()
            critic_target_val_loss = critic_stats['ts_trg_loss'][best_critic_ind, -1].item()
            weighted_critic_target_val_loss = critic_stats['ts_trg_weighted_loss'][best_critic_ind, -1].item()
            # weighted_critic_target_val_loss_weighted_nums = critic_stats['ts_trg_weighted_loss_weighted_nums'][best_critic_ind, -1].item()
        else:
            critic_target_val_loss = 0.
            weighted_critic_target_val_loss = 0.
            for feats, h_logits, labels, target_odd_weights in target_val_loader:
                critic_logits = critic(feats)
                h_pred_labels = h_logits.argmax(1)
                critic_target_val_loss += get_critic_target_loss(
                    critic_logits, h_pred_labels, loss_type=loss_type).sum().item()
                weighted_critic_target_val_loss += get_critic_target_loss(
                    critic_logits, h_pred_labels, target_odd_weights, loss_type=loss_type).sum().item()
            
            critic_target_val_loss = critic_target_val_loss / critic_stats['n_val_target']
            weighted_critic_target_val_loss = weighted_critic_target_val_loss / critic_stats['n_val_target']
            # weighted_critic_target_val_loss_weighted_nums = weighted_critic_target_val_loss / critic_stats['weighted_target_val_nums']

        for feats, h_logits, labels, source_odd_weights in source_loader:
            acc = (h_logits.argmax(1) == labels)
            h_acc += acc.sum().item()
            weighted_h_acc += (acc * source_odd_weights.clamp(max=1)).sum().item()
        h_acc /= n_source
        weighted_h_acc /= n_source
        h_full_acc = (h_acc * n_source + h_val_acc * critic_stats['n_val_source']) / (n_source + critic_stats['n_val_source'])
        weighted_h_full_acc = (weighted_h_acc * n_source + weighted_h_val_acc * critic_stats['n_val_source']) / (n_source + critic_stats['n_val_source'])

        epsilon = np.sqrt((critic_stats['n_val_source'] + 4 * critic_stats['n_val_target']) * np.log(1. / delta) / (2 * critic_stats['n_val_target'] * critic_stats['n_val_source']))
        # epsilon_weighted_nums = np.sqrt((critic_stats['weighted_source_val_nums'] + 4 * critic_stats['weighted_target_val_nums']) * np.log(1. / delta) / (2 * critic_stats['weighted_target_val_nums'] * critic_stats['weighted_source_val_nums']))
        critic_stats['tr_agree_diffs'] = train_agree_diffs
        critic_stats['tr_weighted_agree_diffs'] = train_weighted_agree_diffs
        # critic_stats['tr_weighted_agree_weighted_nums'] = train_weighted_agree_weighted_nums
        # critic_stats['tr_weighted_loss_weighted_nums'] = train_weighted_loss_weighted_nums
        critic_stats['ts_agree_diffs'] = test_agree_diffs
        critic_stats['ts_weighted_agree_diffs'] = test_weighted_agree_diffs
        # critic_stats['ts_weighted_agree_weighted_nums'] = test_weighted_agree_weighted_nums
        # critic_stats['ts_weighted_loss_weighted_nums'] = test_weighted_loss_weighted_nums
        critic_stats['src_val_loss'] = critic_source_val_loss
        critic_stats['trg_val_loss'] = critic_target_val_loss
        critic_stats['src_val_weighted_loss'] = weighted_critic_source_val_loss
        critic_stats['trg_val_weighted_loss'] = weighted_critic_target_val_loss
        # critic_stats['src_val_weighted_loss_weighted_nums'] = weighted_critic_source_val_loss_weighted_nums
        # critic_stats['trg_val_weighted_loss_weighted_nums'] = weighted_critic_target_val_loss_weighted_nums

    # return critic_stats, (h_val_acc, weighted_h_val_acc, h_full_acc, weighted_h_full_acc), (epsilon, epsilon_weighted_nums), best_critic_ind
    return critic_stats, (h_val_acc, weighted_h_val_acc, h_full_acc, weighted_h_full_acc), (epsilon), best_critic_ind


def validation_scores_and_bounds_from_split_data(
        source_feats, source_val_feats, source_logits, source_val_logits, source_labels, source_val_labels,
        target_feats, target_val_feats, target_logits, target_val_logits, target_labels, target_val_labels,
        source_odd_weights, source_val_odd_weights, target_odd_weights, target_val_odd_weights, 
        args, num_repeats=30, num_epochs=100, critic=None, delta=0.01, loss_type='disagreement'
):

    source_loader = create_dataloader(source_feats, source_logits, source_labels, source_odd_weights)
    target_loader = create_dataloader(target_feats, target_logits, target_labels, target_odd_weights)
                    
    source_val_loader = create_dataloader(source_val_feats, source_val_logits, source_val_labels, source_val_odd_weights, shuffle=False)
    target_val_loader = create_dataloader(target_val_feats, target_val_logits, target_val_labels, target_val_odd_weights, shuffle=False)

    return validation_scores_and_bounds_from_loaders(
        source_loader, source_val_loader, target_loader, target_val_loader,
        args, num_repeats=num_repeats, num_epochs=num_epochs, critic=critic, delta=delta, loss_type=loss_type
    )