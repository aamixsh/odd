import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def np_to_device(data, device):
    return torch.from_numpy(data).to(device)


def create_dataloader(feats, logits, labels, odd_weights=None, shuffle=True, batch_size=256):
    if odd_weights is not None:
        return DataLoader(TensorDataset(feats, logits, labels, odd_weights), shuffle=shuffle, batch_size=batch_size)
    else:
        return DataLoader(TensorDataset(feats, logits, labels), shuffle=shuffle, batch_size=batch_size)


def calibration_temp(logits, labels):
    log_temp = torch.nn.Parameter(torch.tensor([0.], device=logits.device))
    temp_opt = torch.optim.LBFGS([log_temp], lr=.1, max_iter=50, tolerance_change=5e-5)

    def closure_fn():
        loss = F.cross_entropy(logits * torch.exp(log_temp), labels)
        temp_opt.zero_grad()
        loss.backward()
        return loss

    temp_opt.step(closure_fn)
    return torch.exp(log_temp).item()


def negentropy(logits):
    return (torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1)).sum(1)


def load_shift_data(feats_dir, dataset, shift, method, device, args):
    detector_iters = args.detector_iters
    detector_layers = args.detector_layers
    detector_bal = args.detector_bal
    detector_width_factor = args.detector_width_factor
    source_clamp = args.source_clamp
    target_clamp = args.target_clamp

    shift_dir = os.path.join(feats_dir, dataset, f'{method}_{shift}_100.0')
    feats = np.load(os.path.join(shift_dir, 'model_feats.npy'), allow_pickle=True).item()
    labels = np.load(os.path.join(shift_dir, 'ytrue.npy'), allow_pickle=True).item()

    source_feats, source_labels = np_to_device(feats['source'], device), np_to_device(labels['source'], device)
    target_feats, target_labels = np_to_device(feats['target'], device), np_to_device(labels['target'], device)

    # train detector on source and target features.
    detector = Detector(n_features=source_feats.shape[1], n_classes=2, n_layers=detector_layers, width_factor=detector_width_factor).to(device)
    optimizer_detector = torch.optim.Adam(detector.parameters(), lr=1e-4)

    if detector_bal:
        # subsample majority class to balance dataset.
        min_n = min(len(source_feats), len(target_feats))
        if min_n == len(source_feats):
            source_data = source_feats
            target_data = target_feats[np.random.choice(len(target_feats), min_n, replace=False)]
        else:
            source_data = source_feats[np.random.choice(len(source_feats), min_n, replace=False)]
            target_data = target_feats

        dX = torch.cat((torch.tensor(source_data, dtype=torch.float).to(device), torch.tensor(target_data, dtype=torch.float).to(device)), dim=0)
        dy = torch.cat((torch.zeros(len(source_data), dtype=torch.int64).to(device), torch.ones(len(target_data), dtype=torch.int64).to(device)), dim=0)

    else:
        dX = torch.cat((source_feats, target_feats), dim=0)
        dy = torch.cat((torch.zeros(len(source_feats), dtype=torch.int64).to(device), torch.ones(len(target_feats), dtype=torch.int64).to(device)), dim=0)

    # train detector till convergence.
    previous_loss = float('inf')
    convergence_threshold = 1e-4
    patience = 10
    convergence_counter = 0

    pbar = tqdm(range(detector_iters))

    for iter in pbar:
        detector.train()
        logits = detector(dX)
        loss = torch.nn.functional.cross_entropy(logits, dy)
        optimizer_detector.zero_grad()
        loss.backward()
        optimizer_detector.step()

        detector.eval()
        logits = detector(dX)
        acc = mean_accuracy(logits, dy, reduction='mean')

        pbar.set_description_str(desc='Detector train Accuracy: '+str(acc.item())+', loss: '+str(loss.item()))

        loss_diff = abs(loss.item() - previous_loss)
        if loss_diff < convergence_threshold:
            convergence_counter += 1
        else:
            convergence_counter = 0

        if convergence_counter >= patience:
            print("Converged!")
            break
        previous_loss = loss.item()

    if args.weighting == 'prob':
        target_odd_weights = detector(target_feats).softmax(dim=1)[:, 1].detach()
        source_odd_weights = detector(source_feats).softmax(dim=1)[:, 1].detach()
    elif args.weighting == 'loss':
        target_odd_weights = torch.nn.functional.cross_entropy(detector(target_feats), torch.ones(len(target_feats), dtype=torch.int64).to(device), reduction='none').detach()
        source_odd_weights = torch.nn.functional.cross_entropy(detector(source_feats), torch.zeros(len(source_feats), dtype=torch.int64).to(device), reduction='none').detach()
        # Clamp the target loss for appropriate weighting.
        if target_clamp != '-1':
            target_odd_weights = torch.clamp(target_odd_weights, max=float(target_clamp))
            if args.target_loss_threshold != '-1':
                target_odd_weights[target_odd_weights > float(args.target_loss_threshold)] = float(target_clamp)
        target_odd_weights = 1 - target_odd_weights
    
        # Clamp the source loss for appropriate weighting.
        if source_clamp != '-1':
            source_odd_weights = torch.clamp(source_odd_weights, max=float(source_clamp))
            if args.source_loss_threshold != '-1':
                source_odd_weights[source_odd_weights > float(args.source_loss_threshold)] = float(source_clamp)

    lin = torch.nn.Linear(source_feats.shape[1], source_labels.max() + 1, device=device)
    lin.load_state_dict(torch.load(os.path.join(shift_dir, 'linear.pth'), map_location=device))
    h1_source_out = (source_feats @ lin.weight.T + lin.bias).detach()
    h1_target_out = (target_feats @ lin.weight.T + lin.bias).detach()

    return (source_feats, source_labels), (target_feats, target_labels), (h1_source_out, h1_target_out), (source_odd_weights, target_odd_weights)


class Detector(torch.nn.Module):
    def __init__(self, n_features=512, n_classes=2, n_layers=2, width_factor=1):
        super(Detector, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_features, n_features // width_factor)])
        for _ in range(n_layers - 2):
            self.layers.append(torch.nn.Linear(n_features // width_factor, n_features // width_factor))
        self.layers.append(torch.nn.Linear(n_features // width_factor, n_classes))
        self.relu = torch.nn.ReLU(True)
        for lin in self.layers:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
    def forward(self, x):
        out = x
        for lin in self.layers[:-1]:
            out = self.relu(lin(out))
        out = self.layers[-1](out)

        return out

def mean_accuracy(logits, y, reduction='mean'):
    preds = torch.argmax(logits, dim=1)
    if reduction == 'mean':
      return torch.count_nonzero(preds == y) / len(preds)
    else:
      return torch.count_nonzero(preds == y)
