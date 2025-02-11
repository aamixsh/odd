import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import multivariate_normal
import argparse
import pickle

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(2, 16)
        self.lin2 = nn.Linear(16, 16)
        self.lin3 = nn.Linear(16, 2)
        self.relu = nn.ReLU(True)
        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
    
    def forward(self, x):
        out = self.relu(self.lin1(x))
        out = self.relu(self.lin2(out))
        out = self.lin3(out)
        return out

def nll(logits, y, reduction='mean'):
    return nn.functional.cross_entropy(logits, y, reduction=reduction)

def mean_accuracy(logits, y, reduce='sum'):
    preds = torch.argmax(logits, dim=1)
    if reduce == 'mean':
        return torch.count_nonzero(preds == y) / len(preds)
    else:
        return torch.count_nonzero(preds == y)

def generate_data(overlap_factor, n_train_samples=1000, n_val_samples=1000):
    def sample_gaussian(mu, sigma_x, sigma_y, theta, num_samples):
        cov = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        rotated_cov = R @ cov @ R.T
        samples = np.random.multivariate_normal(mu, rotated_cov, num_samples)
        return samples

    # num_train_samples = n_train_samples  # Number of training samples per domain
    # num_val_samples = n_val_samples    # Number of validation samples per domain

    # # Create rotated Gaussian sampling masks
    # theta1 = -np.pi/4  # 30 degree tilt for training
    # theta2 = np.pi/5 # -45 degree tilt for testing

    # # Training mask parameters
    # mu1 = np.array([-1, -0.25])
    # sigma_x1 = 0.3  # Increased sigma_x for ellipse
    # sigma_y1 = 0.15 # Reduced sigma_y for ellipse


    # # Test mask parameters
    # mu2 = np.array([0, 2])
    # sigma_x2 = 0.35 # Reduced sigma_x for ellipse
    # sigma_y2 = 0.2  # Increased sigma_y for ellipse

    num_train_samples = np.random.randint(1500, 2500)  # Number of training samples per domain
    num_val_samples = np.random.randint(1000, 1500)

    # Create rotated Gaussian sampling masks
    div = np.random.randint(2, 6)
    theta1 = -np.pi/div  # 30 degree tilt for training
    div = np.random.randint(2, 6)
    theta2 = np.pi/div # -45 degree tilt for testing

    mu1 = np.array([np.random.uniform(-2, 0), np.random.uniform(-0.75, 0.5)])
    sigma_x1 = np.random.uniform(0.2, 0.4)  # Increased sigma_x for ellipse
    sigma_y1 = np.random.uniform(0.1, 0.2) # Reduced sigma_y for ellipse

    # Test mask parameters
    mu2 = np.array([np.random.uniform(-1, 1), np.random.uniform(1.75, 2.5)])
    sigma_x2 = np.random.uniform(0.2, 0.4) # Reduced sigma_x for ellipse
    sigma_y2 = np.random.uniform(0.1, 0.2)  # Increased sigma_y for ellipse

    # Source domain parameters (using mu1, sigma_x1, sigma_y1, theta1 from cell 4)
    source_mu = mu1
    source_sigma_x = sigma_x1
    source_sigma_y = sigma_y1
    source_theta = theta1
    source_cov = np.array([[source_sigma_x**2, 0], [0, source_sigma_y**2]])
    source_R = np.array([[np.cos(source_theta), -np.sin(source_theta)],
                    [np.sin(source_theta), np.cos(source_theta)]])
    source_rotated_cov = source_R @ source_cov @ source_R.T

    # Target domain parameters (using mu2, sigma_x2, sigma_y2, theta2 from cell 4, but adjusting mu2 based on overlap_factor)
    target_mu = mu2 # Initial target mean from cell 4
    target_mu = target_mu + (source_mu - target_mu) * overlap_factor # Translate target mean towards source_mu based on overlap_factor
    target_sigma_x = sigma_x2
    target_sigma_y = sigma_y2
    target_theta = theta2
    target_cov = np.array([[target_sigma_x**2, 0], [0, target_sigma_y**2]])
    target_R = np.array([[np.cos(target_theta), -np.sin(target_theta)],
                    [np.sin(target_theta), np.cos(target_theta)]])
    target_rotated_cov = target_R @ target_cov @ target_R.T

    # Generate samples for source and target domains (train and val sets)
    source_train_samples = sample_gaussian(source_mu, source_sigma_x, source_sigma_y, source_theta, num_train_samples)
    source_val_samples = sample_gaussian(source_mu, source_sigma_x, source_sigma_y, source_theta, num_val_samples)
    target_train_samples = sample_gaussian(target_mu, target_sigma_x, target_sigma_y, target_theta, num_train_samples)
    target_val_samples = sample_gaussian(target_mu, target_sigma_x, target_sigma_y, target_theta, num_val_samples)

    # Use the same curvy boundary from cell 4 to assign labels (for training sets)
    positions_source = source_train_samples.T
    positions_target = target_train_samples.T

    # a = -0.7
    # b = -10
    # c = -0.5
    # d = -0.6

    # a = -0
    # b = -2
    # c = -0.5
    # d = -0.6

    a = np.random.uniform(-2, 2)
    b = np.random.uniform(-5, 5)
    c = np.random.uniform(-2, 0)
    d = np.random.uniform(-2, 0)

    rand_source =  + np.random.normal(0, 0.2, positions_source[0].shape)
    rand_target =  + np.random.normal(0, 0.2, positions_target[0].shape)

    cols_source_train = (positions_source[0, :] > np.cos(a * np.sin(b * positions_source[1, :]) + c * np.exp(d * positions_source[1, :]) + (positions_source[1, :]**2 + 2 * positions_source[1, :] - 5) / 2) + rand_source).astype(np.int32)
    cols_target_train = (positions_target[0, :] > np.cos(a * np.sin(b * positions_target[1, :]) + c * np.exp(d * positions_target[1, :]) + (positions_target[1, :]**2 + 2 * positions_target[1, :] - 5) / 2) + rand_target).astype(np.int32)

    positions_source_val = source_val_samples.T 
    positions_target_val = target_val_samples.T 

    rand_source_val =  + np.random.normal(0, 0.2, positions_source_val[0].shape)
    rand_target_val =  + np.random.normal(0, 0.2, positions_target_val[0].shape)

    cols_source_val = (positions_source_val[0, :] > np.cos(a * np.sin(b * positions_source_val[1, :]) + c * np.exp(d * positions_source_val[1, :]) + (positions_source_val[1, :]**2 + 2 * positions_source_val[1, :] - 5) / 2) + rand_source_val).astype(np.int32)
    cols_target_val = (positions_target_val[0, :] > np.cos(a * np.sin(b * positions_target_val[1, :]) + c * np.exp(d * positions_target_val[1, :]) + (positions_target_val[1, :]**2 + 2 * positions_target_val[1, :] - 5) / 2) + rand_target_val).astype(np.int32)

    return (source_train_samples, source_val_samples, cols_source_train, cols_source_val,
            target_train_samples, target_val_samples, cols_target_train, cols_target_val)

def train_domain_detector(X, Xt, Xv, Xtv, device):
    detector = MLP().to(device)
    optimizer_detector = optim.Adam(detector.parameters(), lr=0.001)
    
    X_det = torch.cat([X, Xt], dim=0).to(device)
    y_det = torch.cat([torch.zeros(len(X), dtype=torch.int64), 
                      torch.ones(len(Xt), dtype=torch.int64)]).to(device)
    
    Xv_det = torch.cat([Xv, Xtv], dim=0).to(device)
    yv_det = torch.cat([torch.zeros(len(Xv), dtype=torch.int64), 
                       torch.ones(len(Xtv), dtype=torch.int64)]).to(device)
    
    epochs = 1000
    pbar = tqdm(range(epochs), desc="Training domain detector")
    
    for epoch in pbar:
        detector.train()
        logits = detector(X_det)
        loss = nll(logits, y_det)
        
        optimizer_detector.zero_grad()
        loss.backward()
        optimizer_detector.step()
        
        detector.eval()
        acc = mean_accuracy(detector(Xv_det), yv_det, reduce='mean')
        pbar.set_description(f'Train loss: {loss.item():.4f}, acc: {acc.item():.4f}')
    
    return detector

def train_classifier(X, y, Xv, yv, device):
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1000
    pbar = tqdm(range(epochs), desc="Training classifier")
    
    for epoch in pbar:
        model.train()
        logits = model(X)
        loss = nll(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        acc = mean_accuracy(model(Xv), yv, reduce='mean')
        pbar.set_description(f'Train loss: {loss.item():.4f}, acc: {acc.item():.4f}')
    
    return model

def compute_bound(model, detector, X, Xt, Xv, Xtv, y, yt, yv, ytv, with_det_trg_loss=False, device=None):
    model.eval()
    detector.eval()

    with torch.no_grad():
        pv, ptv = model(Xv).detach(), model(Xtv).detach()
        p, pt = model(X).detach(), model(Xt).detach()
        dp, dpt = detector(X).detach(), detector(Xt).detach()
        dpv, dptv = detector(Xv).detach(), detector(Xtv).detach()

    critic = MLP().to(device)
    critic.load_state_dict(model.state_dict())

    optimizer = optim.Adam(critic.parameters(), lr=0.001)
    
    epochs = 1000
    pbar = tqdm(range(epochs), desc="Training critic")

    for epoch in pbar:
        critic.train()

        source_feats, h_source_logits = X, p
        target_feats, h_target_logits = Xt, pt

        source_pred_labels = h_source_logits.argmax(1)
        target_pred_labels = h_target_logits.argmax(1)

        critic_source_logits = critic(source_feats)
        critic_target_logits = critic(target_feats)

        src_loss = torch.nn.functional.cross_entropy(critic_source_logits, source_pred_labels, reduction='none')

        agree_logits = critic_target_logits.gather(1, target_pred_labels.unsqueeze(1))[:, 0]
        logit_margin = (agree_logits.unsqueeze(1) - critic_target_logits).sum(1) / (critic_target_logits.shape[1] - 1)
        trg_loss = torch.log1p(torch.exp(logit_margin))

        if with_det_trg_loss:
            det_trg_loss = dpt.softmax(1)[:, 1]
            trg_loss = trg_loss * (1 - det_trg_loss)

        loss = (src_loss.mean() + trg_loss.mean()) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        critic.eval()
        acc = mean_accuracy(critic(X), y, reduce='mean')
        acc_h = mean_accuracy(critic(X), source_pred_labels, reduce='mean')
        pbar.set_description_str(desc='Source actual accuracy: '+str(acc.item())+', Source predicted accuracy: '+str(acc_h.item())+', Source loss: '+str(src_loss.mean().item())+', Target loss: '+str(trg_loss.mean().item()))

    
    with torch.no_grad():
        hhat_source_val_preds = pv.argmax(1)
        hhat_target_val_preds = ptv.argmax(1)

        critic_source_val_preds = critic(Xv).argmax(1)
        critic_target_val_preds = critic(Xtv).argmax(1)

        source_agree = (hhat_source_val_preds == critic_source_val_preds).float().mean().item()
        target_agree = (hhat_target_val_preds == critic_target_val_preds).float().mean().item()

        overlap_target_agree = ((hhat_target_val_preds == critic_target_val_preds).float() * dptv.softmax(1)[:, 0]).mean().item()
        non_overlap_target_agree = ((hhat_target_val_preds == critic_target_val_preds).float() * dptv.softmax(1)[:, 1]).mean().item()
        overlap_source_agree = ((hhat_source_val_preds == critic_source_val_preds).float() * dpv.softmax(1)[:, 1]).mean().item()
        non_overlap_source_agree = ((hhat_source_val_preds == critic_source_val_preds).float() * dpv.softmax(1)[:, 0]).mean().item()

        print ('Source agree (with critic): ', source_agree, 'Target agree (with critic): ', target_agree)
        print ('Overlap target agree (with critic) (OTAC): ', overlap_target_agree, 'Non-overlap target agree (with critic) (NOTAC): ', non_overlap_target_agree)
        print ('Overlap source agree (with critic) (OSAC): ', overlap_source_agree, 'Non-overlap source agree (with critic) (NOSAC): ', non_overlap_source_agree)
        
        diff = source_agree - target_agree
        print ('Discrepancy (D): ', diff)
        weighted_diff = non_overlap_source_agree - non_overlap_target_agree
        print ('Weighted discrepancy (WD): ', weighted_diff)

        source_val_acc = (hhat_source_val_preds == yv).float()
        overlap_source_val_acc = (source_val_acc * dpv.softmax(1)[:, 1]).mean().item()
        
        source_val_acc = source_val_acc.mean().item()

        print ('Source validation accuracy (SVA): ', source_val_acc)
        print ('Overlap source validation accuracy (OSVA): ', overlap_source_val_acc)

        epsilon = np.sqrt((len(yv) + 4 * len(ytv)) * np.log(1. / 0.01) / (2 * len(ytv) * len(yv)))

        print ('Epsilon (E): ', epsilon)

        bound = source_val_acc - diff - epsilon
        bound_weps = source_val_acc - diff
        weighted_bound = source_val_acc - weighted_diff - epsilon
        weighted_bound_weps = source_val_acc - weighted_diff
        new_bound = overlap_source_val_acc + non_overlap_target_agree - epsilon
        new_bound_weps = overlap_source_val_acc + non_overlap_target_agree

        print ('Bound (B): SVA - D - E: ', bound)
        print ('Bound (Bwe): SVA - D: ', bound_weps)
        print ('Weighted bound (WB): SVA - WD - E: ', weighted_bound)
        print ('Weighted bound (WBwe): SVA - WD: ', weighted_bound_weps)
        print ('New bound (NB): OSVA + NOTAC - E: ', new_bound)
        print ('New bound (NBwe): OSVA + NOTAC: ', new_bound_weps)
                
        target_val_acc = (hhat_target_val_preds == ytv).float().mean().item()

        print ('Target validation accuracy (TVA): ', target_val_acc)
            
    return np.array([source_agree, target_agree, \
                    overlap_source_agree, non_overlap_source_agree, overlap_target_agree, non_overlap_target_agree, \
                    diff, weighted_diff, source_val_acc, overlap_source_val_acc, target_val_acc, \
                    bound, bound_weps, weighted_bound, weighted_bound_weps, new_bound, new_bound_weps])

def run_experiment(overlap_rates):
    results = {
        'overlap_rates': overlap_rates,
        'standard_bounds': [],
        'weighted_bounds': [],
    }

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    for overlap in tqdm(overlap_rates, desc="Testing overlap rates"):
        # Generate data
        (source_train_samples, source_val_samples, 
         cols_source_train, cols_source_val,
         target_train_samples, target_val_samples, 
         cols_target_train, cols_target_val) = generate_data(overlap)
        
        # Convert to tensors
        X = torch.tensor(source_train_samples).float().to(device)
        Xv = torch.tensor(source_val_samples).float().to(device)
        y = torch.tensor(cols_source_train).long().to(device)
        yv = torch.tensor(cols_source_val).long().to(device)
        
        Xt = torch.tensor(target_train_samples).float().to(device)
        Xtv = torch.tensor(target_val_samples).float().to(device)
        yt = torch.tensor(cols_target_train).long().to(device)
        ytv = torch.tensor(cols_target_val).long().to(device)
        
        # Train domain detector
        detector = train_domain_detector(X, Xt, Xv, Xtv, device)
        
        # Train classifier
        model = train_classifier(X, y, Xv, yv, device)
        
        # Compute bounds
        standard_results = compute_bound(model, detector, X, Xt, Xv, Xtv, y, yt, yv, ytv, with_det_trg_loss=False, device=device)
        weighted_results = compute_bound(model, detector, X, Xt, Xv, Xtv, y, yt, yv, ytv, with_det_trg_loss=True, device=device)
        
        results['standard_bounds'].append(standard_results)
        results['weighted_bounds'].append(weighted_results)

    results['standard_bounds'] = np.array(results['standard_bounds'])
    results['weighted_bounds'] = np.array(results['weighted_bounds'])
    
    return results

def plot_results(results):
    plt.figure(figsize=(20, 20))
    # Change the font size of the plot
    plt.rcParams.update({'font.size': 22})
    plt.plot(results['overlap_rates'], results['standard_bounds'][:, 0], 'purple', label='Source Validation Accuracy')
    plt.plot(results['overlap_rates'], results['standard_bounds'][:, 1], 'r--', label='(DIS2) Discrepancy (Source - Target agreement with critic)')
    plt.plot(results['overlap_rates'], results['standard_bounds'][:, 2], 'g:', label='(DIS2) Epsilon (E)')
    plt.plot(results['overlap_rates'], results['standard_bounds'][:, 3], 'y-', label='(DIS2) Bound (B)')
    plt.plot(results['overlap_rates'], results['standard_bounds'][:, 4], 'c-', label='(DIS2) Bound (Bwe)')
    plt.plot(results['overlap_rates'], results['standard_bounds'][:, 5], 'k--', label='(DIS2) Target Validation Accuracy')
    
    plt.plot(results['overlap_rates'], results['weighted_bounds'][:, 1], 'b--', label='(ODD) Discrepancy (Source - Target agreement with critic)')
    plt.plot(results['overlap_rates'], results['weighted_bounds'][:, 3], 'magenta', linestyle=':', label='(ODD) Bound (B)')
    plt.plot(results['overlap_rates'], results['weighted_bounds'][:, 4], 'brown', label='(ODD) Bound (Bwe)')
    

    plt.xlabel('Overlap Factor')
    plt.title('Comparison of Bounds vs. Overlap')
    plt.legend()
    plt.grid(True)
    plt.savefig('bound_comparison.png')
    plt.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_train_samples', type=int, default=1000)
    # parser.add_argument('--n_val_samples', type=int, default=1000)
    parser.add_argument('--num_runs', type=int, default=20)
    args = parser.parse_args()

    res_list = []
    
    for i in range(args.num_runs):
        print ('Running experiment ', i)
        # overlap_rates = np.linspace(0.0, 1.0, 20)  # Test overlap factors from 0 to 4
        overlap_rates = np.random.uniform(0.0, 1.0, 100)
        results = run_experiment(overlap_rates)
        res_list.append(results)
    
    pickle.dump(res_list, open('results.pkl', 'wb'))
    # plot_results(res_list)