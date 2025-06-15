import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

sns.set_style("whitegrid")
# plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here
plt.rcParams.update({'font.size': 18})

parser = argparse.ArgumentParser()
parser.add_argument("--dis2", type=str, required=True)
parser.add_argument("--odd", type=str, required=True)
parser.add_argument("--plot_dir", type=str, default="plots/")
args = parser.parse_args()

data = pd.read_pickle(args.dis2)
odd_data = pd.read_pickle(args.odd)
data = data[~data['train_method'].isin(['DANN', 'CDANN'])]
odd_data = odd_data[~odd_data['train_method'].isin(['DANN', 'CDANN'])]
ns, nt = data['n_val_source'].to_numpy(), data['n_val_target'].to_numpy()
eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
data['lower_bound'] = np.maximum(data['h_val_acc'] - data['max_ts_agree_diff'] - eps, 0)
odd_data['lower_bound'] = np.maximum(odd_data['h_val_acc'] - odd_data['max_ts_agree_diff'] - eps, 0)

orig_data = data[data['bound_strategy'] == 'PCA1']
odd_orig_data = odd_data[odd_data['bound_strategy'] == 'PCA1']

orig_indices = orig_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
odd_orig_indices = odd_orig_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()

orig_preds = orig_data.loc[orig_indices][['lower_bound', 'trg_accuracy']].to_numpy()
odd_orig_preds = odd_orig_data.loc[odd_orig_indices][['lower_bound', 'trg_accuracy']].to_numpy()

logit_data = data[data['bound_strategy'] == 'logits']
odd_logit_data = odd_data[odd_data['bound_strategy'] == 'logits']
logit_indices = logit_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
odd_logit_indices = odd_logit_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
logit_preds = logit_data.loc[logit_indices][['lower_bound', 'trg_accuracy']].to_numpy()
odd_logit_preds = odd_logit_data.loc[odd_logit_indices][['lower_bound', 'trg_accuracy']].to_numpy()

logit_data['lower_bound2'] = logit_data['h_val_acc'] - logit_data['max_ts_agree_diff']
odd_logit_data['lower_bound2'] = odd_logit_data['h_val_acc'] - odd_logit_data['max_ts_agree_diff']

improved_indices = logit_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound2'].idxmax()
odd_improved_indices = odd_logit_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound2'].idxmax()

improved_preds = logit_data.loc[improved_indices][['lower_bound2', 'trg_accuracy']].to_numpy()
odd_improved_preds = odd_logit_data.loc[odd_improved_indices][['lower_bound2', 'trg_accuracy']].to_numpy()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
all_preds = [orig_preds, odd_orig_preds, logit_preds, odd_logit_preds, improved_preds, odd_improved_preds]
names = ['Features', 'Logits', 'Logits w/o $\delta$']

for i, ax in enumerate(axs.flatten()):
    dis2_preds = all_preds[2*i]
    ax.scatter(dis2_preds[:, 1], dis2_preds[:, 0], s=16, marker='.', color='olive', label="DIS$^2$")
    odd_preds = all_preds[2*i + 1]
    ax.scatter(odd_preds[:, 1], odd_preds[:, 0], s=16, marker='*', color='blue', label="ODD")
    ax.set_title(names[i], fontsize=22)
    ax.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1.)
    ax.set_xlabel('Target Accuracy', fontsize=18)
    ax.set_ylabel('Prediction', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0, 1)
    if i == 0:
        ax.legend(fontsize=18)

    # Draw arrows between corresponding Dis^2 and ODD points
    for dis2_point, odd_point in zip(dis2_preds, odd_preds):
        # Get x,y coordinates for both points
        x1, y1 = dis2_point[1], dis2_point[0]  # Dis^2 point
        x2, y2 = odd_point[1], odd_point[0]    # ODD point  
        
        # Choose color based on whether ODD prediction is better
        arrow_color = 'blue' if y2 > y1 else 'red'

        # Draw arrow
        ax.arrow(x1, y1,               # Start point
              x2-x1, y2-y1,         # Delta x, delta y
              head_width=0.005,      # Arrow head width
              head_length=0.005,     # Arrow head length
              color=arrow_color,    # Color based on comparison
              alpha=0.2,            # Make slightly transparent
              length_includes_head=True)

plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'dis2_odd_variants')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')