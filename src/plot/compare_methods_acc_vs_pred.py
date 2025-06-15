import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import torch

sns.set_style("whitegrid")
# plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here

parser = argparse.ArgumentParser()
parser.add_argument("--dis2", type=str, required=True)
parser.add_argument("--odd", type=str, required=True)
parser.add_argument("--other", type=str, required=True)
parser.add_argument("--DA", action='store_true')
parser.add_argument("--strategy", type=str, default='logits')
parser.add_argument("--plot_dir", type=str, default="plots/")
parser.add_argument("--lb_type", type=str, default="lower_bound")
args = parser.parse_args()

dis2data = pd.read_pickle(args.dis2)
dis2data_det = pd.read_pickle(args.odd)

dis2data = dis2data[np.logical_xor(args.DA, ~dis2data['train_method'].isin(['DANN', 'CDANN']))]
dis2data_det = dis2data_det[np.logical_xor(args.DA, ~dis2data_det['train_method'].isin(['DANN', 'CDANN']))]

logit_dis2_data = dis2data[dis2data['bound_strategy'].isin([args.strategy])]
logit_dis2_det_data = dis2data_det[dis2data_det['bound_strategy'].isin([args.strategy])]

ns, nt = logit_dis2_data['n_val_source'].to_numpy(), logit_dis2_data['n_val_target'].to_numpy()
ns_det, nt_det = logit_dis2_det_data['n_val_source'].to_numpy(), logit_dis2_det_data['n_val_target'].to_numpy()

eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
eps_det = np.sqrt((4 * nt_det + ns_det) * np.log(100) / (2 * nt_det * ns_det))

# logit_dis2_data['lower_bound'] = logit_dis2_data['h_val_acc'] - logit_dis2_data['max_ts_agree_diff'] - eps
# logit_dis2_det_data['lower_bound'] = logit_dis2_det_data['h_val_acc'] - logit_dis2_det_data['max_ts_agree_diff'] - eps_det

logit_dis2_indices = logit_dis2_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
logit_dis2_det_indices = logit_dis2_det_data.groupby(['dataset', 'shift', 'train_method'])[args.lb_type].idxmax()


logit_dis2_preds = logit_dis2_data.loc[logit_dis2_indices][['lower_bound', 'trg_accuracy']].to_numpy()
logit_dis2_det_preds = logit_dis2_det_data.loc[logit_dis2_det_indices][[args.lb_type, 'trg_accuracy']].to_numpy()

# count= 0
# for i in range(len(logit_dis2_preds)):
#     if 1.1 * logit_dis2_preds[i, 0] < logit_dis2_det_preds[i, 0]:
#         count += 1
#         dis2_idx = logit_dis2_indices[i]
#         det_idx = logit_dis2_det_indices[i]
#         print("Dis2 Pred > Det Pred:")
#         print("Dis2 Data:")
#         for col in logit_dis2_data.columns:
#             print(f"{col}: {logit_dis2_data.loc[dis2_idx][col]}")
#         print ('-' * 50)
#         print("Det Data:")
#         for col in logit_dis2_det_data.columns:
#             print(f"{col}: {logit_dis2_det_data.loc[det_idx][col]}")
#         print("*" * 50)
 
# print (count)
# print (len(logit_dis2_preds))
# print (count / len(logit_dis2_preds))

for log in logit_dis2_det_preds:
    if type(log[0]) == torch.Tensor:
        log[0] = log[0].item()

other_data = pd.read_pickle(args.other)
other_data = other_data[np.logical_xor(args.DA, ~other_data['train_method'].isin(['DANN', 'CDANN']))]

ATC_preds = other_data[(other_data['prediction_method'] == 'ATC_NE') & (other_data['temperature'] == 'source')]
ATC_preds = ATC_preds[['lower_bound', 'trg_accuracy']].to_numpy()
COT_preds = other_data[(other_data['prediction_method'] == 'COT') & (other_data['temperature'] == 'source')]
COT_preds = COT_preds[['lower_bound', 'trg_accuracy']].to_numpy()
AC_preds = other_data[(other_data['prediction_method'] == 'AC') & (other_data['temperature'] == 'source')]
AC_preds = AC_preds[['lower_bound', 'trg_accuracy']].to_numpy()

plt.figure(figsize=(8, 4))
plt.scatter(logit_dis2_preds[:, 1], logit_dis2_preds[:, 0], s=25, label=r'Dis^2', zorder=1, marker='*', color='red', alpha=0.7)
plt.scatter(logit_dis2_det_preds[:, 1], logit_dis2_det_preds[:, 0], s=25, label=r'ODD', zorder=2, marker='^', color='blue', alpha=0.7)

# Draw arrows between corresponding Dis^2 and ODD points
for dis2_point, odd_point in zip(logit_dis2_preds, logit_dis2_det_preds[1:, :]):
    # Get x,y coordinates for both points
    x1, y1 = dis2_point[1], dis2_point[0]  # Dis^2 point
    x2, y2 = odd_point[1], odd_point[0]    # ODD point
    
    # Choose color based on whether ODD prediction is better
    arrow_color = 'blue' if y2 > y1 else 'red'
    
    # Draw arrow
    plt.arrow(x1, y1,               # Start point
              x2-x1, y2-y1,         # Delta x, delta y
              head_width=0.01,      # Arrow head width
              head_length=0.01,     # Arrow head length
              color=arrow_color,    # Color based on comparison
              alpha=0.5,            # Make slightly transparent
              length_includes_head=True)


plt.scatter(ATC_preds[:, 1], ATC_preds[:, 0], s=16, label='ATC', zorder=1, alpha=0.4)
plt.scatter(COT_preds[:, 1], COT_preds[:, 0], s=16, label='COT', zorder=1, alpha=0.4)
plt.scatter(AC_preds[:, 1], AC_preds[:, 0], s=16, label='AC', zorder=1, alpha=0.4)
plt.xlabel('Target Accuracy', fontsize=19)
plt.ylabel('Target Accuracy Prediction', fontsize=19)
plt.xlim(0.2, 1)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1., zorder=13)
plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(1, 1))
plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
# method_name = '_'.join(args.odd.split('/')[0].split('_')[1:])
method_name = args.odd.split('/')[0] + '_' + '_'.join(args.odd.split('/')[-1].split('.')[1].split('_')[1:])
name = os.path.join(args.plot_dir, f'{method_name}' + f'_{args.strategy}' + ('_DA' if args.DA else ''))

print (np.mean(logit_dis2_det_preds[:, 1] - logit_dis2_det_preds[:, 0]))
print (np.mean(logit_dis2_preds[:, 1] - logit_dis2_preds[:, 0]))
print (np.mean(ATC_preds[:, 1] - ATC_preds[:, 0]))
print (np.mean(COT_preds[:, 1] - COT_preds[:, 0]))
print (np.mean(AC_preds[:, 1] - AC_preds[:, 0]))

plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')
