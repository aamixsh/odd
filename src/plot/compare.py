import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import argparse
import os
import torch
import pickle
from src.lib.consts import DATASET_SHIFTS, TRAIN_METHODS, BOUND_STRATEGIES
from src.lib.utils import load_shift_data, np_to_device

sns.set_style("whitegrid")
# plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here

parser = argparse.ArgumentParser()
parser.add_argument("--dis2", type=str, required=True, help="Path to DIS2 results")
parser.add_argument("--odd", type=str, required=True, help="Path to ODD results")
parser.add_argument("--other", type=str, required=True, help="Path to other methods results")
parser.add_argument("--DA", action='store_true', help="Include DA dataset")
parser.add_argument("--strategy", type=str, default='logits', help="Strategy to use for lower bound")
parser.add_argument("--plot_dir", type=str, default="plots/", help="Path to save plots")
parser.add_argument("--lb_type", type=str, default="lower_bound", help="Type of lower bound to use")
parser.add_argument("--include_cc", action='store_true', help="Include CC dataset")
args = parser.parse_args()

dis2data = pd.read_pickle(args.dis2)
odd_data = pd.read_pickle(args.odd)
other_data = pd.read_pickle(args.other)

process_order = pickle.load(open('results_both/process_order2.pkl', 'rb'))

inds = []

# process_order_data = []
for ind, dataset, shift, method, strategy in process_order:
    if dataset == 'cc' and not args.include_cc:
        continue
    inds.append(ind)
    # process_order_data.append({
    #     'process_order_index': ind
    # })

inds = np.array(inds)

if len(dis2data) == len(odd_data):
    dis2data = dis2data.iloc[inds].copy()
odd_data = odd_data.iloc[inds].copy()
    # for col, values in pd.DataFrame(process_order_data).items():
#     odd_data[f'process_{col}'] = values.values
# # other_data = other_data.iloc[inds].copy()

dis2data = dis2data[np.logical_xor(args.DA, ~dis2data['train_method'].isin(['DANN', 'CDANN']))]
odd_data = odd_data[np.logical_xor(args.DA, ~odd_data['train_method'].isin(['DANN', 'CDANN']))]
other_data = other_data[np.logical_xor(args.DA, ~other_data['train_method'].isin(['DANN', 'CDANN']))]

print (len(dis2data))
print (len(odd_data))
print (len(other_data))

print ("Unique bound strategies in dis2data:")

logit_dis2_data = dis2data[dis2data['bound_strategy'].isin([args.strategy])]
logit_odd_data = odd_data[odd_data['bound_strategy'].isin([args.strategy])]

ns, nt = logit_dis2_data['n_val_source'].to_numpy(), logit_dis2_data['n_val_target'].to_numpy()
ns_odd, nt_odd = logit_odd_data['n_val_source'].to_numpy(), logit_odd_data['n_val_target'].to_numpy()

eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))

# logit_dis2_data['lower_bound'] = logit_dis2_data['h_val_acc'] - logit_dis2_data['max_ts_agree_diff'] - eps
# logit_dis2_odd_data['lower_bound'] = logit_dis2_odd_data['h_val_acc'] - logit_dis2_odd_data['max_ts_agree_diff'] - eps_odd

logit_dis2_indices = logit_dis2_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
logit_odd_indices = logit_odd_data.groupby(['dataset', 'shift', 'train_method'])[args.lb_type].idxmax()


logit_dis2_preds = logit_dis2_data.loc[logit_dis2_indices][['lower_bound', 'trg_accuracy']].to_numpy()
logit_odd_preds = logit_odd_data.loc[logit_odd_indices][[args.lb_type, 'trg_accuracy']].to_numpy()
logit_odd_other_preds = logit_odd_data.loc[logit_odd_indices][['lower_bound_weighted', 'lower_bound_weighted_weps', 'lower_bound_new', 'lower_bound_new_weps', 'trg_accuracy']].to_numpy()

dis2_invalid_odd_invalid = {'dis2_gt_odd': [], 'dis2_lte_odd': []}
dis2_invalid_odd_valid = {'dis2_gt_odd': []}
dis2_valid_odd_invalid = {'dis2_lte_odd': []}
dis2_valid_odd_valid = {'dis2_gt_odd': [], 'dis2_lte_odd': []}

for i in range(len(logit_dis2_preds)):
    odd_idx = logit_odd_indices[i]
    if logit_dis2_preds[i, 0] > logit_dis2_preds[i, 1]:
        if logit_odd_preds[i, 0] > logit_odd_preds[i, 1]:
            if logit_dis2_preds[i, 0] > logit_odd_preds[i, 0]:
                dis2_invalid_odd_invalid['dis2_gt_odd'].append([logit_dis2_preds[i], logit_odd_preds[i, 0], process_order[odd_idx]])
            else:
                dis2_invalid_odd_invalid['dis2_lte_odd'].append([logit_dis2_preds[i], logit_odd_preds[i, 0], process_order[odd_idx]])
        else:
            dis2_invalid_odd_valid['dis2_gt_odd'].append([logit_dis2_preds[i], logit_odd_preds[i, 0], process_order[odd_idx]])
    else:
        if logit_odd_preds[i, 0] > logit_odd_preds[i, 1]:
            dis2_valid_odd_invalid['dis2_lte_odd'].append([logit_dis2_preds[i], logit_odd_preds[i, 0], process_order[odd_idx]])
        else:
            if logit_dis2_preds[i, 0] > logit_odd_preds[i, 0]:
                dis2_valid_odd_valid['dis2_gt_odd'].append([logit_dis2_preds[i], logit_odd_preds[i, 0], process_order[odd_idx]])
            else:
                dis2_valid_odd_valid['dis2_lte_odd'].append([logit_dis2_preds[i], logit_odd_preds[i, 0], process_order[odd_idx]])
        
#     odd_idx = logit_odd_indices[i]
#     print("Dis2 Pred > Odd Pred.")
#     print (process_order[odd_idx], i, count)
#     print("Dis2 Data:")
#     print (logit_dis2_data.loc[dis2_idx])
#     print ('-' * 50)
#     print("Odd Data:")
#     print (logit_odd_data.loc[odd_idx])
#     print("*" * 50)
 
# print (count)
# print (count / len(logit_dis2_preds))
print ('-' * 50)
print (dis2_invalid_odd_invalid, len(dis2_invalid_odd_invalid['dis2_gt_odd']), len(dis2_invalid_odd_invalid['dis2_lte_odd']))
print ('-' * 50)
print (dis2_invalid_odd_valid, len(dis2_invalid_odd_valid['dis2_gt_odd']))
print ('-' * 50)
print (dis2_valid_odd_invalid, len(dis2_valid_odd_invalid['dis2_lte_odd']))
print ('-' * 50)
print (dis2_valid_odd_valid, len(dis2_valid_odd_valid['dis2_gt_odd']), len(dis2_valid_odd_valid['dis2_lte_odd']))
print ('-' * 50)

accs = []
for data in dis2_invalid_odd_invalid['dis2_gt_odd']:
    odd_acc = data[0][1] - data[1]
    odd_acc_abs = np.abs(data[0][1] - data[1])
    dis2_acc = data[0][1] - data[0][0]
    dis2_acc_abs = np.abs(data[0][1] - data[0][0])
    accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

for data in dis2_invalid_odd_invalid['dis2_lte_odd']:
    odd_acc = data[0][1] - data[1]
    odd_acc_abs = np.abs(data[0][1] - data[1])
    dis2_acc = data[0][1] - data[0][0]
    dis2_acc_abs = np.abs(data[0][1] - data[0][0])
    accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

if len(accs) > 0:
    accs = np.array(accs)
    print ('Dis2 Invalid Odd Invalid:')
    print ('Dis2:', np.mean(accs[:, 0]), np.mean(accs[:, 1]))
    print ('ODD:', np.mean(accs[:, 2]), np.mean(accs[:, 3]))


# accs = []


# if len(accs) > 0:
#     accs = np.array(accs)
#     print ('Dis2 Invalid Odd Invalid (Dis2 <= ODD):')
#     print ('Dis2:', np.mean(accs[:, 0]), np.mean(accs[:, 1]))
#     print ('ODD:', np.mean(accs[:, 2]), np.mean(accs[:, 3]))

accs = []
for data in dis2_invalid_odd_valid['dis2_gt_odd']:
    odd_acc = data[0][1] - data[1]
    odd_acc_abs = np.abs(data[0][1] - data[1])
    dis2_acc = data[0][1] - data[0][0]
    dis2_acc_abs = np.abs(data[0][1] - data[0][0])
    accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

if len(accs) > 0:
    accs = np.array(accs)
    print ('Dis2 Invalid Odd Valid:')
    print ('Dis2:', np.mean(accs[:, 0]), np.mean(accs[:, 1]))
    print ('ODD:', np.mean(accs[:, 2]), np.mean(accs[:, 3]))

accs = []
for data in dis2_valid_odd_invalid['dis2_lte_odd']:
    odd_acc = data[0][1] - data[1]
    odd_acc_abs = np.abs(data[0][1] - data[1])
    dis2_acc = data[0][1] - data[0][0]
    dis2_acc_abs = np.abs(data[0][1] - data[0][0])
    accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

if len(accs) > 0:
    accs = np.array(accs)
    print ('Dis2 Valid Odd Invalid:')
    print ('Dis2:', np.mean(accs[:, 0]), np.mean(accs[:, 1]))
    print ('ODD:', np.mean(accs[:, 2]), np.mean(accs[:, 3]))

accs = []
for data in dis2_valid_odd_valid['dis2_gt_odd']:
    odd_acc = data[0][1] - data[1]
    odd_acc_abs = np.abs(data[0][1] - data[1])
    dis2_acc = data[0][1] - data[0][0]
    dis2_acc_abs = np.abs(data[0][1] - data[0][0])
    accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

for data in dis2_valid_odd_valid['dis2_lte_odd']:
    odd_acc = data[0][1] - data[1]
    odd_acc_abs = np.abs(data[0][1] - data[1])
    dis2_acc = data[0][1] - data[0][0]
    dis2_acc_abs = np.abs(data[0][1] - data[0][0])
    accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

accs = np.array(accs)
print ('Dis2 Valid Odd Valid:')
print ('Dis2:', np.mean(accs[:, 0]), np.mean(accs[:, 1]))
print ('ODD:', np.mean(accs[:, 2]), np.mean(accs[:, 3]))

# accs = []
# for data in dis2_valid_odd_valid['dis2_lte_odd']:
#     odd_acc = data[0][1] - data[1]
#     odd_acc_abs = np.abs(data[0][1] - data[1])
#     dis2_acc = data[0][1] - data[0][0]
#     dis2_acc_abs = np.abs(data[0][1] - data[0][0])
#     accs.append([dis2_acc, dis2_acc_abs, odd_acc, odd_acc_abs])

# accs = np.array(accs)
# print ('Dis2 Valid Odd Valid (Dis2 <= ODD):')
# print ('Dis2:', np.mean(accs[:, 0]), np.mean(accs[:, 1]))
# print ('ODD:', np.mean(accs[:, 2]), np.mean(accs[:, 3]))

print ('-' * 50)

for log in logit_dis2_preds:
    for i in range(len(log)):
        if type(log[i]) == torch.Tensor:
            log[i] = log[i].item()

for log in logit_odd_preds:
    for i in range(len(log)):
        if type(log[i]) == torch.Tensor:
            log[i] = log[i].item()

for log in logit_odd_other_preds:
    for i in range(len(log)):
        if type(log[i]) == torch.Tensor:
            log[i] = log[i].item()

ATC_preds = other_data[(other_data['prediction_method'] == 'ATC_NE') & (other_data['temperature'] == 'source')]
ATC_preds = ATC_preds[['lower_bound', 'trg_accuracy']].to_numpy()
COT_preds = other_data[(other_data['prediction_method'] == 'COT') & (other_data['temperature'] == 'source')]
COT_preds = COT_preds[['lower_bound', 'trg_accuracy']].to_numpy()
AC_preds = other_data[(other_data['prediction_method'] == 'AC') & (other_data['temperature'] == 'source')]
AC_preds = AC_preds[['lower_bound', 'trg_accuracy']].to_numpy()
DoC_preds = other_data[(other_data['prediction_method'] == 'DOC') & (other_data['temperature'] == 'source')]
DoC_preds = DoC_preds[['lower_bound', 'trg_accuracy']].to_numpy()

plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 18})
plt.scatter(logit_dis2_preds[:, 1], logit_dis2_preds[:, 0], s=50, label=r'DIS$^2$', zorder=1, marker='.', color='olive', alpha=0.7)
plt.scatter(logit_odd_preds[:, 1], logit_odd_preds[:, 0], s=50, label=r'ODD', zorder=2, marker='*', color='blue', alpha=0.7)

# Draw arrows between corresponding Dis^2 and ODD points
for dis2_point, odd_point in zip(logit_dis2_preds, logit_odd_preds):
    # Get x,y coordinates for both points
    x1, y1 = dis2_point[1], dis2_point[0]  # Dis^2 point
    x2, y2 = odd_point[1], odd_point[0]    # ODD point
    
    # Choose color based on whether ODD prediction is better
    arrow_color = 'blue' if y2 > y1 else 'red'
    
    # Draw arrow
    plt.arrow(x1, y1,               # Start point
              x2-x1, y2-y1,         # Delta x, delta y
              head_width=0.005,      # Arrow head width
              head_length=0.005,     # Arrow head length
              color=arrow_color,    # Color based on comparison
              alpha=0.5,            # Make slightly transparent
              length_includes_head=True)


plt.scatter(ATC_preds[:, 1], ATC_preds[:, 0], s=16, label='ATC', zorder=1, alpha=0.4)
plt.scatter(COT_preds[:, 1], COT_preds[:, 0], s=16, label='COT', zorder=1, alpha=0.4)
plt.scatter(AC_preds[:, 1], AC_preds[:, 0], s=16, label='AC', zorder=1, alpha=0.4)
plt.xlabel('Target Accuracy', fontsize=18)
plt.ylabel('Target Accuracy Prediction', fontsize=18)
plt.xlim(0.2, 1)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1., zorder=13)
plt.legend(loc='lower right', fontsize=14, ncol=2)


# plt.legend(loc='upper right', fontsize=14, bbox_to_anchor=(0, 1))
plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
# method_name = '_'.join(args.odd.split('/')[0].split('_')[1:])
method_name = args.odd.split('/')[0] + '_' + '_'.join(args.odd.split('/')[-1].split('.')[1].split('_')[1:])
name = os.path.join(args.plot_dir, f'{method_name}' + f'_{args.strategy}' + f'_{args.lb_type}' + ('_DA' if args.DA else ''))

print ('ODD:', np.mean(logit_odd_preds[:, 1] - logit_odd_preds[:, 0]))
print ('ODD (abs):', np.mean(np.abs(logit_odd_preds[:, 1] - logit_odd_preds[:, 0])))
print ('ODD weighted:', np.mean(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 0]))
print ('ODD weighted (abs):', np.mean(np.abs(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 0])))
print ('ODD weighted with eps:', np.mean(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 1]))
print ('ODD weighted with eps (abs):', np.mean(np.abs(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 1])))
print ('ODD new:', np.mean(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 2]))
print ('ODD new (abs):', np.mean(np.abs(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 2])))
print ('ODD new with eps:', np.mean(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 3]))
print ('ODD new with eps (abs):', np.mean(np.abs(logit_odd_other_preds[:, -1] - logit_odd_other_preds[:, 3])))
print ('Dis2:', np.mean(logit_dis2_preds[:, 1] - logit_dis2_preds[:, 0]))
print ('Dis2 (abs):', np.mean(np.abs(logit_dis2_preds[:, 1] - logit_dis2_preds[:, 0])))
print ('ATC:', np.mean(ATC_preds[:, 1] - ATC_preds[:, 0]))
print ('ATC (abs):', np.mean(np.abs(ATC_preds[:, 1] - ATC_preds[:, 0])))
print ('COT:', np.mean(COT_preds[:, 1] - COT_preds[:, 0]))
print ('COT (abs):', np.mean(np.abs(COT_preds[:, 1] - COT_preds[:, 0])))
print ('AC:', np.mean(AC_preds[:, 1] - AC_preds[:, 0]))
print ('AC (abs):', np.mean(np.abs(AC_preds[:, 1] - AC_preds[:, 0])))

print ('-' * 50)
print ('DIS2 (MAE):', np.mean(np.abs(logit_dis2_preds[:, 1] - logit_dis2_preds[:, 0])))
print ('DIS2 weps (MAE):', np.mean(np.abs(logit_dis2_preds[:, 1] - eps - logit_dis2_preds[:, 0])))
print ('ODD (MAE):', np.mean(np.abs(logit_odd_preds[:, 1] - logit_odd_preds[:, 0])))
print ('ODD weps (MAE):', np.mean(np.abs(logit_odd_preds[:, 1] - eps - logit_odd_preds[:, 0])))
print ('ATC (MAE):', np.mean(np.abs(ATC_preds[:, 1] - ATC_preds[:, 0])))
print ('COT (MAE):', np.mean(np.abs(COT_preds[:, 1] - COT_preds[:, 0])))
print ('AC (MAE):', np.mean(np.abs(AC_preds[:, 1] - AC_preds[:, 0])))
print ('DoC (MAE):', np.mean(np.abs(DoC_preds[:, 1] - DoC_preds[:, 0])))

invalid_dis2_preds = logit_dis2_preds[logit_dis2_preds[:, 0] > logit_dis2_preds[:, 1]]
invalid_dis2_weps_preds = logit_dis2_preds[logit_dis2_preds[:, 0] + eps > logit_dis2_preds[:, 1]]
dis2_invalid_eps = eps[logit_dis2_preds[:, 0] + eps > logit_dis2_preds[:, 1]]
invalid_odd_preds = logit_odd_preds[logit_odd_preds[:, 0] > logit_odd_preds[:, 1]]
invalid_odd_weps_preds = logit_odd_preds[logit_odd_preds[:, 0] + eps > logit_odd_preds[:, 1]]
odd_invalid_eps = eps[logit_odd_preds[:, 0] + eps > logit_odd_preds[:, 1]]
invalid_ATC_preds = ATC_preds[ATC_preds[:, 0] > ATC_preds[:, 1]]
invalid_COT_preds = COT_preds[COT_preds[:, 0] > COT_preds[:, 1]]
invalid_AC_preds = AC_preds[AC_preds[:, 0] > AC_preds[:, 1]]
invalid_DoC_preds = DoC_preds[DoC_preds[:, 0] > DoC_preds[:, 1]]

print ('DIS2 (coverage):', 1 - len(invalid_dis2_preds)/len(logit_dis2_preds), len(invalid_dis2_preds), len(logit_dis2_preds))
print ('DIS2 weps (coverage):', 1 - len(invalid_dis2_weps_preds)/len(logit_dis2_preds), len(invalid_dis2_weps_preds), len(logit_dis2_preds))
print ('ODD (coverage):', 1 - len(invalid_odd_preds)/len(logit_odd_preds), len(invalid_odd_preds), len(logit_odd_preds))
print ('ODD weps (coverage):', 1 - len(invalid_odd_weps_preds)/len(logit_odd_preds), len(invalid_odd_weps_preds), len(logit_odd_preds))
print ('ATC (coverage):', 1 - len(invalid_ATC_preds)/len(ATC_preds), len(invalid_ATC_preds), len(ATC_preds))
print ('COT (coverage):', 1 - len(invalid_COT_preds)/len(COT_preds), len(invalid_COT_preds), len(COT_preds))
print ('AC (coverage):', 1 - len(invalid_AC_preds)/len(AC_preds), len(invalid_AC_preds), len(AC_preds))
print ('DoC (coverage):', 1 - len(invalid_DoC_preds)/len(DoC_preds), len(invalid_DoC_preds), len(DoC_preds))

dis2_invalid_mae = np.abs(invalid_dis2_preds[:, 1] - invalid_dis2_preds[:, 0])
dis2_invalid_weps_mae = np.abs(invalid_dis2_weps_preds[:, 1] - dis2_invalid_eps - invalid_dis2_weps_preds[:, 0])
odd_invalid_mae = np.abs(invalid_odd_preds[:, 1] - invalid_odd_preds[:, 0])
odd_invalid_weps_mae = np.abs(invalid_odd_weps_preds[:, 1] - odd_invalid_eps - invalid_odd_weps_preds[:, 0])
atc_invalid_mae = np.abs(invalid_ATC_preds[:, 1] - invalid_ATC_preds[:, 0])
cot_invalid_mae = np.abs(invalid_COT_preds[:, 1] - invalid_COT_preds[:, 0])
ac_invalid_mae = np.abs(invalid_AC_preds[:, 1] - invalid_AC_preds[:, 0])
doc_invalid_mae = np.abs(invalid_DoC_preds[:, 1] - invalid_DoC_preds[:, 0])


print ('DIS2 (invalid MAE):', np.mean(dis2_invalid_mae))
print ('DIS2 weps (invalid MAE):', np.mean(dis2_invalid_weps_mae))
print ('ODD (invalid MAE):', np.mean(odd_invalid_mae))
print ('ODD weps (invalid MAE):', np.mean(odd_invalid_weps_mae))
print ('ATC (invalid MAE):', np.mean(atc_invalid_mae))
print ('COT (invalid MAE):', np.mean(cot_invalid_mae))
print ('AC (invalid MAE):', np.mean(ac_invalid_mae))
print ('DoC (invalid MAE):', np.mean(doc_invalid_mae))



plt.savefig(f'../real_{args.strategy}_{args.DA}.pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')
