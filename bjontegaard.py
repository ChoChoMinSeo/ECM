import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from numpy import linalg

from matplotlib import pyplot as plt


def integrate(coeff, X, degree=3):
    ret = 0
    for i in range(degree + 1):
        ret += coeff[i] * (X ** (i + 1)) / (i + 1)
    return ret


def polyfit_and_integrate(X, Y, lowX, highX):
    degree = 3
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X = poly_features.fit_transform(np.expand_dims(X, axis=-1))
    Y = np.expand_dims(Y, axis=-1)

    # polyfit
    coeff = np.matmul(np.matmul(linalg.inv(np.matmul(X.T, X)), X.T), Y)[:, 0]

    # integrate
    int1 = integrate(coeff, lowX, degree=degree)
    int2 = integrate(coeff, highX, degree=degree)

    return int2 - int1


def bjontegaard_diff(r1, d1, r2, d2):
    rL = np.array([max(r1.min(), r2.min())])
    rH = np.array([min(r1.max(), r2.max())])

    int1 = polyfit_and_integrate(r1, d1, rL, rH)
    int2 = polyfit_and_integrate(r2, d2, rL, rH)

    return (int2 - int1) / (rH - rL)


def bjontegaard_diff_deltabpp(r1, d1, r2, d2, r1_from, r2_to):
    int1 = polyfit_and_integrate(r1, d1, r1_from, r2_to)
    int2 = polyfit_and_integrate(r2, d2, r1_from, r2_to)

    diff = (int2 - int1) / (r2_to - r1_from)
    return 100 * (math.e ** diff - 1)


def BDMetrics(r1, d1, r2, d2):
    r1 = np.log(r1)
    r2 = np.log(r2)

    BDSNR = bjontegaard_diff(r1, d1, r2, d2).item()
    BDBR = 100 * (math.e ** bjontegaard_diff(d1, r1, d2, r2).item() - 1)

    return BDBR, BDSNR

if __name__ == "__main__":
    import pandas as pd
    from build_dataset import initial_token_stats, task_colors
    root = '../exp_results/pretrained_models/v016_unfreeze/stats/'
    n_datas = {
        'Kodak':24,
        'Y1Y2Y3':76,
        'Residue':76,
        'Document':24,
        'KITTI':38,
        'LineDrawing':32,
        'natural_clic':41,
    }
    per_mode = {'total':{'psnr':[0.0]*6,'rate':[0.0]*6}}
    for mode in list(n_datas.keys()):
        per_mode[mode] = {'psnr':[], 'rate':[]}
    for i in range(6):
        df = pd.read_csv(root+f'qs{i}.csv', index_col=0)
        for mode in list(n_datas.keys()):
            per_mode[mode]['psnr'].append(df.loc[mode]['avg psnr'])
            per_mode[mode]['rate'].append(df.loc[mode]['avg rate loss'])
            per_mode['total']['psnr'][i] += df.loc[mode]['avg psnr']/7
            per_mode['total']['rate'][i] += df.loc[mode]['avg rate loss']/7
    initial_total = {'psnr':[0.0]*6,'rate':[0.0]*6}
    for mode in list(n_datas.keys()):
        bd_rate, _ = BDMetrics(r1 = np.array(initial_token_stats[mode]['rate']), d1 = np.array(initial_token_stats[mode]['psnr']),
                  r2 = np.array(per_mode[mode]['rate']), d2 = np.array(per_mode[mode]['psnr']))
        print(mode, f'{bd_rate:.3f}')
        for qs in range(6):
            initial_total['psnr'][qs] += initial_token_stats[mode]['psnr'][qs]/7
            initial_total['rate'][qs] += initial_token_stats[mode]['rate'][qs]/7
        
    bd_rate, _ = BDMetrics(r1 = np.array(initial_total['rate']), d1 = np.array(initial_total['psnr']),
            r2 = np.array(per_mode['total']['rate']), d2 = np.array(per_mode['total']['psnr']))
    print('total', f'{bd_rate:.3f}')
    
    ########## RD curve #############
    fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    for i,task_name in enumerate(list(n_datas.keys())):
        cur_rate = per_mode[mode]['rate']
        cur_psnr = per_mode[mode]['psnr']
        axes.plot(cur_rate,cur_psnr,"o--", c = task_colors[task_name],label = task_name)
        axes.plot(initial_token_stats[task_name]['rate'],initial_token_stats[task_name]['psnr'],"*--", c = task_colors[task_name])
        
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=3)
    fig.tight_layout()
    axes.grid(True)
    fig.savefig("../exp_results/debug/checkpoint/rd_curve.png")
    fig.clf()