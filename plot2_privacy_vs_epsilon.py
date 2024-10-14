import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
})

if __name__ == "__main__":
    
    eps_values = [1, 10, 100, 1000]
    max_depths = [5, 10]

    n_samples_list = ['10', '100', '1000']

    n_clients = 18

    rates_dict = {n_samples: [] for n_samples in n_samples_list}

    for eps in eps_values:
        for max_depth in max_depths:
            
            df_ = pd.read_csv(f"results/federated/per_partition_{eps}_d{max_depth}.csv")
            df_ = df_.groupby('trial').mean().reset_index(drop=False)
            rates_min = df_[n_samples_list].min()
            rates_max = df_[n_samples_list].max()
            rates_mean = df_[n_samples_list].mean()

            for n_samples in n_samples_list:
                rates_dict[n_samples].append({
                    'epsilon': eps,
                    'max_depth': max_depth,
                    'n_samples': n_samples,
                    'rate_mean': rates_mean[n_samples],
                    'rate_min': rates_min[n_samples],
                    'rate_max': rates_max[n_samples]
                })



    rates_dfs = {n_samples: pd.DataFrame(rates_dict[n_samples]) for n_samples in n_samples_list}


    colors = ['tab:blue', 'tab:red']
    linetype = ['s-', 'd--', '*:']
    plt.figure(figsize=(5, 3))
    x_range = np.arange(len(eps_values))
    shift = 0.1
    for i, n_samples in enumerate(n_samples_list):
        for j, max_depth in enumerate(max_depths):
            df_ = rates_dfs[n_samples][rates_dfs[n_samples]['max_depth'] == max_depth]
            # plt.bar(
            #     x_range + (j-1)*shift, df_['rate_mean'], width=.5*shift, 
            #     yerr=(df_['rate_mean'] - df_['rate_min'], df_['rate_max'] - df_['rate_mean']), 
            #     label=f"$d$={max_depth}, $n$={n_samples}", color=colors[j]
            # )
            plt.semilogx(df_['epsilon'], df_['rate_mean'], linetype[i], label=f"$d$={max_depth}, $n$={n_samples}", color=colors[j])
            # plt.errorbar(df_['epsilon'], df_['rate_mean'], yerr=(df_['rate_mean'] - df_['rate_min'], df_['rate_max'] - df_['rate_mean']), fmt='none', capsize=4, color='k')
    # plt.xticks(x_range, eps_values)
    # plt.xlim([0.9*eps_values[0], 1.1*eps_values[-1]])

    plt.ylim([0, 1])
    plt.grid(linestyle=':', which='both')
    plt.xlabel('$\epsilon$', fontsize=14)
    plt.ylabel('Memorization rate')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig/privacy_vs_epsilon.pdf')
    plt.show()

    # colors = ['tab:blue', 'tab:red']
    # plt.figure(figsize=(6, 3))
    # for i, max_depth in enumerate(max_depths):
    #     df_ = df[df['max_depth'] == max_depth]
    #     plt.semilogx(df_['epsilon'], df_['rate'], 's-', label=f"d={max_depth}", color=colors[i])
    # plt.xlim([0.9*eps_values[0], 1.1*eps_values[-1]])
    # plt.ylim([0, 1])
    # plt.grid(linestyle=':')
    # plt.xlabel('Epsilon')
    # plt.ylabel('Attack Success Rate')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()