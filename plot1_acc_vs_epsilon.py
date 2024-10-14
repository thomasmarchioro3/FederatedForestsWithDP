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

    results = []
    for eps in eps_values:
        for max_depth in max_depths:
            acc = pd.read_csv(f"results/federated/global_{eps}_d{max_depth}.csv")['accuracy']
            results.append({"epsilon": eps, "max_depth": max_depth, "acc_mean": acc.mean(), "acc_min": acc.min(), "acc_max": acc.max()})

    df = pd.DataFrame(results)

    colors = ['tab:blue', 'tab:red']
    plt.figure(figsize=(5, 3))
    for i, max_depth in enumerate(max_depths):
        df_ = df[df['max_depth'] == max_depth]
        plt.semilogx(df_['epsilon'], df_['acc_mean'], 's-', label=f"$d$={max_depth}", color=colors[i])
        plt.errorbar(df_['epsilon'], df_['acc_mean'], yerr=(df_['acc_mean'] - df_['acc_min'], df_['acc_max'] - df_['acc_mean']), fmt='none', capsize=4, color='k')
    plt.xlim([0.9*eps_values[0], 1.1*eps_values[-1]])
    plt.ylim([0, 1])
    plt.grid(linestyle=':', which='both')
    plt.xlabel('$\epsilon$', fontsize=14)
    plt.ylabel('Accuracy')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig/acc_vs_epsilon.pdf')
    plt.show()