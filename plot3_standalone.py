import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("results/standalone/global.csv")

    x_range = range(1, len(df)+1)
    plt.figure(figsize=(5, 3))
    plt.bar(x_range, df["0"])
    plt.xticks(x_range)
    plt.xlabel('Client')
    plt.ylim(0, 1)
    plt.ylabel('Standalone accuracy (no DP)')
    plt.tight_layout()
    plt.savefig('fig/acc_standalone.pdf')
    plt.draw()

    plt.show()