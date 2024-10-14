# Federated Forests with Differential Privacy

Code used in the paper "Federated Forests with Differential Privacy for Distributed Wearable Sensors ".

Requirements (older versions of these libraries may also work but have not been tested):
- Numpy >= 1.26
- Pandas >= 2.1
- Matplotlib >= 3.8

Steps to reproduce the paper's results:
1. Clone the repo and navigate to the main directory
2. Run `bash setup.sh` if on Linux, or follow the instructions inside the `setup.sh` file if on Windows/MacOS
3. Run `python -m utils.split_dataset`
4. Run `python -m local training`
5. Run `python -m train_centralized`
6. Run `python -m train_federated_forests --epsilon <EPSILON> --max_depth <MAX_DEPTH>` (replace `<EPSILON> ` with 1, 10, 100, and 1000, and `<MAX_DEPTH>` with 5, and 10)
7. Run `python -m plot1_acc_vs_epsilon`:

![image](fig/acc_vs_epsilon.pdf)

8: Run `python -m plot2_privacy_vs_epsilon`:

![image](fig/privacy_vs_epsilon.pdf)

9: Run `python -m plot3_standalone.py`:

![image](fig/acc_standalone.pdf)