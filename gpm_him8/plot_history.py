import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pickle
import math
import seaborn as sns
sns.set()


def plot_val_history(hist_file): 
    h = pickle.load(open(hist_file, "rb"))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean')
    ax1.set_title('Validation')
    ax1.plot(h['val_mean_yhat'], '-b')
    ax2.set_ylabel('Variance')
    ax2.plot(h['val_var_yhat'], '--r')
    ax2.grid(False)

    fig.legend(["Mean", 'Variance'], loc='upper right')

    plt.savefig('{}_val.png'.format(hist_file[:-4]))


def plot_train_history(hist_file): 
    h = pickle.load(open(hist_file, "rb"))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean')
    ax1.set_title('Train')
    ax1.plot(h['mean_yhat'], '-b')
    ax2.set_ylabel('Variance')
    ax2.plot(h['var_yhat'], '--r')
    ax2.grid(False)

    fig.legend(["Mean", 'Variance'], loc='upper right')

    plt.savefig('{}_train.png'.format(hist_file[:-4]))


def plot_mse(hist_file): 
    h = pickle.load(open(hist_file, "rb"))
    plt.clf()
    plt.plot(h['mse'], label='train')
    plt.plot(h['val_mse'], label='validation')

    plt.legend(["train", 'validation'], loc='upper right')
    plt.title('MSE')
    plt.savefig('{}_mse.png'.format(hist_file[:-4]))

def plot_loss(hist_file): 
    h = pickle.load(open(hist_file, "rb"))
    plt.clf()
    plt.plot(h['loss'], label='train')
    plt.plot(h['val_loss'], label='validation')

    plt.legend(["train", 'validation'], loc='upper right')
    plt.title('Conv 5 Loss')
    plt.savefig('{}_5convloss.png'.format(hist_file[:-4]))

"""
plot_train_history("history_conv5_beta1.pkl")
plot_train_history("history_conv5_beta2.pkl")
plot_train_history("history_conv5_beta4.pkl")
plot_train_history("history_conv5_beta8.pkl")
plot_train_history("history_mse.pkl")

plot_val_history("history_conv5_beta1.pkl")
plot_val_history("history_conv5_beta2.pkl")
plot_val_history("history_conv5_beta4.pkl")
plot_val_history("history_conv5_beta8.pkl")
plot_val_history("history_mse.pkl")
"""

for alpha in [1,2,4]:
    for beta in [1,2,4]:
        hist_file = "history_2months_150epochs_4chan_conv5_alpha{}_beta{}.pkl".format(alpha, beta)
        plot_train_history(hist_file)
        plot_val_history(hist_file)
        plot_mse(hist_file)
        plot_loss(hist_file)

plot_mse("history_2months_150epochs_mse.pkl")
