import numpy as np
import input
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

file_path = "../Data/Train.txt"

def plot_simple():
    # Your code for reading data and plotting goes here
    data = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            elements = line.split(",")
            data.append([float(element) for element in elements])
            num_params = len(data[0]) - 1
    
    labels = ["Parameter {}".format(i) for i in range(num_params)]

    for i in range(num_params):
        fig, ax = plt.subplots(figsize=(8, 6))
        # Your code for plotting histograms goes here
        # Use ax.hist to plot histograms for each class
        x_values = [entry[i] for entry in data]
        y_values = [entry[i] for entry in data]
        classes = [entry[-1] for entry in data]

        x_class0 = [x for x, c in zip(x_values, classes) if c == 0]
        x_class1 = [x for x, c in zip(x_values, classes) if c == 1]

        ax.hist(x_class0, bins=30, color='blue', alpha=0.5, density=True, label='Class 0')
        ax.hist(x_class1, bins=30, color='red', alpha=0.5, density=True, label='Class 1')
        
        ax.set_xlabel(labels[i], fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.legend()
    
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)  # spacing between graphs
    
        plt.savefig('plot_{}.png'.format(i))  # save each plot as a separate PNG file
    
        plt.close() 
    
    
def plot_multiple():
    data = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            elements = line.split(",")
            data.append([float(element) for element in elements])

    num_params = len(data[0]) - 1
    labels = ["Parameter {}".format(i) for i in range(num_params)]

    num_rows = 5
    num_cols = 2

    fig = plt.figure(figsize=(12, 6 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)

    for i in range(num_params):
        row = i % num_rows
        col = i // num_rows

        ax = fig.add_subplot(gs[row, col])
        
        x_values = [entry[i] for entry in data]
        y_values = [entry[i] for entry in data]
        classes = [entry[-1] for entry in data]

        x_class0 = [x for x, c in zip(x_values, classes) if c == 0]
        x_class1 = [x for x, c in zip(x_values, classes) if c == 1]

        ax.hist(x_class0, bins=30, color='blue', alpha=0.5, density=True, label='Class 0')
        ax.hist(x_class1, bins=30, color='red', alpha=0.5, density=True, label='Class 1')
        ax.set_xlabel(labels[i])
        ax.set_ylabel("{}".format(labels[i]))
        ax.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) # spacing between graphs

    plt.savefig('istogrammi.png') # save in .png

    plt.show()

def compute_pearson_coeff(X, Y):
    N = X.shape[0]
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    numerator = 1/N * np.dot(X,Y.T) - mean_y * mean_x
    denominator = np.sqrt(1/N**2 * np.dot(X-mean_x,X.T-mean_x) * np.dot(Y-mean_y,Y.T-mean_y))

    corr = numerator / denominator
    return corr

def plot_correlations(D, cmap="Greys"):
    corr = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = D[x, :]
            Y = D[y, :]
            pearson_coeff = compute_pearson_coeff(X, Y)
            corr[x][y] = pearson_coeff
    sns.set()
    sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    plt.show()

if __name__ == '__main__':
    D, L = input.load(input.traininput)
    plot_simple()
    #plot_correlations(D)

    