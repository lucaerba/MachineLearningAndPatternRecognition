import numpy as np
import input
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from prettytable import PrettyTable
import pandas as pd
from model import PCA, LDA

file_path = input.traininput

def plot_simple():
    data = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            elements = line.split(",")
            data.append([float(element) for element in elements])
            num_params = len(data[0]) - 1
    
    labels = ["Parameter {}".format(i+1) for i in range(num_params)]

    for i in range(num_params):
        x_values = [entry[i] for entry in data]
        classes = [entry[-1] for entry in data]

        x_class0 = [x for x, c in zip(x_values, classes) if c == 0]
        x_class1 = [x for x, c in zip(x_values, classes) if c == 1]

        plt.hist(x_class0, bins=70, color='blue', alpha=0.4, density=True, label='Class 0', linewidth=1.0, edgecolor='black')
        plt.hist(x_class1, bins=70, color='red', alpha=0.4, density=True, label='Class 1', linewidth=1.0, edgecolor='black')
        plt.legend(fontsize=16)

        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.title(labels[i], fontsize=24)
    
        plt.savefig('../Plots/plot_{}.png'.format(i))
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

def plot_correlations(D, cmap="Greys",dimensions=10):
    corr = np.zeros((dimensions, dimensions))
    for x in range(dimensions):
        for y in range(dimensions):
            X = D[x, :]
            Y = D[y, :]
            pearson_coeff = compute_pearson_coeff(X, Y)
            corr[x][y] = pearson_coeff
    sns.set()
    sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=True)
    plt.savefig("../Plots/correlations")

def plot_Scatter(DTR, LTR):
    idx = 0
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[0]):
            if i != j:
                plt.figure()
                plt.scatter(DTR[i, LTR == 0], DTR[j, LTR == 0], label="0")
                plt.scatter(DTR[i, LTR == 1], DTR[j, LTR == 1], label="1")
                plt.legend()
                plt.xlabel("Feature " + str(i))
                plt.ylabel("Feature " + str(j))
                plt.savefig("../Plots/scatter_" + str(idx))
                idx += 1
                plt.close()

def plot_PCA(D,L,m,s=16):

    D_class1 = D[:,L == 1]
    D_class0 = D[:,L == 0]

    for j in range(m):
        for i in range(m):
            if i != j: # otherwise useless plots, data on same line
                plt.figure()
                plt.scatter(D_class1[j, :], D_class1[i, :], label='class 1', s=s, alpha=0.8)
                plt.scatter(D_class0[j, :], D_class0[i, :], label='class 0', s=s, alpha=0.8)
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                # plt.title("PCA scatter plot (m={})".format(m))
                plt.legend(fontsize=16,s=30)
                plt.savefig("../Plots/PCA_j={}_i={}.png".format(j,i))
                # plt.show()

def plot_LDA(D, L):
    D_class1 = D[:, L == 1]
    D_class0 = D[:, L == 0]

    plt.figure()
    plt.hist(D_class1[0, :], bins=70, alpha=1, label='class 1', linewidth=1.0, edgecolor='black')
    plt.hist(D_class0[0, :], bins=70, alpha=0.5, label='class 0', linewidth=1.0, edgecolor='black')
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=16)
    plt.savefig('../Plots/LDA.png')
    plt.show()
    plt.close()

gmm_file = '../FINALoutputs/gmm_output.txt'
mvg_file = '../FINALoutputs/mvg_output.txt'
nb_file = '../FINALoutputs/nb_output.txt'
tmvg_file = '../FINALoutputs/tmvg_output.txt'
tnb_file = '../FINALoutputs/tnb_output.txt'
logreg_file = '../FINALoutputs/logreg_output.txt'

fields_gmm = ['PCA','g (target)','g (NON target)','function', 'Working Point','minDCF','C_prim']
fields_gaussians = ['PCA', 'Type', 'Working Point', 'minDCF', 'C_prim']
fields_logreg = ['PCA', 'Type', 'lambda', 'Working Point', 'minDCF', 'C_prim']

def PrettyTab_to_data(tab, fields, model):
    with open(tab, 'r') as file:
        table_data = file.read()

    table = PrettyTable()
    table.border = False
    table.header = False
    table.align = 'l'
    table.field_names = fields
    for line in table_data.splitlines():
        if '|' in line:
            data = [cell.strip() for cell in line.split('|') if cell.strip()]
            table.add_row(data)

    with open('../FINALoutputs/table_data.csv', 'w', newline='') as csvfile:
        csvfile.write(table.get_csv_string())

    if model == 'gmm' or model == 'gaussian':
        with open('../FINALoutputs/table_data.csv', 'r') as csvfile:
            array = pd.read_csv(csvfile)
            array = array.values.tolist()
            array = np.array(array)
            C_prim_list = []
            for el in array[:,array.shape[1]-1]:
                if el != '-':
                    C_prim_list.append(float(el))
        return C_prim_list
    elif model == 'logreg':
        with open('../FINALoutputs/table_data.csv', 'r') as csvfile:
            array = pd.read_csv(csvfile)
            array = array.values.tolist()
            array = np.array(array)
            C_prim_list_lin = []
            C_prim_list_quad = []
            lam_list_lin = []
            lam_list_quad = []
            for el in array[2::6,array.shape[1]-1]:
                C_prim_list_lin.append(float(el))
            for el in array[5::6, array.shape[1] - 1]:
                C_prim_list_quad.append(float(el))
            for el in array[0::6,2]:
                lam_list_lin.append(float(el))
            for el in array[3::6,2]:
                lam_list_quad.append(float(el))
        return C_prim_list_lin, C_prim_list_quad, lam_list_lin, lam_list_quad



def C_prim_plot(C_prim_list, ind_PCA=0, model='gmm', l=''):

    C_prim_tot = len(C_prim_list)

    tot_PCA = ['No PCA'] + [a for a in range(2,10)]
    C_prim_list_forPCA = np.reshape(C_prim_list, (9,int(C_prim_tot/9))) # now each row represents a different PCA

    if model == 'gmm':
        n = 5  # first n K_target values
        C_prim_list_fullMVG = C_prim_list_forPCA[:,0::3] # C_prim values are stripped
        C_plot_data = np.reshape(C_prim_list_fullMVG[ind_PCA][:int(5*n)], (n,5)) # only the first n K_target are selected
        width = 0.15  # the width of the bars
        multiplier = 0
        K = [1, 2, 4, 8, 16]
        fig, ax = plt.subplots(layout='constrained')
        colors = ['navy', 'mediumblue', 'blue', 'mediumslateblue', 'blueviolet']

        for i in range(len(K)):
            offset = width * multiplier
            ax.bar(np.arange(n) + offset, C_plot_data[:, i], width, label=f'NON target K: {2 ** i}', color=colors[i])
            multiplier += 1
        if ind_PCA == 0:
            plt.text(0.27,0.21,'{:.3f}'.format(np.min(C_plot_data)))
        ax.set_ylabel('C_prim')
        ax.set_xlabel('Target K')
        ax.set_title(f'PCA value :{tot_PCA[ind_PCA]}')
        ax.set_xticks(np.arange(n) + width * 2, K[:n])
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 1)
        plt.savefig(f'../Plots/gmm_Cprim_PCA: {tot_PCA[ind_PCA]}')
        # plt.show()
        plt.close()

    elif model == 'MVG':
        plt.figure()
        tot_PCA = [tot_PCA[0]] + tot_PCA[::-1][:-1]
        C_prim_list_forPCA = list(C_prim_list_forPCA.ravel())
        C_plot_data = [C_prim_list_forPCA[0]] + C_prim_list_forPCA[::-1][:-1]
        plt.plot(tot_PCA, C_plot_data)
        plt.title('Primary cost for MVG model')
        plt.xlabel('PCA value')
        plt.ylabel('C_prim')
        # plt.ylim(0,1)
        # plt.xlim(0,8)
        plt.annotate('{:.3f}'.format(np.min(C_plot_data)),  # this is the text
                 (2,C_plot_data[1]-0.01),  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')
        plt.grid()
        # plt.savefig('../Plots/MVG_Cprim_PCA')
        plt.show()
        plt.close()

    elif model == 'logreg':
        plt.figure()
        l = l[0:11]
        C_plot_data = C_prim_list_forPCA[ind_PCA,:]
        plt.plot(l, C_plot_data)
        plt.title(f'C_prim for logreg (quadratic) model, PCA : {tot_PCA[ind_PCA]}')
        plt.xlabel('λ')
        plt.xscale('log')
        plt.ylabel('C_prim')
        if ind_PCA == 7:
            plt.annotate('{:.3f}'.format(np.min(C_plot_data)),  # this is the text
                     (1e-5,C_plot_data[1]-0.01),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')
        ax = plt.gca()
        ax.set_xticks(l)
        ax.set_xlim(min(l)-100, max(l)+10000)
        d = (max(C_plot_data)-min(C_plot_data))/10
        ax.set_ylim(min(C_plot_data)-d,max(C_plot_data)+d)
        ax.set_ylim(0,1)
        plt.grid()
        # plt.savefig(f'../Plots/logreg_quad_PCA: {tot_PCA[ind_PCA]}')
        plt.show()
        plt.close()








if __name__ == '__main__':
    # D, L = input.load(input.traininput)

    # m = 4
    # DP = PCA(D,m)
    # plot_PCA(DP,L,m)

    # DP = LDA(D,L,m=1)
    # plot_LDA(DP,L)

    # plot_simple()
    # plot_correlations(D)
    # plot_correlations(D[:,L == 1], 'Reds')
    # plot_correlations(D[:,L == 0], 'Blues')

    # C = PrettyTab_to_data(gmm_file, fields_gmm)
    # for i in range(9):
    #     C_prim_plot(C, i, model='gmm')

    # C = PrettyTab_to_data(nb_file, fields_gaussians)
    # C_prim_plot(C, model='MVG')

    C_lin, C_quad, l_lin, l_quad = PrettyTab_to_data(logreg_file, fields_logreg, model='logreg')
    for i in range(9):
        C_prim_plot(C_quad, i, model='logreg', l=l_quad)