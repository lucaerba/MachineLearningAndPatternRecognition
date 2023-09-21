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

def vcol(v):
    v = v.reshape((v.size, 1))
    return v

def vrow(v):
    v = v.reshape((1, v.size))
    return v

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rc('font', weight='bold')

def plot_simple():
    data = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            elements = line.split(",")
            data.append([float(element) for element in elements])
            num_params = len(data[0]) - 1
    
    labels = ['\\textbf{' + "Parameter {}".format(i+1) + '}' for i in range(num_params)]

    for i in range(num_params):
        x_values = [entry[i] for entry in data]
        classes = [entry[-1] for entry in data]

        x_class0 = [x for x, c in zip(x_values, classes) if c == 0]
        x_class1 = [x for x, c in zip(x_values, classes) if c == 1]
        plt.figure()

        plt.hist(x_class0, bins=50, color='blue', alpha=0.4, density=True, label='\\textbf{Class 0}', linewidth=1.0, edgecolor='black')
        plt.hist(x_class1, bins=50, color='red', alpha=0.4, density=True, label='\\textbf{Class 1}', linewidth=1.0, edgecolor='black')
        plt.legend(fontsize=16)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)

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

def plot_correlations(D, cmap="Greys",dimensions=10, subset='both classes'):
    corr = np.zeros((dimensions, dimensions))
    for x in range(dimensions):
        for y in range(dimensions):
            X = D[x, :]
            Y = D[y, :]
            pearson_coeff = compute_pearson_coeff(X, Y)
            corr[x][y] = pearson_coeff
    plt.figure()
    sns.set()
    sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=True)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.title('\\textbf{' + f'heatmap for {subset}' + '}', fontsize=20, y=1.05)
    plt.savefig(f"../Plots/correlations_{subset}")
    # plt.show()
    plt.close()

def explained_variance_plot(D):
    N = len(D)
    mu = D.mean(1)
    DC = D - vcol(mu)
    C = N ** -1 * np.dot(DC, DC.T)
    s, U = np.linalg.eigh(C)

    explained_variance = np.cumsum(s[::-1] / np.sum(s[::-1]))
    explained_variance = np.append([0], explained_variance)

    major_ticksx = np.arange(0, 11, 1)
    minor_ticksx = np.arange(0, 11, 0.2)
    major_ticksy = np.arange(0, 2, 0.1)
    minor_ticksy = np.arange(0, 2, 0.02)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(major_ticksx)
    ax.set_xticks(minor_ticksx, minor=True)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    x_coord = np.arange(0,11,1)
    plt.plot(x_coord,explained_variance, '-o')
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.xlabel('\\textbf{Dimensions}', fontsize=16)
    plt.title('\\textbf{Explained covariance}', fontsize=20)
    for x, y in zip(x_coord,explained_variance):
        if x == 0:
            plt.text(x + 0.2, y + 0.01, '\\textbf{' + f'{y:.3f}' + '}')
        else:
            plt.text(x-0.85,y+0.01,'\\textbf{' + f'{y:.3f}' + '}')
    plt.savefig(f"../Plots/explained_variance")
    # plt.show()
    plt.close()

def plot_Scatter(DTR, LTR):
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.5, wspace=0.5)
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            elements = line.split(",")
            data.append([float(element) for element in elements])

    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[0]):
            if i != j:
                ax = axes[i, j]
                ax.scatter(DTR[i, LTR == 0], DTR[j, LTR == 0], alpha=0.2, color='red')
                ax.scatter(DTR[i, LTR == 1], DTR[j, LTR == 1], alpha=0.2, color='blue')
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                # ax.xlabel("\\textbf{Feature} " + str(i), fontsize=16)
                # ax.ylabel("\\textbf{Feature} " + str(j), fontsize=16)
                # plt.savefig("../Plots/scatter_" + str(idx))
            else:
                ax = axes[i, j]
                x_values = [entry[i] for entry in data]
                classes = [entry[-1] for entry in data]

                x_class0 = [x for x, c in zip(x_values, classes) if c == 0]
                x_class1 = [x for x, c in zip(x_values, classes) if c == 1]
                ax.hist(x_class0, bins=30, alpha=0.4, density=True, linewidth=1.0, color='red')
                ax.hist(x_class1, bins=30, alpha=0.4, density=True, linewidth=1.0, color='blue')
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
    # ax.legend(fontsize=16)
    # plt.show()
    plt.savefig('../Plots/Scatter_biggrid.png', bbox_inches='tight')
    plt.close()

def plot_PCA(D,L,m,s=16):

    D_class1 = D[:,L == 1]
    D_class0 = D[:,L == 0]

    for j in range(m):
        for i in range(m):
            if i != j: # otherwise useless plots, data on same line
                plt.figure()
                plt.scatter(D_class1[j, :], D_class1[i, :], label='\\textbf{class 1}', s=s, alpha=0.8)
                plt.scatter(D_class0[j, :], D_class0[i, :], label='\\textbf{class 0}', s=s, alpha=0.8)
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                # plt.title("PCA scatter plot (m={})".format(m))
                plt.legend(fontsize=16)
                plt.savefig("../Plots/PCA_j={}_i={}.png".format(j,i))
                # plt.show()

def plot_LDA(D, L):
    D_class1 = D[:, L == 1]
    D_class0 = D[:, L == 0]

    plt.figure()
    plt.hist(D_class1[0, :], bins=70, alpha=1, label='\\textbf{class 1}', linewidth=1.0, edgecolor='black')
    plt.hist(D_class0[0, :], bins=70, alpha=0.5, label='\\textbf{class 0}', linewidth=1.0, edgecolor='black')
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.legend(fontsize=16)
    plt.savefig('../Plots/LDA.png')
    # plt.show()
    plt.close()

gmm_file = '../FINALoutputs/gmm_output.txt'
mvg_file = '../FINALoutputs/mvg_output.txt'
nb_file = '../FINALoutputs/nb_output.txt'
tmvg_file = '../FINALoutputs/tmvg_output.txt'
tnb_file = '../FINALoutputs/tnb_output.txt'
logreg_file = '../FINALoutputs/logreg_output.txt'
logreg_Znorm_file = '../FINALoutputs/logreg_output_Znorm.txt'
svm_file = '../FINALoutputs/svm_output.txt'

fields_gmm = ['PCA','g (target)','g (NON target)','function', 'Working Point','minDCF','C_prim']
fields_gaussians = ['PCA', 'Type', 'Working Point', 'minDCF', 'C_prim']
fields_logreg = ['PCA', 'Type', 'lambda', 'Working Point', 'minDCF', 'C_prim']
fields_svm = ['PCA', 'Kernel', 'c', 'degree', 'gamma',  'K', 'Working Point',  'minDCF',  'C_prim']

def PrettyTab_to_data(tab, fields, model, Znorm=False):
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
            C_Znorm = []
            if Znorm == False:
                for el in array[2::6,array.shape[1]-1]:
                    C_prim_list_lin.append(float(el))
                for el in array[5::6, array.shape[1] - 1]:
                    C_prim_list_quad.append(float(el))
                return C_prim_list_lin, C_prim_list_quad
            else:
                for el in array[2::3,array.shape[1]-1]:
                    C_Znorm.append(float(el))
                return C_Znorm
    elif model == 'svm':
        with open('../FINALoutputs/table_data.csv', 'r') as csvfile:
            array = pd.read_csv(csvfile)
            array = array.values.tolist()
            array = np.array(array)
            flag_lin = array[0::3,1] == 'Linear'
            flag_d2 = array[0::3,3] == '2'
            flag_d3 = array[0::3, 3] == '3'
            flag_RBF = array[0::3, 1] == 'RBF'
            C_prim = np.asarray(array[2::3, -1], dtype = np.float64)
            if flag_lin.any() == True:
                C_prim_list_lin = C_prim[flag_lin]
            if flag_d2.any() == True:
                C_prim_list_poly2 = C_prim[flag_d2]
            if flag_d3.any() == True:
                C_prim_list_poly3 = C_prim[flag_d3]
            if flag_RBF.any() == True:
                C_prim_list_RBF = C_prim[flag_RBF]
            return C_prim_list_lin, C_prim_list_poly2, C_prim_list_poly3, C_prim_list_RBF



def C_prim_plot(C_prim_list, ind_PCA=0, model='gmm', l='', C_Znorm=[], j=0):

    if model == 'gmm':
        C_prim_tot = len(C_prim_list)

        if C_Znorm != []:
            C_Znorm = np.reshape(C_Znorm, (4, int(len(C_Znorm) / 4)))

        tot_PCA = ['No PCA'] + [a for a in range(2, 10)]
        C_prim_list_forPCA = np.reshape(C_prim_list, (9, int(C_prim_tot / 9)))
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
            ax.bar(np.arange(n) + offset, C_plot_data[:, i], width, label='\\textbf{' + f'NON target K: {2 ** i}' + '}', color=colors[i])
            multiplier += 1
        if ind_PCA == 0:
            plt.text(0.27,0.21,'${:.3f}$'.format(np.min(C_plot_data)), fontsize=14)
        ax.set_ylabel('\\textbf{$C_{prim}$}', fontsize=16)
        ax.set_xlabel('\\textbf{Target K}', fontsize=16)
        ax.set_title('\\textbf{' + f'PCA value: {tot_PCA[ind_PCA]}' + '}', fontsize=20)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        ax.set_xticks(np.arange(n) + width * 2, K[:n])
        ax.legend(loc='upper left', ncols=2, fontsize=12)
        ax.set_ylim(0, 0.45)
        plt.savefig(f'../Plots/gmm_Cprim_PCA: {tot_PCA[ind_PCA]}')
        # plt.show()
        plt.close()

    elif model == 'MVG':
        C_prim_tot = len(C_prim_list)

        if C_Znorm != []:
            C_Znorm = np.reshape(C_Znorm, (4, int(len(C_Znorm) / 4)))

        tot_PCA = ['No PCA'] + [a for a in range(2, 10)]
        C_prim_list_forPCA = np.reshape(C_prim_list, (9, int(C_prim_tot / 9)))
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
        C_prim_tot = len(C_prim_list)

        if C_Znorm != []:
            C_Znorm = np.reshape(C_Znorm, (4, int(len(C_Znorm) / 4)))

        tot_PCA = ['No PCA'] + [a for a in range(2, 10)]
        C_prim_list_forPCA = np.reshape(C_prim_list, (9, int(C_prim_tot / 9)))
        plt.figure()
        l = l[:8]
        C_plot_data = C_prim_list_forPCA[ind_PCA,:]
        if C_Znorm != [] and ind_PCA == 0 or ind_PCA == 6 or ind_PCA == 7 or ind_PCA == 8:
            C_plot_data = C_plot_data[:8]
            C_Znorm_data = C_Znorm[j,:]
            plt.plot(l, C_Znorm_data, label='\\textbf{Quad log-reg Z-norm}', color='blue', linewidth=2)
        else:
            pass
        plt.plot(l, C_plot_data, label='\\textbf{Quad log-reg}', color='red', linewidth=2)
        plt.legend(fontsize=16)
        plt.title('$C_{prim}$ ' + '\\textbf{for logreg (quadratic) model, PCA: '+ f'{tot_PCA[ind_PCA]}' + '}', fontsize=24, y=1.05)
        plt.xlabel('$\lambda$', fontsize=16)
        plt.xscale('log')
        plt.ylabel('$C_{prim}$', fontsize=16)
        if ind_PCA == 7:
            plt.annotate('\\textbf{'+f'{np.min(C_plot_data):.3f}'+'}',  # this is the text
                     (1e-5,C_plot_data[1]-0.08),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center',
                     color='red', fontsize=12)
        if ind_PCA == 6:
            plt.annotate('\\textbf{' + f'{np.min(C_Znorm_data):.3f}' + '}',  # this is the text
                         (1e-5, C_Znorm_data[1] - 0.08),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center',
                         color='blue', fontsize=12)
        ax = plt.gca()
        ax.set_xticks(l)
        ax.set_xlim(min(l)-100, max(l)+10)
        d = (max(C_plot_data)-min(C_plot_data))/10
        ax.set_ylim(min(C_plot_data)-d,max(C_plot_data)+d)
        ax.set_ylim(0,1)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.grid()
        plt.savefig(f'../Plots/logreg_quad_PCA_Znorm: {tot_PCA[ind_PCA]}', bbox_inches='tight')
        # plt.show()
        plt.close()

    elif model == 'svm poly':
        C_prim_list_d2 = C_prim_list[0, 2::3]
        C_prim_list_d3 = C_prim_list[1, 2::3]
        C_prim_list_forPCA_d2 = np.reshape(C_prim_list_d2, (5, int(25 / 5)))
        C_prim_list_forPCA_d3 = np.reshape(C_prim_list_d3, (5, int(25 / 5)))
        tot_PCA = ['No PCA',9,8,7,6]
        colors = ['fuchsia', 'red', 'orange', 'gold', 'green']
        if ind_PCA == 0:
            plt.figure()
        cs = [1e-4, 1e-3, 1e-2, 1e-1, 1]
        C_plot_data_d2 = C_prim_list_forPCA_d2[ind_PCA,:]
        C_plot_data_d3 = C_prim_list_forPCA_d3[ind_PCA, :]
        plt.plot(cs, C_plot_data_d2, label='\\textbf{SVM poly (2)} ' + f'PCA: {tot_PCA[ind_PCA]}',
                 color=colors[ind_PCA], linewidth=3 if ind_PCA == 1 else 2,
                 zorder=np.inf if ind_PCA == 1 else ind_PCA)
        # plt.plot(cs, C_plot_data_d3, label='\\textbf{SVM poly (3)} ' + f'PCA: {tot_PCA[ind_PCA]}', color='blue', linewidth=2)
        plt.legend(fontsize=16)
        plt.xlabel('\\textbf{c}', fontsize=16)
        plt.xscale('log')
        plt.ylabel('$C_{prim}$', fontsize=16)
        if ind_PCA == 1:
            plt.annotate('\\textbf{'+f'{np.min(C_plot_data_d2):.3f}'+'}',  # this is the text
                     (1.1e-4,C_plot_data_d2[0]-0.02),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 0),  # distance from text to points (x,y)
                     ha='center',
                     color=colors[ind_PCA], fontsize=14)
        ax = plt.gca()
        ax.set_xticks(cs)
        # ax.set_xlim(min(cs)-100, max(cs))
        ax.set_ylim(0.1,0.5)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.grid()
        if ind_PCA == 4:
            # plt.show()
            plt.savefig(f'../Plots/SVM_poly_d2', bbox_inches='tight')
            plt.close()




if __name__ == '__main__':
    D, L = input.load(input.traininput)

    # plot_simple()

    plot_Scatter(D,L)

    # m = 4
    # DP = PCA(D,m)
    # plot_PCA(DP,L,m)

    # DP = LDA(D,L,m=1)
    # plot_LDA(DP,L)

    # explained_variance_plot(D)

    # plot_correlations(D)
    # plot_correlations(D[:,L == 1], 'Blues', subset='same speaker')
    # plot_correlations(D[:,L == 0], 'Reds', subset='different speaker')


    # C = PrettyTab_to_data(gmm_file, fields_gmm, model='gmm')
    # for i in [0]: # range(9) for all PCA
    #     C_prim_plot(C, i, model='gmm')

    # C = PrettyTab_to_data(nb_file, fields_gaussians)
    # C_prim_plot(C, model='MVG')

    # C_lin, C_quad = PrettyTab_to_data(logreg_file, fields_logreg, model='logreg')
    # C_Znorm = PrettyTab_to_data(logreg_Znorm_file, fields_logreg, model='logreg', Znorm=True)
    # lambda_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    # for i, j in zip([0, 6, 7, 8], [0,3,2,1]): #range(9) if no Znorm
    #     C_prim_plot(C_quad, i, model='logreg', l=lambda_list, C_Znorm=C_Znorm, j=j)

    # C_lin, C_poly2, C_poly3, C_rbf = PrettyTab_to_data(svm_file, fields_svm, model='svm')
    # C_polyTOT = np.append(C_poly2, C_poly3)
    # C_polyTOT = C_polyTOT[np.newaxis,:]
    # C_polyTOT = np.reshape(C_polyTOT, (2,int(C_polyTOT.shape[1]/2)))
    # for i in range(5):
    #     C_prim_plot(C_polyTOT, i, model='svm poly')