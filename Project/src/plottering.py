import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

file_path = "../Data/Train.txt"

def plot():
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

if __name__ == '__main__':
    plot()
    