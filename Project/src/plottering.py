import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Definisci il percorso del file
file_path = "../Data/Train.txt"

def plot():
    # Inizializza una lista vuota per i dati
    data = []

    # Leggi il file e aggiungi i dati alla lista
    with open(file_path, "r") as file:
        for line in file:
            # Rimuovi eventuali spazi bianchi e suddividi la linea in elementi separati
            line = line.strip()
            elements = line.split(",")

            # Converti gli elementi in float e aggiungili alla lista dei dati
            data.append([float(element) for element in elements])

    # Stampa i dati importati
    #print(data)
    # Esegui il plotting dei grafici con i parametri sulle x e i parametri elevati alla terza potenza sulle y
    num_params = len(data[0]) - 1
    labels = ["Parametro {}".format(i) for i in range(num_params)]

    # Calcola il numero di righe e colonne per la griglia
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

        # Divide i punti in due gruppi sulla base delle classi
        # Divide i punti in due gruppi sulla base delle classi
        x_class0 = [x for x, c in zip(x_values, classes) if c == 0]
        x_class1 = [x for x, c in zip(x_values, classes) if c == 1]

        ax.hist(x_class0, bins=20, color='blue', alpha=0.5, density=True, label='Classe 0')
        ax.hist(x_class1, bins=20, color='red', alpha=0.5, density=True, label='Classe 1')
        ax.set_xlabel(labels[i])
        ax.set_ylabel("{}".format(labels[i]))
        ax.legend()  
    
    # Regola lo spaziamento tra i grafici
    plt.tight_layout()

    # Salva l'immagine come file PNG
    plt.savefig('istogrammi.png')

    # Mostra l'immagine
    plt.show()

if __name__ == '__main__':
    plot()
    