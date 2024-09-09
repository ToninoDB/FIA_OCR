import numpy as np
import joblib
from flask import Flask, request, render_template
from PIL import Image, ImageOps
import io

model = joblib.load('svm_model.pkl')

char_dictionary = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "d",
    39: "e",
    40: "f",
    41: "g",
    42: "h",
    43: "n",
    44: "q",
    45: "r",
    46: "t"
}


def preprocessa_immagine(file_path):
    # Carica l'immagine
    img = Image.open(file_path).convert('L')
    
    # Inverti i colori (sfondo nero e carattere bianco)
    img = ImageOps.invert(img)
    
    # Ridimensiona l'immagine a 28x28 pixel
    img = img.resize((28, 28), Image.ANTIALIAS)
    
    # Converti l'immagine in un array numpy
    img_array = np.array(img)

    # Normalizza l'immagine
    media = np.mean(img_array)
    img_array[img_array <= media] = 0
    img_array = img_array / 255.0

    # Flatten dell'array (se il modello si aspetta un input 1D)
    img_array = img_array.flatten()

    # Aggiungi un asse per rappresentare un singolo esempio (poiché il modello si aspetta più esempi)
    img_array = img_array.reshape(1, -1)

    return img_array

# Inizializza l'app Flask
app = Flask(__name__)

# Pagina principale
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint per fare predizioni
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'Nessun file inviato', 400

    file = request.files['file']

    if file.filename == '':
        return 'Nessun file selezionato', 400

    try:
        # Preprocessa l'immagine
        img_array = preprocessa_immagine(file)

        # Fai la predizione
        prediction = model.predict(img_array)

        # Ottieni il carattere corrispondente dalla predizione
        predicted_char = char_dictionary[prediction[0]]

        # Restituisci la predizione come testo
        return f'Predizione del carattere: {predicted_char}'

    except Exception as e:
        return str(e), 500

# Avvia l'app Flask
if __name__ == '__main__':
    app.run(debug=True)