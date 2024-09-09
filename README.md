# FIA_OCR

MAT. 0512110004 RUSSOMANDO ANTONIO

Progetto FIA: 
OCR con ensemble learning con modello di classificazione SVM (Support Vector Machine)

Obiettivi del progetto:
Riconoscere e digitalizzare caratteri alfanumerici da testi scritti a mano o stampati.

I file importanti contenenti il codice principale sono:
- svm_onevsrest.ipynb, contiene il tuning degli iperparametri e l'addestramento finale del modello one-vs-all.
- svm_onevsrest_without_hyperparameters.ipynb, contiene il codice dell'addestramento del modello one-vs-rest senza il tuning degli iperparametri.
- extract_dataset.ipynb, contiene il processo di estrazione del dataset da formato .ubyte a formato .ong con la suddivisione delle classi in sottocartelle.
- webapp.py, è la web-app funzionante e la si può avviare tramite il comando da terminale "python3 webapp.py".
- templates, non è altro che la cartella contenente il file html che descrive la webapp.
