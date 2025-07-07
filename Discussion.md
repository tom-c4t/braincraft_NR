# Unsere Fragen
- In welches file soll unser Trainingscode?
- Wie bekommt die train-Funktion in challenge.py den Code übergeben?
- Müssen wir das Model selbst noch definieren?
   - Übergabe der Model-Parameter als Liste (Orientierung an player_random.py)

# Ansatz
- Nutze Änderung des Energy-Levels als Reward
- Worauf haben wir Einfluss?
    - activation function
    - Weight Matrizen
    - dünnbesetzte Matrix W (recurrent weights) für geringe Berechnungszeit
- Tracke während des Trainings Anzahl der Schritte ohne Hit
 
1. Model wird mit Startgewichten initialisiert
2. Bot macht 5? Schritte
3. Wie hat sich Energy Level verändert?
   a) bei = -5: Fortbewegung ohne Kollision , bei < -5: Wall hit, > -5: Source überschritten
4. Was leiten wir daraus für die Anpassung der Matrizen ab?

# To Dos
- bis Mittwoch (**09.07.2025**) Echo State Networks recherchieren
