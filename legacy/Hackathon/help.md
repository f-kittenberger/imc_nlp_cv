
## 10. Schritt-für-Schritt Ablauf der Pipeline

Im Folgenden wird beschrieben, **welche Dateien ausgeführt werden**, **welche Zwischendateien entstehen** und **wie das finale Ergebnis erzeugt wird**.

### Schritt 1: Ausgangsdaten vorbereiten

Vorhandene Eingabedateien:

```text
vehicle_subset/
vehicle_subset_descriptions.json
vehicle_captions.json

Bedeutung:

vehicle_subset/
enthält die Originalbilder
vehicle_subset_descriptions.json
enthält pro Bild:
Dateiname
Pfad
Bild-ID
mehrere Textbeschreibungen
vehicle_captions.json
enthält zusätzliche Caption-Zuordnungen
Schritt 2: YOLO-Crops erzeugen

Ausgeführte Datei:

scripts/generate_yolo_crops.py

Aufgabe:

lädt jedes Bild aus vehicle_subset/
führt YOLO-Objekterkennung aus
erkennt relevante Klassen:
car
truck
bus
motorcycle
schneidet für jede erkannte Bounding Box einen Crop aus
speichert die Crops als einzelne Bilddateien

Neu erzeugte Dateien/Ordner:

outputs/crops/
outputs/crop_metadata.json

Beispiele:

outputs/crops/000000034820_car_0.jpg
outputs/crops/000000034820_bus_1.jpg

crop_metadata.json enthält pro Crop:

Crop-Datei
Originalbild
erkannte Klasse
Confidence
Bounding Box
Pfad
Schritt 3: YOLO-Metadaten zu Bildstruktur zusammenfassen

Diese Logik passiert innerhalb der Suchpipeline, insbesondere in:

scripts/clip_search_extended.py

Aufgabe:

outputs/crop_metadata.json wird geladen
alle Crops werden wieder dem jeweiligen Originalbild zugeordnet
für jedes Bild werden Objektzahlen gezählt, z. B.:
Bild A:
- 3 cars
- 1 truck
- 0 buses
- 0 motorcycles

Interne Datenstruktur:

image_index

Diese wird nicht zwingend als separate Datei gespeichert, sondern beim Start erzeugt.

Schritt 4: User Query eingeben

Ausgeführte Datei:

scripts/clip_search_extended.py

Beispiel-Query:

three cars in the countryside

Aufgabe:

Query wird eingelesen
relevante Objektklassen werden extrahiert, z. B.:
car
bus
truck

Beispiel:

Query: "cars and buses together"
→ erkannte Klassen: car, bus
Schritt 5: Kandidatenbilder mit YOLO vorfiltern

Ebenfalls in:

scripts/clip_search_extended.py

Aufgabe:

nur Bilder werden weiter berücksichtigt, die die gesuchten Klassen enthalten

Beispiele:

Query enthält car
→ alle Bilder mit mindestens einem car
Query enthält car und bus
→ alle Bilder mit mindestens einem car und mindestens einem bus

YOLO dient hier also als grober Klassenfilter, nicht als finale Entscheidung.

Schritt 6: Erweiterte Query für CLIP erzeugen

Für jedes Kandidatenbild wird eine zusätzliche Textinformation aus YOLO erzeugt, z. B.:

detected objects: 7 cars, 4 trucks, 0 buses, 0 motorcycles

Diese Information wird mit der ursprünglichen User Query kombiniert.

Beispiel:

Original Query:
three cars in the countryside

Erweiterte Query:
three cars in the countryside. detected objects: 3 cars, 1 truck, 0 buses, 0 motorcycles
Schritt 7: CLIP-Ranking

Ausgeführt in:

scripts/clip_search_extended.py

Verwendete Funktionen aus:

src/model/clip_model.py

Aufgabe:

CLIP berechnet ein Embedding für die Query
CLIP berechnet ein Embedding für jedes Kandidatenbild
Ähnlichkeit zwischen Text und Bild wird berechnet
Bilder werden nach Relevanz sortiert

Verwendete Komponenten:

encode_text(...)
encode_image(...)
Cosine Similarity

Ergebnis pro Bild:

query_score
augmented_score
final_score
Schritt 8: Ergebnisse ausgeben

Ausgabe in der Konsole durch:

scripts/clip_search_extended.py

Pro Ergebnisbild werden angezeigt:

Score
Bildpfad
erkannte Objektzahlen
YOLO-Zusatzinfo
zugehörige Crops als Untergruppe

Beispiel:

Final score: 0.8123
Image: C:\...\vehicle_subset\000000034820.jpg
Counts: {'car': 2, 'bus': 1}
YOLO info: detected objects: 2 cars, 0 trucks, 1 bus, 0 motorcycles

Crops:
    car:
        - C:\...\outputs\crops\000000034820_car_0.jpg
        - C:\...\outputs\crops\000000034820_car_1.jpg
    bus:
        - C:\...\outputs\crops\000000034820_bus_0.jpg
11. Wichtige Dateien im Projekt
Eingabedateien
vehicle_subset/
vehicle_subset_descriptions.json
vehicle_captions.json
Verarbeitungsdateien
scripts/generate_yolo_crops.py
scripts/clip_search_extended.py
src/model/clip_model.py
Erzeugte Dateien
outputs/crops/
outputs/crop_metadata.json
Finale Ausgabe
Top-Ergebnisbilder
zugehörige YOLO-Crops
Scores
strukturierte Objektdaten
12. Zusammenfassung des gesamten Ablaufs

Die Pipeline läuft also in dieser Reihenfolge:

1. Originalbilder + JSON laden
2. YOLO auf Bilder anwenden
3. Crops erzeugen
4. crop_metadata.json erzeugen
5. User Query eingeben
6. relevante Klassen aus Query extrahieren
7. Kandidatenbilder mit YOLO vorfiltern
8. YOLO-Zusatzinfo pro Bild erzeugen
9. CLIP auf Kandidatenbilder anwenden
10. Bilder nach Score sortieren
11. Top-Ergebnisse mit Crops ausgeben

Kurzform:

Daten → YOLO → Crops + Metadaten → Klassenfilter → CLIP → Ranking → Ergebnis