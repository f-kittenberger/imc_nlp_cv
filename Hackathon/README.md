# 🚀 Multimodal Image Search using YOLO + CLIP

## 🇩🇪 Deutsch

---

## 1. Overview

In diesem Projekt entwickeln wir ein **multimodales System**, das **Computer Vision (CV)** und **Natural Language Processing (NLP)** kombiniert.  

Ziel ist es, Bilder anhand von natürlicher Sprache zu durchsuchen und zu verstehen.

Unser Ansatz basiert auf:

- YOLO zur Objekterkennung  
- CLIP zur semantischen Verknüpfung von Bild und Text  

---

## 2. Problem Statement

Wir wollen folgendes Problem lösen:

> 🔍 Bilder anhand komplexer natürlicher Sprache finden

Beispiele:

- "three cars in the countryside"  
- "a small blue truck on grass"  

Herausforderungen:

- Bildmodelle verstehen keine Sprache  
- NLP versteht keine Bilder  
- Objektklassen sind oft unscharf (truck vs car)  

---

## 3. Technische Pipeline

#### 3.1 Computer Vision (YOLO)

YOLO wird verwendet für:

- Objekterkennung  
- Klassifikation (car, truck, bus, motorcycle)  
- Zählung der Objekte  

Beispiel:

```text
3 cars, 1 truck, 0 buses
```

#### 3.2 NLP (CLIP)

CLIP wird verwendet für:

Text → Embedding
Bild → Embedding
Vergleich mittels Cosine Similarity

👉 CLIP verbindet Sprache und Bild direkt.


### 3.3 Integration

Pipeline:

User Query
   ↓
Klassen-Erkennung (car, bus, ...)
   ↓
YOLO Filter (relevante Bilder)
   ↓
CLIP Ranking (semantische Bewertung)
   ↓
Top Ergebnisse

Erweiterung durch YOLO-Kontext:

"three cars in the countryside"
→ "three cars in the countryside. detected objects: 3 cars, 1 truck"

## 4. Designentscheidungen
Kein harter YOLO-Filter
CLIP übernimmt das Ranking
YOLO liefert Zusatzinformationen
Robust gegen Klassifikationsfehler
## 5. Datensatz
COCO-basierter Subset
Fahrzeugbilder
Beschreibungen pro Bild

#### Struktur:

vehicle_subset/
vehicle_subset_descriptions.json
outputs/crop_metadata.json
## 6. Golden Set

Testanfragen:

three cars in the countryside
a red car in traffic
two buses in a city
a truck on grass
cars and buses together
## 7. Hackathon Ziel (MVP)
Eingabe: Text
Ausgabe: Top 3–4 Bilder
funktionierende Suche
## 8. Stretch Goals
bessere Zahlen- und Farbverarbeitung
genauere Objektlogik
Web Interface
CLIP Fine-Tuning
Crop-basierte Suche
## 9. Fazit

Das System kombiniert:

CV → erkennt Objekte
NLP → versteht Sprache
CLIP → verbindet beides

👉 Ergebnis: robuste multimodale Bildsuche