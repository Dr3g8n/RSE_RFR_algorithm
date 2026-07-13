# Studienarbeit — Code-Starter-Kit

Dieses Verzeichnis enthält den Code, mit dem du arbeiten wirst.
Der Hauptalgorithmus, den du erweitern sollst, liegt in
`src/generate_fiber_distribution.py`.

## Aufbau

```
studienarbeit_starter/
├── src/
│   ├── generate_fiber_distribution.py   # ← HIER arbeitest du hauptsächlich
│   ├── seeding.py                       # Zufallsgenerator-Helfer
│   ├── config_loader.py                 # Lädt Werte aus config/*.json
│   ├── plot_fiber_distribution.py       # Visualisiert die erzeugten RVEs
│   ├── plot_utils.py                    # Plot-Helferfunktionen
│   ├── generate_vtu_mesh.py             # Mesh-Erstellung (für adaptives Meshing)
│   └── plot_vtu_mesh.py                 # Mesh-Visualisierung
├── config/
│   ├── algorithm.json     # Seed, Algorithmus-Parameter
│   ├── paths.json         # Pfade für In-/Output
│   ├── rve.json           # RVE-Größe, Faserradius, Mesh-Einstellungen
│   └── target_vfs.json    # Zielwerte für Faservolumenanteil
└── data/                  # Output landet hier
    ├── csv/               # Faser-Koordinaten als CSV
    ├── msh/, vtu/         # Meshes (später)
    └── images/            # Plots
```

## Erster Lauf

```bash
conda activate studienarbeit         # falls noch nicht aktiv
cd ~/studienarbeit_starter
python src/generate_fiber_distribution.py
```

Wenn alles funktioniert, landen CSV-Dateien mit Faser-Koordinaten in
`data/csv/`. Mit `python src/plot_fiber_distribution.py` kannst du sie
visualisieren.

## Wo du erweiterst

Der Algorithmus nutzt aktuell einen **konstanten Faserradius**
(`rve.fiber.radius` in `config/rve.json`). Dein Ziel: ihn so erweitern,
dass die Radien einer **Verteilung** folgen (z.B. log-normal, Weibull —
zu begründen aus der Literatur).

Suchbegriffe im Code: `fiber_radius` — überall, wo dieser als skalarer
Wert benutzt wird, musst du eine Lösung finden.
