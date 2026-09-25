# ahzab-minutages

Minutages au verset et au mot des récitations du Coran, pour l'application
**Ahzab** : savoir, dans le fichier audio d'une sourate, où commence et où finit
chaque verset et chaque mot. C'est ce qui permet de lire une sourate d'un trait
sans blanc entre les versets, de faire entendre un mot seul sans le mot voisin,
et de suivre le mot que dit l'imam.

Le dépôt ne contient que du code et des chiffres, jamais d'audio.

## Trois niveaux de confiance

Chaque table dit d'où elle vient (`donnees/<récitation>/recitation.json`, champ
`niveau`) :

1. **publié** — un tiers a minuté ce fichier et le publie : Quranic Universal
   Audio, QUL (Tarteel), mp3quran. On le récolte tel quel.
2. **reporté** — (à venir) une table publiée pour les fichiers de versets,
   reportée dans le fichier de sourate de la même prise par corrélation de la
   forme d'onde.
3. **aligné** — ce que personne ne publie, produit ici par alignement forcé :
   le modèle `obadx/muaalem-model-v3_2` reconnaît les phonèmes, et l'on cherche
   le chemin qui dit tout le texte, dans l'ordre. Aucune borne n'est tirée d'un
   silence ; les silences servent seulement à placer la frontière à l'intérieur
   de l'écart que l'alignement laisse entre deux mots. Les reprises du récitant
   (un passage redit avant de poursuivre) sont suivies : chaque mot garde sa
   dernière occurrence.

## Le format

Un fichier par sourate, `donnees/<récitation>/<sss>.json` :

```json
{
  "audio": "https://…/018.mp3",
  "octets": null,
  "ouverture": [0, 5160],
  "versets": [[5660, 15540], …],
  "mots": [[5660, 6440, 6440, 7370, …], …]
}
```

Temps en millisecondes. `versets[a-1]` = [début, fin] du verset `a` ; `mots[a-1]`
= les [début, fin] de ses mots, à plat, `-1, -1` pour un mot absent. Les tables
alignées ajoutent `lies` (mots qui se prononcent d'un bloc : idghām, iqlāb),
`reprises` et `confiance` (log-probabilité moyenne par verset).

## Les commandes

Python 3.13, dans un environnement virtuel :

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install quran-muaalem "transformers>=4.55,<5" librosa edge-tts
```

(`transformers` 5 a retiré une constante qu'importe `quran-muaalem`.)

Récolter et normaliser ce qui est publié :

```
py -m outil.qua recolter tout && py -m outil.qua normaliser tout
py -m outil.qul lister && py -m outil.qul recolter 159 && py -m outil.qul normaliser 159
py -m outil.mp3quran recolter 137 && py -m outil.mp3quran normaliser 137
```

Contrôler :

```
py -m outil.verifier tout      # bornes croissantes, mots manquants, chevauchements
py -m outil.duree auditer      # la table décrit-elle le fichier servi aujourd'hui ?
py -m outil.comparer           # deux sources sur le même fichier : leur écart
py -m outil.mesurer aligne-humaid qua-ahmed_talib_bin_humaid_mp3quran 1 18 --detail
```

Ajouter une voix que personne ne minute (alignement ; environ 1,7 fois la durée
de l'audio sur un processeur portable, bien moins sur un GPU) :

```
.venv\Scripts\python -m outil.aligner sourate aligne-badr 018.mp3 18 --url https://server10.mp3quran.net/bader/Rewayat-Hafs-A-n-Assem/018.mp3
```

`--jusqua 150 --versets 1-10` aligne seulement le début d'un fichier.

## Ce qui est versionné

Le code, les tables **alignées** ici (`donnees/aligne-*`), les fiches de toutes
les récitations et les bilans (`audit-durees.json`, `comparaisons.json`). Les
tables publiées par des tiers se régénèrent avec les commandes ci-dessus et ne
sont pas recopiées ici.

## Sources et attributions

- **Quranic Universal Audio**, Wider Community — minutages au verset et au mot,
  licence CC BY 4.0. <https://github.com/Wider-Community/quranic-universal-audio>
- **QUL — Quranic Universal Library**, Tarteel — segments au mot et au verset ;
  usage commercial permis selon sa FAQ, avec attribution. <https://qul.tarteel.ai>
- **mp3quran.net** — fichiers audio et bornes au verset (`ayat_timing`).
- **quran-muaalem** et **quran-transcript** (licence MIT), modèle
  `obadx/muaalem-model-v3_2` — reconnaissance des phonèmes et transcription
  phonétique de l'uthmani.
