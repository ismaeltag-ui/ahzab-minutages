"""Le format canonique d'une table de minutage, commun à toutes les sources.

Un fichier par sourate et par récitation, `donnees/<recitation>/<sss>.json` :

    {
      "audio": "https://…/002.mp3",      # le fichier que les bornes décrivent
      "octets": 96913557,                # sa taille : l'empreinte (null si inconnue)
      "ouverture": [0, 7490],            # ce qui précède le verset 1 (istiʿādha, basmala)
      "versets": [[7490, 13620], …],     # [début, fin] en ms, index = verset − 1 ; null = absent
      "mots": [[7690, 8350, 8390, 9190, …], …]   # par verset, [début, fin] de chaque mot à plat ;
                                         # -1, -1 pour un mot absent ; null = pas de mots
    }

Les métadonnées de la récitation (nom, riwaya, source, licence, niveau de
confiance, bilan de vérification) sont dans `donnees/<recitation>/recitation.json`.
"""

from __future__ import annotations

from .commun import DONNEES, ecrire_json


def table_sourate(
    audio: str,
    octets: int | None,
    nb_mots: list[int],
    versets: dict[int, tuple[int, int]],
    mots: dict[int, dict[int, tuple[int, int]]],
) -> dict:
    """`nb_mots` : mots attendus par verset (sa longueur donne le nombre de versets)."""
    tv, tm = [], []
    for a, n in enumerate(nb_mots, start=1):
        v = versets.get(a)
        tv.append([int(v[0]), int(v[1])] if v else None)
        mv = mots.get(a)
        if not mv:
            tm.append(None)
            continue
        plat: list[int] = []
        for w in range(1, n + 1):
            s = mv.get(w)
            plat += [int(s[0]), int(s[1])] if s else [-1, -1]
        tm.append(plat)
    premier = tv[0] if tv else None
    return {
        "audio": audio,
        "octets": octets,
        "ouverture": [0, premier[0]] if premier and premier[0] > 0 else None,
        "versets": tv,
        "mots": tm,
    }


def ecrire_sourate(recitation: str, sourate: int, table: dict) -> None:
    ecrire_json(DONNEES / recitation / f"{sourate:03d}.json", table)


def ecrire_fiche(recitation: str, fiche: dict) -> None:
    ecrire_json(DONNEES / recitation / "recitation.json", fiche, compact=False)
