"""Fiche d'intégration d'une table : ce que l'application doit savoir d'une voix.

Pour chaque table : le répertoire des MP3 et s'ils suivent tous le nommage
`001.mp3` ; les sourates où manque au moins un verset (l'application ne propose
pas la voix pour celles-là) ; la durée du juz' 30 mesurée sur la table, qui range
la voix au menu ; la lecture par morceaux (`Range`) chez l'hébergeur.

    py -m outil.inventaire qua-ahmed_amer_tvquran aligne-badr …
"""

from __future__ import annotations

import sys
import urllib.request

from .commun import AGENT, DONNEES, lire_json


def _range(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Range": "bytes=1000-1999"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return str(r.status)
    except Exception as e:  # noqa: BLE001
        return f"erreur {e}"


def fiche(rec: str) -> dict:
    d = DONNEES / rec
    f = lire_json(d / "recitation.json")
    bases, irreguliers, absentes, manques = set(), [], [], {}
    juz30 = 0
    for s in range(1, 115):
        p = d / f"{s:03d}.json"
        if not p.exists():
            absentes.append(s)
            continue
        t = lire_json(p)
        a = t["audio"]
        base, nom = a.rsplit("/", 1)
        bases.add(base + "/")
        if nom != f"{s:03d}.mp3":
            irreguliers.append(s)
        trous = [i + 1 for i, v in enumerate(t["versets"]) if v is None]
        if trous:
            manques[s] = trous
        if s >= 78:
            v = [x for x in t["versets"] if x]
            if v:
                juz30 += v[-1][1] - v[0][0]
    exemple = next((lire_json(d / f"{s:03d}.json")["audio"] for s in (112, 1) if (d / f"{s:03d}.json").exists()), None)
    return {
        "rec": rec, "nom": f.get("nom"), "riwaya": f.get("riwaya"), "comptage": f.get("comptage"),
        "style": f.get("style"), "bases": sorted(bases), "irreguliers": irreguliers,
        "sourates_absentes": absentes, "versets_absents": manques,
        "juz30_min": round(juz30 / 60000, 1), "range": _range(exemple) if exemple else None,
    }


def main(args: list[str]) -> None:
    for rec in args:
        x = fiche(rec)
        indispo = sorted(set(x["sourates_absentes"]) | set(x["versets_absents"]))
        print(f"{x['rec']} | {x['nom']} | {x['riwaya']} {x['comptage']} {x['style']} | juz30 {x['juz30_min']} min"
              f" | range {x['range']}", flush=True)
        print(f"    bases : {x['bases']}", flush=True)
        if x["irreguliers"]:
            print(f"    nommage irrégulier : {len(x['irreguliers'])} sourates", flush=True)
        if indispo:
            print(f"    sourates indisponibles : {indispo}  (versets : {x['versets_absents']})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
