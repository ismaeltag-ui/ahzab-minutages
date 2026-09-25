"""Récolte des minutages publiés par QUL (Quranic Universal Library, Tarteel).

QUL sert, sans compte, une interface qu'utilise son propre lecteur d'aperçu :

    /api/v1/audio/surah_segments/<id>?surah=S&from=1&to=N&per_page=300
    → {"audio": {"url", "duration", "audio_size"},
       "segments": {"S:A": {"time_from", "time_to", "segments": [[mot, début, fin], …]}}}

Seules les récitations « par sourate » sont récoltées : celles « par verset »
portent des bornes de remplissage (premier mot toujours à 70 ms, beaucoup de mots
d'exactement 100 ms), relevé du 2026-09-24.

    py -m outil.qul lister                 # les récitations avec segments
    py -m outil.qul recolter 159 13 14     # télécharge dans cache/qul/<id>/
    py -m outil.qul recolter tout
    py -m outil.qul normaliser 159         # écrit donnees/qul-159/
"""

from __future__ import annotations

import html
import re
import sys
import time

from .commun import CACHE, comptes, ecrire_json, http_brut, http_json, lire_json
from .format import ecrire_fiche, ecrire_sourate, table_sourate

BASE = "https://qul.tarteel.ai"
LISTE = CACHE / "qul" / "liste.json"


def lister(rafraichir: bool = False) -> list[dict]:
    """Les récitations QUL « par sourate » avec segments : id de fiche, id de
    récitation (celui de l'API), libellé de recherche."""
    if LISTE.exists() and not rafraichir:
        return lire_json(LISTE)
    page = http_brut(f"{BASE}/resources/recitation")[0].decode("utf-8")
    lignes = re.findall(r'id="downloadable_resource_(\d+)"[^>]*?data-search="([^"]*)"', page, re.S)
    res = []
    for fiche, recherche in lignes:
        recherche = html.unescape(recherche)
        if "with segments" not in recherche or "surah by surah" not in recherche:
            continue
        p = http_brut(f"{BASE}/resources/recitation/{fiche}")[0].decode("utf-8")
        m = re.search(r'data-controller="surah-segment-player"[^>]*?data-recitation="(\d+)"', p, re.S)
        if not m:
            continue
        nom = re.sub(r"\s+(surah by surah|ayah by ayah).*$", "", recherche).strip()
        res.append({"fiche": int(fiche), "id": int(m.group(1)), "nom": nom, "etiquettes": recherche})
        time.sleep(0.2)
    ecrire_json(LISTE, res, compact=False)
    return res


def recolter(ident: int, fils: int = 4) -> None:
    """Quatre requêtes à la fois au plus : l'interface répond en 1 à 3 s par
    sourate, et une récitation entière prendrait sinon plus de cinq minutes."""
    from concurrent.futures import ThreadPoolExecutor

    dossier = CACHE / "qul" / str(ident)
    hafs = comptes("hafs")

    def une(s: int) -> None:
        cible = dossier / f"{s:03d}.json"
        if cible.exists():
            return
        n = len(hafs[s])
        d = http_json(f"{BASE}/api/v1/audio/surah_segments/{ident}?surah={s}&from=1&to={n}&per_page=300")
        ecrire_json(cible, d)
        time.sleep(0.1)

    with ThreadPoolExecutor(max_workers=fils) as ex:
        list(ex.map(une, range(1, 115)))
    print(f"qul {ident} : 114 sourates en cache", flush=True)


def normaliser(ident: int) -> dict:
    liste = {r["id"]: r for r in lister()}
    info = liste.get(ident, {"nom": f"QUL {ident}", "fiche": None, "etiquettes": ""})
    hafs = comptes("hafs")
    nom_rec = f"qul-{ident}"
    couverture_v = couverture_m = 0
    for s in range(1, 115):
        brut = lire_json(CACHE / "qul" / str(ident) / f"{s:03d}.json")
        versets, mots = {}, {}
        for cle, v in (brut.get("segments") or {}).items():
            _, a = (int(x) for x in cle.split(":"))
            if v.get("time_from") is not None and v.get("time_to") is not None:
                versets[a] = (v["time_from"], v["time_to"])
            mv = {}
            for seg in v.get("segments") or []:
                if len(seg) >= 3 and seg[1] is not None and seg[2] is not None:
                    mv[int(seg[0])] = (seg[1], seg[2])
            if mv:
                mots[a] = mv
        audio = (brut.get("audio") or {}).get("url") or ""
        audio = re.sub(r"(?<!:)//+", "/", audio)  # « year1440//112.mp3 » chez quranicaudio
        taille = (brut.get("audio") or {}).get("audio_size")
        t = table_sourate(audio, int(taille) if taille else None, hafs[s], versets, mots)
        couverture_v += sum(1 for x in t["versets"] if x)
        couverture_m += sum(1 for x in t["mots"] if x)
        ecrire_sourate(nom_rec, s, t)
    fiche = {
        "recitation": nom_rec,
        "nom": info["nom"],
        "riwaya": "hafs",
        "comptage": "kufi",
        "source": {
            "nom": "QUL — Quranic Universal Library (Tarteel)",
            "url": f"{BASE}/resources/recitation/{info['fiche']}" if info["fiche"] else BASE,
            "api": f"{BASE}/api/v1/audio/surah_segments/{ident}",
            "conditions": "Usage commercial permis selon la FAQ de QUL ; attribution à QUL/Tarteel.",
        },
        "niveau": "publie",
        "production": "apprentissage automatique (documentation QUL)",
        "couverture": {"versets": couverture_v, "versets_avec_mots": couverture_m},
    }
    ecrire_fiche(nom_rec, fiche)
    print(f"{nom_rec} : {couverture_v} versets, {couverture_m} avec mots", flush=True)
    return fiche


def main(args: list[str]) -> None:
    if not args or args[0] == "lister":
        for r in lister(rafraichir="--rafraichir" in args):
            print(f"{r['id']:>4}  {r['nom']}")
        return
    cmd, ids = args[0], args[1:]
    if ids == ["tout"]:
        ids = [str(r["id"]) for r in lister()]
    for i in ids:
        if cmd == "recolter":
            recolter(int(i))
        elif cmd == "normaliser":
            normaliser(int(i))


if __name__ == "__main__":
    main(sys.argv[1:])
