"""Récolte des tables `ayat_timing` de mp3quran : bornes au verset, posées par l'éditeur.

    GET https://www.mp3quran.net/api/v3/ayat_timing?surah=S&read=ID
    → [{"ayah": 0, "start_time", "end_time"}, {"ayah": 1, …}, …]   (ms ; ayah 0 = basmala)

C'est la source que l'application lit déjà (`lib/verseTiming.ts`). Ici, elle sert
surtout de **référence** : là où une autre source a minuté le même fichier, on
mesure l'écart entre les deux.

    py -m outil.mp3quran lister
    py -m outil.mp3quran recolter 137 134
    py -m outil.mp3quran normaliser 137 134
"""

from __future__ import annotations

import sys
import time

from .commun import CACHE, comptes, ecrire_json, http_json, lire_json
from .format import ecrire_fiche, ecrire_sourate, table_sourate

API = "https://www.mp3quran.net/api/v3"
DOSSIER = CACHE / "mp3quran"

RIWAYA = {"حفص عن عاصم": "hafs", "ورش عن نافع": "warsh", "قالون عن نافع": "qalun",
          "شعبة  عن عاصم": "shuba", "شعبة عن عاصم": "shuba", "المصحف المجود": "hafs"}
COMPTAGE = {"hafs": "kufi", "shuba": "kufi", "warsh": "madani", "qalun": "madani"}


def lectures() -> list[dict]:
    f = DOSSIER / "reads.json"
    if not f.exists():
        ecrire_json(f, http_json(f"{API}/ayat_timing/reads"))
    return lire_json(f)


def recolter(read: int, fils: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def une(s: int) -> None:
        f = DOSSIER / str(read) / f"{s:03d}.json"
        if f.exists():
            return
        try:
            d = http_json(f"{API}/ayat_timing?surah={s}&read={read}")
        except Exception:
            d = None  # sourate absente chez cet éditeur
        ecrire_json(f, d if isinstance(d, list) else [])
        time.sleep(0.1)

    with ThreadPoolExecutor(max_workers=fils) as ex:
        list(ex.map(une, range(1, 115)))
    print(f"mp3quran {read} : en cache", flush=True)


def normaliser(read: int) -> dict:
    info = {r["id"]: r for r in lectures()}[read]
    riwaya = RIWAYA.get(info["rewaya"].strip(), "hafs")
    if info["rewaya"].strip() not in RIWAYA:
        print(f"  riwaya inconnue « {info['rewaya']} », comptée comme Hafs", flush=True)
    attendus = comptes(riwaya)
    base = info["folder_url"].rstrip("/") + "/"
    nom_rec = f"mp3quran-{read}"
    couverture = 0
    for s in range(1, 115):
        lignes = lire_json(DOSSIER / str(read) / f"{s:03d}.json")
        if not lignes:
            continue
        versets = {l["ayah"]: (l["start_time"], l["end_time"]) for l in lignes
                   if l.get("ayah", 0) > 0 and l.get("start_time") is not None}
        t = table_sourate(f"{base}{s:03d}.mp3", None, attendus[s], versets, {})
        zero = next((l for l in lignes if l.get("ayah") == 0), None)
        if zero:
            t["ouverture"] = [zero["start_time"], zero["end_time"]]
        couverture += sum(1 for x in t["versets"] if x)
        ecrire_sourate(nom_rec, s, t)
    fiche = {
        "recitation": nom_rec,
        "nom": info["name"],
        "riwaya": riwaya,
        "comptage": COMPTAGE[riwaya],
        "source": {"nom": "mp3quran.net — ayat_timing", "url": f"{API}/ayat_timing?read={read}",
                   "conditions": "Source déjà utilisée par l'application."},
        "niveau": "publie",
        "production": "bornes posées par l'éditeur",
        "couverture": {"versets": couverture, "versets_avec_mots": 0},
    }
    ecrire_fiche(nom_rec, fiche)
    print(f"{nom_rec} : {couverture} versets", flush=True)
    return fiche


def main(args: list[str]) -> None:
    if not args or args[0] == "lister":
        for r in lectures():
            print(f"{r['id']:>4}  {r['rewaya']:<22} {r['name']}  {r['folder_url']}")
        return
    for i in args[1:]:
        (recolter if args[0] == "recolter" else normaliser)(int(i))


if __name__ == "__main__":
    main(sys.argv[1:])
