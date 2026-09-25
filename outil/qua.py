"""Récolte des minutages de Quranic Universal Audio (Wider Community), CC BY 4.0.

Chaque récitation est une archive d'une version publiée sur GitHub :
`verse_timestamps.json.gz`, `word_timestamps.json.gz` (et `letter_…` pour Hafs).
Le fichier des mots porte :

    {"_meta": {…, "riwayah", "verse_count", "occurrence_count"},
     "rows": [[réf "S:A", début, fin, canonique, silence_après, [[mot, début, fin], …]], …]}

Un verset que le récitant reprend apparaît plusieurs fois ; seule l'occurrence
marquée « canonique » donne les bornes, les reprises sont comptées à part.
Les rangs de mots sont ceux de la riwaya (`<riwaya>_words.json.gz`).

    py -m outil.qua lister
    py -m outil.qua recolter tout          # hors YouTube et Google Drive
    py -m outil.qua normaliser tout
"""

from __future__ import annotations

import sys
import zipfile

from .commun import CACHE, comptes, ecrire_json, http_brut, lire_json
from .format import ecrire_fiche, ecrire_sourate, table_sourate

VERSION = "v3.1.0"
BASE = f"https://github.com/Wider-Community/quranic-universal-audio/releases/download/{VERSION}"
DOSSIER = CACHE / "qua"

RIWAYA = {"hafs_an_asim": "hafs", "warsh_an_nafi": "warsh", "qalon_an_nafi": "qalun", "shubah_an_asim": "shuba"}
COMPTAGE = {"hafs": "kufi", "shuba": "kufi", "warsh": "madani", "qalun": "madani"}
# Sources dont l'audio ne se lit pas tel quel dans l'application (réencodage nécessaire).
NON_DIFFUSABLES = ("youtube", "drive")


def _fichier(nom: str) -> None:
    cible = DOSSIER / nom
    if not cible.exists():
        DOSSIER.mkdir(parents=True, exist_ok=True)
        cible.write_bytes(http_brut(f"{BASE}/{nom}")[0])


def catalogue() -> list[dict]:
    for nom in ("catalog.json", "manifest.json", "surah_info.json", "LICENSE",
                "warsh_words.json.gz", "qalun_words.json.gz", "shuba_words.json.gz"):
        _fichier(nom)
    return lire_json(DOSSIER / "catalog.json")["recitations"]


def diffusable(r: dict) -> bool:
    return r.get("channel") not in NON_DIFFUSABLES


def recolter(slug: str) -> None:
    _fichier(f"{slug}.zip")
    cible = DOSSIER / slug
    if not (cible / "word_timestamps.json.gz").exists():
        with zipfile.ZipFile(DOSSIER / f"{slug}.zip") as z:
            for n in z.namelist():
                if not n.startswith("letter_"):  # les lettres ne servent pas ici
                    z.extract(n, cible)
    print(f"qua {slug} : en cache", flush=True)


def normaliser(slug: str) -> dict:
    cat = {r["slug"]: r for r in catalogue()}[slug]
    riwaya = RIWAYA[cat["riwayah"]]
    attendus = comptes(riwaya)
    donnees = lire_json(DOSSIER / slug / "word_timestamps.json.gz")
    urls = cat["audio"].get("chapter_urls") or {}
    decalages = cat["audio"].get("chapter_offsets_ms") or {}
    par_sourate: dict[int, tuple[dict, dict]] = {}
    reprises: dict[str, int] = {}
    for ref, debut, fin, canonique, _silence, mots in donnees["rows"]:
        s, a = (int(x) for x in ref.split(":"))
        if not canonique:
            reprises[ref] = reprises.get(ref, 0) + 1
            continue
        versets, mv = par_sourate.setdefault(s, ({}, {}))
        versets[a] = (debut, fin)
        mv[a] = {int(w): (d, f) for w, d, f in mots}
    nom_rec = f"qua-{slug}"
    couverture_v = couverture_m = 0
    for s in range(1, 115):
        if s not in par_sourate or str(s) not in urls:
            continue
        versets, mv = par_sourate[s]
        t = table_sourate(urls[str(s)], None, attendus[s], versets, mv)
        if str(s) in decalages:
            # Décalage signalé par la source entre son fichier de référence et l'URL
            # publiée : on le garde, l'intégration devra le vérifier avant usage.
            t["decalage_source_ms"] = decalages[str(s)]
        couverture_v += sum(1 for x in t["versets"] if x)
        couverture_m += sum(1 for x in t["mots"] if x)
        ecrire_sourate(nom_rec, s, t)
    fiche = {
        "recitation": nom_rec,
        "nom": f"{cat.get('name_ar', '')} — {cat.get('name_en', '')}",
        "riwaya": riwaya,
        "comptage": COMPTAGE[riwaya],
        "style": cat.get("style"),
        "contexte": cat.get("recording_context"),
        "canal": cat.get("channel"),
        "source": {
            "nom": f"Quranic Universal Audio {VERSION} (Wider Community)",
            "url": f"https://github.com/Wider-Community/quranic-universal-audio/releases/tag/{VERSION}",
            "licence": "CC BY 4.0 — attribution à Wider Community",
            "texte": donnees["_meta"].get("script"),
        },
        "niveau": "publie",
        "production": "alignement phonémique, validateurs automatiques, correction humaine possible (documentation QUA)",
        "couverture": {"versets": couverture_v, "versets_avec_mots": couverture_m,
                       "annoncee": cat.get("coverage")},
        "reprises": reprises,
    }
    ecrire_fiche(nom_rec, fiche)
    print(f"{nom_rec} : {couverture_v} versets, {couverture_m} avec mots, {len(reprises)} versets repris", flush=True)
    return fiche


def main(args: list[str]) -> None:
    recs = catalogue()
    if not args or args[0] == "lister":
        for r in recs:
            marque = "" if diffusable(r) else "  (non diffusable)"
            print(f"{r['slug']:<48} {RIWAYA[r['riwayah']]:<6} {r['coverage']['ayahs']:>5}{marque}")
        return
    cmd, slugs = args[0], args[1:]
    if slugs == ["tout"]:
        slugs = [r["slug"] for r in recs if diffusable(r)]
    echecs = []
    for sl in slugs:
        try:
            if cmd == "recolter":
                recolter(sl)
            elif cmd == "normaliser":
                normaliser(sl)
        except Exception as e:  # une archive en échec ne doit pas arrêter les autres
            echecs.append(sl)
            print(f"ÉCHEC {sl} : {e}", flush=True)
    if echecs:
        print(f"{len(echecs)} en échec, à relancer : {' '.join(echecs)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
