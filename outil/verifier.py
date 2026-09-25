"""Contrôles internes d'une table normalisée, sans oreille ni référence extérieure.

Ce qu'on vérifie, sourate par sourate :
- versets absents, mots absents ;
- bornes à l'envers (début après la fin), versets qui se chevauchent ;
- mots qui débordent de leur verset ;
- mots anormalement courts (< 120 ms : aucun mot du Coran ne se dit si vite,
  c'est la signature des bornes de remplissage) ou longs (> 12 s).

    py -m outil.verifier tout
    py -m outil.verifier qua-ahmed_talib_bin_humaid_mp3quran
"""

from __future__ import annotations

import sys

from .commun import DONNEES, ecrire_json, lire_json

COURT_MS = 120
LONG_MS = 12_000
TOLERANCE_MS = 60


def verifier(rec: str) -> dict:
    dossier = DONNEES / rec
    b = {"sourates": 0, "versets": 0, "versets_absents": 0, "versets_sans_mots": 0,
         "mots": 0, "mots_absents": 0, "a_l_envers": 0, "chevauchements": 0,
         "hors_verset": 0, "courts": 0, "longs": 0}
    exemples: dict[str, list[str]] = {k: [] for k in ("a_l_envers", "chevauchements", "hors_verset", "courts", "longs")}

    def note(cle: str, ref: str) -> None:
        b[cle] += 1
        if len(exemples[cle]) < 8:
            exemples[cle].append(ref)

    for s in range(1, 115):
        f = dossier / f"{s:03d}.json"
        if not f.exists():
            continue
        t = lire_json(f)
        b["sourates"] += 1
        fin_prec = None
        for i, v in enumerate(t["versets"]):
            ref = f"{s}:{i + 1}"
            b["versets"] += 1
            if v is None:
                b["versets_absents"] += 1
                continue
            if v[0] > v[1]:
                note("a_l_envers", ref)
            if fin_prec is not None and v[0] < fin_prec - TOLERANCE_MS:
                note("chevauchements", ref)
            fin_prec = v[1]
            mots = t["mots"][i]
            if mots is None:
                b["versets_sans_mots"] += 1
                continue
            for w in range(0, len(mots), 2):
                d, e = mots[w], mots[w + 1]
                b["mots"] += 1
                r = f"{ref}:{w // 2 + 1}"
                if d < 0:
                    b["mots_absents"] += 1
                    continue
                if d > e:
                    note("a_l_envers", r)
                if d < v[0] - TOLERANCE_MS or e > v[1] + TOLERANCE_MS:
                    note("hors_verset", r)
                if e - d < COURT_MS:
                    note("courts", r)
                elif e - d > LONG_MS:
                    note("longs", r)
    presents = b["mots"] - b["mots_absents"]
    b["taux_courts_pct"] = round(100 * b["courts"] / presents, 2) if presents else None
    res = {"bilan": b, "exemples": exemples}
    ecrire_json(dossier / "verification.json", res, compact=False)
    return res


def main(args: list[str]) -> None:
    recs = sorted(p.name for p in DONNEES.iterdir() if p.is_dir()) if args == ["tout"] else args
    print(f"{'récitation':<52} {'sour':>4} {'vers.':>5} {'abs':>4} {'s/mots':>6} {'mots':>6} "
          f"{'m.abs':>5} {'envers':>6} {'chev':>4} {'hors':>5} {'courts%':>7}")
    for rec in recs:
        b = verifier(rec)["bilan"]
        print(f"{rec:<52} {b['sourates']:>4} {b['versets']:>5} {b['versets_absents']:>4} "
              f"{b['versets_sans_mots']:>6} {b['mots']:>6} {b['mots_absents']:>5} {b['a_l_envers']:>6} "
              f"{b['chevauchements']:>4} {b['hors_verset']:>5} {str(b['taux_courts_pct']):>7}")


if __name__ == "__main__":
    main(sys.argv[1:])
