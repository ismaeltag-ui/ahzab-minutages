"""Confronter deux sources qui ont minuté **le même fichier audio**.

Deux tables relevées indépendamment sur le même enregistrement doivent tomber au
même endroit. Leur écart mesure la précision de la moins bonne des deux — c'est
ainsi qu'on juge une source sans oreille : QUA contre les bornes d'éditeur de
mp3quran, QUA contre QUL, et plus tard notre propre aligneur contre les deux.

    py -m outil.comparer              # toutes les paires qui partagent un fichier
"""

from __future__ import annotations

import re
import statistics
import sys
from itertools import combinations

from .commun import DONNEES, ecrire_json, lire_json


def cle_dossier(url: str) -> str:
    u = re.sub(r"^https?://", "", url.strip().lower())
    u = re.sub(r"^www\.", "", u)
    u = re.sub(r"/+", "/", u)
    return u.rsplit("/", 1)[0]


def recitations() -> dict[str, str]:
    """{récitation: dossier audio}"""
    res = {}
    for d in sorted(p for p in DONNEES.iterdir() if p.is_dir()):
        for s in range(1, 115):
            f = d / f"{s:03d}.json"
            if f.exists():
                res[d.name] = cle_dossier(lire_json(f)["audio"])
                break
    return res


def stats(ecarts: list[int]) -> dict | None:
    if not ecarts:
        return None
    a = sorted(abs(x) for x in ecarts)
    q = lambda p: a[min(len(a) - 1, int(p * len(a)))]
    return {"n": len(a), "mediane": q(0.5), "p90": q(0.9), "p95": q(0.95), "pire": a[-1],
            "biais": round(statistics.median(ecarts)), "sous_50ms_pct": round(100 * sum(1 for x in a if x <= 50) / len(a), 1)}


def comparer(a: str, b: str) -> dict:
    vd, vf, md, mf = [], [], [], []
    for s in range(1, 115):
        fa, fb = DONNEES / a / f"{s:03d}.json", DONNEES / b / f"{s:03d}.json"
        if not (fa.exists() and fb.exists()):
            continue
        ta, tb = lire_json(fa), lire_json(fb)
        if len(ta["versets"]) != len(tb["versets"]):
            continue  # comptages différents : pas comparable verset à verset
        for i, (x, y) in enumerate(zip(ta["versets"], tb["versets"])):
            if x and y:
                vd.append(y[0] - x[0])
                vf.append(y[1] - x[1])
            ma, mb = ta["mots"][i], tb["mots"][i]
            if ma and mb and len(ma) == len(mb):
                for w in range(0, len(ma), 2):
                    if ma[w] >= 0 and mb[w] >= 0:
                        md.append(mb[w] - ma[w])
                        mf.append(mb[w + 1] - ma[w + 1])
    return {"a": a, "b": b, "versets_debut": stats(vd), "versets_fin": stats(vf),
            "mots_debut": stats(md), "mots_fin": stats(mf)}


def main(args: list[str]) -> None:
    dossiers = recitations()
    groupes: dict[str, list[str]] = {}
    for rec, dos in dossiers.items():
        groupes.setdefault(dos, []).append(rec)
    res = []
    for dos, recs in sorted(groupes.items()):
        for a, b in combinations(sorted(recs), 2):
            c = comparer(a, b)
            c["audio"] = dos
            res.append(c)
            def f(x):
                return "—" if not x else f"{x['mediane']:>4}/{x['p90']:>5}/{x['pire']:>6} ms (biais {x['biais']:+d}, n={x['n']})"
            print(f"\n{a}  ↔  {b}\n  fichier : {dos}\n  versets, début : {f(c['versets_debut'])}\n"
                  f"  versets, fin   : {f(c['versets_fin'])}\n  mots, début    : {f(c['mots_debut'])}\n"
                  f"  mots, fin      : {f(c['mots_fin'])}", flush=True)
    ecrire_json(DONNEES / "comparaisons.json", res, compact=False)
    print(f"\n{len(res)} paires comparées (médiane / 90e centile / pire).")


if __name__ == "__main__":
    main(sys.argv[1:])
