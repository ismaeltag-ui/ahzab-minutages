"""Écart mot à mot entre deux tables d'un même fichier, sur les versets communs.

    py -m outil.mesurer aligne-humaid qua-ahmed_talib_bin_humaid_mp3quran 1 18
"""

from __future__ import annotations

import sys

from .commun import DONNEES, lire_json
from .comparer import stats


def ecarts(a: str, b: str, sourates: list[int], detail: bool = False) -> dict:
    vd, vf, md, mf = [], [], [], []
    for s in sourates:
        ta, tb = lire_json(DONNEES / a / f"{s:03d}.json"), lire_json(DONNEES / b / f"{s:03d}.json")
        for i, (x, y) in enumerate(zip(ta["versets"], tb["versets"])):
            if not (x and y):
                continue
            vd.append(x[0] - y[0])
            vf.append(x[1] - y[1])
            ma, mb = ta["mots"][i], tb["mots"][i]
            if not (ma and mb):
                continue
            for w in range(0, min(len(ma), len(mb)), 2):
                if ma[w] < 0 or mb[w] < 0:
                    continue
                md.append(ma[w] - mb[w])
                mf.append(ma[w + 1] - mb[w + 1])
                if detail:
                    print(f"{s}:{i + 1}:{w // 2 + 1:<3} début {ma[w] - mb[w]:+6d}  fin {ma[w + 1] - mb[w + 1]:+6d}"
                          f"   ({ma[w]}-{ma[w + 1]} contre {mb[w]}-{mb[w + 1]})")
    return {"versets_debut": stats(vd), "versets_fin": stats(vf), "mots_debut": stats(md), "mots_fin": stats(mf)}


def main(args: list[str]) -> None:
    detail = "--detail" in args
    args = [x for x in args if x != "--detail"]
    r = ecarts(args[0], args[1], [int(x) for x in args[2:]], detail)
    for k, v in r.items():
        if v:
            print(f"{k:<14} médiane {v['mediane']:>5} ms  90e {v['p90']:>5}  95e {v['p95']:>5}  pire {v['pire']:>6}"
                  f"  biais {v['biais']:+5d}  ≤50 ms {v['sous_50ms_pct']:>5} %  (n={v['n']})")


if __name__ == "__main__":
    main(sys.argv[1:])
