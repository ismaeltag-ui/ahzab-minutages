"""Durée réelle d'un MP3 distant, sans le télécharger : taille par HEAD, puis
l'en-tête lu par une requête partielle (les premiers kilo-octets suffisent).

C'est le garde-fou qui manquait : une table peut être impeccable en elle-même
(bornes croissantes, tous les mots) et pourtant décrire **un autre enregistrement**
que celui que sert l'adresse aujourd'hui. Relevé du 2026-09-25 : la table
mp3quran d'Ibn Humaid s'arrête à 1687 s dans Al-Kahf, dont le fichier dure 2044 s.

    py -m outil.duree https://…/018.mp3
    py -m outil.duree auditer                 # toutes les tables, sourates témoins
    py -m outil.duree auditer --toutes        # les 114 sourates
"""

from __future__ import annotations

import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .commun import AGENT, CACHE, DONNEES, ecrire_json, lire_json, taille_distante

FICHIER = CACHE / "durees.json"
_verrou = Lock()
_durees: dict | None = None

TEMOINS = (1, 2, 12, 18, 36, 55, 67, 78, 93, 112)

_DEBITS = {  # kbit/s, couche III
    1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}
_FREQ = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def _plage(url: str, debut: int, fin: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Range": f"bytes={debut}-{fin}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def mesurer(url: str) -> dict | None:
    """{"octets", "duree_ms", "mode": "xing"|"vbri"|"cbr"} ou None."""
    octets = taille_distante(url)
    if not octets:
        return None
    tete = _plage(url, 0, 65535)
    debut = 0
    if tete[:3] == b"ID3":
        taille = (tete[6] << 21) | (tete[7] << 14) | (tete[8] << 7) | tete[9]
        debut = 10 + taille + (10 if tete[5] & 0x10 else 0)
        tete = _plage(url, debut, debut + 65535)
        debut_local = 0
    else:
        debut_local = 0
    i = debut_local
    while i < len(tete) - 4:
        if tete[i] == 0xFF and (tete[i + 1] & 0xE0) == 0xE0:
            h = int.from_bytes(tete[i:i + 4], "big")
            version = (h >> 19) & 3  # 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
            couche = (h >> 17) & 3
            idx_debit = (h >> 12) & 15
            idx_freq = (h >> 10) & 3
            if couche == 1 and version != 1 and 0 < idx_debit < 15 and idx_freq < 3:
                break
        i += 1
    else:
        return None
    mpeg1 = version == 3
    freq = _FREQ[version][idx_freq]
    debit = _DEBITS[1 if mpeg1 else 2][idx_debit] * 1000
    mono = ((h >> 6) & 3) == 3
    ech_par_trame = 1152 if mpeg1 else 576
    cote = (17 if mono else 32) if mpeg1 else (9 if mono else 17)
    x = i + 4 + cote
    if tete[x:x + 4] in (b"Xing", b"Info") and tete[x + 7] & 1:
        trames = int.from_bytes(tete[x + 8:x + 12], "big")
        return {"octets": octets, "duree_ms": round(trames * ech_par_trame * 1000 / freq), "mode": "xing"}
    v = i + 4 + 32
    if tete[v:v + 4] == b"VBRI":
        trames = int.from_bytes(tete[v + 14:v + 18], "big")
        return {"octets": octets, "duree_ms": round(trames * ech_par_trame * 1000 / freq), "mode": "vbri"}
    audio = octets - (debut + i) - 128
    return {"octets": octets, "duree_ms": round(audio * 8 * 1000 / debit), "mode": "cbr"}


def duree(url: str) -> dict | None:
    global _durees
    with _verrou:
        if _durees is None:
            _durees = lire_json(FICHIER) if FICHIER.exists() else {}
        if url in _durees:
            return _durees[url]
    try:
        d = mesurer(url)
    except Exception:
        d = None
    with _verrou:
        _durees[url] = d
    return d


def sauver() -> None:
    if _durees is not None:
        ecrire_json(FICHIER, _durees, compact=False)


def auditer(toutes: bool = False) -> dict:
    """Pour chaque table : la fin du dernier verset contre la durée du fichier.
    Au-delà de la durée, ou très en deçà, la table décrit un autre enregistrement."""
    sourates = range(1, 115) if toutes else TEMOINS
    travaux = []
    for d in sorted(p for p in DONNEES.iterdir() if p.is_dir()):
        for s in sourates:
            f = d / f"{s:03d}.json"
            if f.exists():
                t = lire_json(f)
                fins = [v[1] for v in t["versets"] if v]
                if fins and t["audio"]:
                    travaux.append((d.name, s, t["audio"], max(fins)))
    with ThreadPoolExecutor(max_workers=6) as ex:
        mesures = list(ex.map(lambda w: duree(w[2]), travaux))
    sauver()
    res: dict[str, dict] = {}
    for (rec, s, url, fin), m in zip(travaux, mesures):
        r = res.setdefault(rec, {"justes": 0, "au_dela": [], "en_deca": [], "sans_mesure": 0})
        if not m:
            r["sans_mesure"] += 1
            continue
        ecart = m["duree_ms"] - fin  # silence final attendu : positif et court
        if ecart < -1500:
            r["au_dela"].append([s, round(-ecart / 1000, 1)])
        elif ecart > max(20000, m["duree_ms"] * 0.08):
            r["en_deca"].append([s, round(ecart / 1000, 1)])
        else:
            r["justes"] += 1
    ecrire_json(DONNEES / "audit-durees.json", res, compact=False)
    for rec, r in res.items():
        mauvais = len(r["au_dela"]) + len(r["en_deca"])
        marque = "  <-- AUTRE ENREGISTREMENT" if mauvais > r["justes"] else ("  (à voir)" if mauvais else "")
        print(f"{rec:<52} justes {r['justes']:>3}  au-delà {len(r['au_dela']):>3}  en deçà {len(r['en_deca']):>3}"
              f"  non mesurées {r['sans_mesure']:>3}{marque}", flush=True)
    return res


def main(args: list[str]) -> None:
    if args and args[0] == "auditer":
        auditer(toutes="--toutes" in args)
        return
    for url in args:
        print(url, mesurer(url))


if __name__ == "__main__":
    main(sys.argv[1:])
