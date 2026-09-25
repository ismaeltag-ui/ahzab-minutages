"""Retrouver des fichiers de versets dans le fichier de leur sourate (niveau « reporté »).

Quand un même enregistrement est publié verset par verset (everyayah) **et** en
sourates entières (mp3quran…), chaque fichier de verset est un morceau exact du
fichier de sourate : la corrélation de leurs signaux le retrouve à la
milliseconde, et ses bornes sont alors **les coupes de l'éditeur**, rien n'est
deviné. Si la corrélation ne trouve rien de net, ce n'est pas la même prise :
c'est aussi le contrôle qui dit si l'on peut passer une voix au fichier de sourate
sans changer d'enregistrement.

    py -m outil.empreinte comparer <dossier everyayah ou url des versets> <url du dossier de sourates> [--sourates 112,113]

Le score est une corrélation normalisée (1 = identique). Mesuré le 2026-09-25 :
même prise au-dessus de 0,8 ; autre prise sous 0,3.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .aligner import charger_audio
from .commun import CACHE, comptes, http_brut

SR_E = 8000  # la forme d'onde à 8 kHz suffit à reconnaître une prise
EVERYAYAH = "https://everyayah.com/data/"


def _cache(url: str) -> str:
    nom = url.split("://", 1)[1].replace("/", "_")
    f = CACHE / "empreinte" / nom
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(http_brut(url)[0])
    return str(f)


def signal(url: str) -> np.ndarray:
    a = charger_audio(_cache(url))  # 16 kHz
    return a[::2].astype(np.float64)  # 8 kHz, sans filtre : le bruit de repliement est commun aux deux


def placer(verset: np.ndarray, sourate: np.ndarray, depuis: int = 0) -> tuple[float, float]:
    """(début en s, score) du meilleur emplacement de `verset` dans `sourate[depuis:]`."""
    from scipy.signal import fftconvolve

    v = verset - verset.mean()
    s = sourate[depuis:]
    if len(s) < len(v):
        return -1.0, 0.0
    num = fftconvolve(s, v[::-1], mode="valid")
    cumul = np.concatenate(([0.0], np.cumsum(s * s)))
    energie = cumul[len(v):] - cumul[:-len(v)]
    score = num / (np.sqrt(np.maximum(energie, 1e-9)) * np.linalg.norm(v))
    k = int(np.argmax(score))
    return (depuis + k) / SR_E, float(score[k])


def comparer_sourate(dossier_ea: str, dossier_s: str, s: int) -> list[tuple[int, float, float, float]]:
    """[(verset, début s, fin s, score)] — chaque verset cherché après le précédent."""
    n = len(comptes("hafs")[s])
    base = dossier_ea if dossier_ea.startswith("http") else f"{EVERYAYAH}{dossier_ea}/"
    urls = [f"{base}{s:03d}{a:03d}.mp3" for a in range(1, n + 1)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        versets = list(ex.map(signal, urls))
    sourate = signal(f"{dossier_s}{s:03d}.mp3")
    res, pos = [], 0
    for a, v in enumerate(versets, start=1):
        # on cherche le cœur du verset (sans les silences de bord que l'éditeur a pu ajouter)
        e = np.abs(v) > 0.02 * np.abs(v).max()
        i, j = int(np.argmax(e)), len(e) - int(np.argmax(e[::-1]))
        coeur = v[i:j] if j - i > SR_E // 2 else v
        debut, score = placer(coeur, sourate, pos)
        res.append((a, debut - i / SR_E, debut - i / SR_E + len(v) / SR_E, score))
        if score > 0.5:
            pos = int((debut + len(coeur) / SR_E) * SR_E) - SR_E // 10
    return res


def main(args: list[str]) -> None:
    if len(args) < 3 or args[0] != "comparer":
        print(__doc__)
        return
    opts = dict(zip(args[3::2], args[4::2]))
    sourates = [int(x) for x in opts.get("--sourates", "112,113").split(",")]
    for s in sourates:
        r = comparer_sourate(args[1], args[2], s)
        scores = [x[3] for x in r]
        print(f"{args[1]} {s:03d} : score médian {np.median(scores):.2f}, minimum {min(scores):.2f} "
              f"({len(r)} versets)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
