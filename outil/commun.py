"""Outils communs : chemins, accès réseau, comptes de versets et de mots par riwaya.

Tout ce qui est téléchargé va dans `cache/` (ignoré par git) ; seules les tables
normalisées vont dans `donnees/`.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CACHE = RACINE / "cache"
DONNEES = RACINE / "donnees"

AGENT = "ahzab-minutages/0.1 (+https://github.com/ismaeltag-ui/ahzab-minutages)"


def lire_json(chemin: Path):
    ouvrir = gzip.open if chemin.suffix == ".gz" else open
    with ouvrir(chemin, "rt", encoding="utf-8") as f:
        return json.load(f)


def ecrire_json(chemin: Path, donnees, compact: bool = True) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="\n") as f:
        if compact:
            json.dump(donnees, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(donnees, f, ensure_ascii=False, indent=2)
        f.write("\n")


def http_brut(url: str, essais: int = 6, delai: float = 3.0, methode: str = "GET"):
    """Rend (corps, entêtes). Réessaie sur les erreurs réseau et les 5xx."""
    derniere = None
    for i in range(essais):
        try:
            req = urllib.request.Request(url, method=methode, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            derniere = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            derniere = e
        time.sleep(delai * (i + 1))
    raise RuntimeError(f"échec après {essais} essais : {url} ({derniere})")


def http_json(url: str, **kw):
    corps, _ = http_brut(url, **kw)
    return json.loads(corps.decode("utf-8"))


def taille_distante(url: str) -> int | None:
    """Taille du fichier par une requête HEAD : c'est l'empreinte qu'utilise
    l'application (`lib/verseTiming.ts`) pour savoir si un minutage vaut encore."""
    try:
        _, entetes = http_brut(url, methode="HEAD", essais=2)
        valeur = entetes.get("Content-Length") or entetes.get("content-length")
        return int(valeur) if valeur else None
    except Exception:
        return None


# --- Comptes attendus -------------------------------------------------------

RIWAYAS = ("hafs", "warsh", "qalun", "shuba")


@lru_cache(maxsize=None)
def comptes(riwaya: str) -> dict[int, list[int]]:
    """{sourate: [nombre de mots du verset 1, du verset 2, …]} dans le comptage
    propre à la riwaya (koufi pour Hafs et Shuʿba, médinois pour Warsh et Qālūn).

    Hafs : `surah_info.json` de Quranic Universal Audio (mêmes mots que Quran.com).
    Autres : `<riwaya>_words.json.gz` du même projet, clé `s:a:m`."""
    qua = CACHE / "qua"
    if riwaya == "hafs":
        info = lire_json(qua / "surah_info.json")
        return {int(s): [v["num_words"] for v in d["verses"]] for s, d in info.items()}
    nom = {"warsh": "warsh", "qalun": "qalun", "shuba": "shuba"}[riwaya]
    mots = lire_json(qua / f"{nom}_words.json.gz")  # [{"ref": "1:1:1", "text": …}, …]
    cles = mots.keys() if isinstance(mots, dict) else [m["ref"] for m in mots]
    maxi: dict[tuple[int, int], int] = {}
    for cle in cles:
        s, a, m = (int(x) for x in cle.split(":"))
        maxi[(s, a)] = max(maxi.get((s, a), 0), m)
    res: dict[int, list[int]] = {}
    for (s, a), n in sorted(maxi.items()):
        res.setdefault(s, [])
        while len(res[s]) < a - 1:
            res[s].append(0)
        res[s].append(n)
    return res
