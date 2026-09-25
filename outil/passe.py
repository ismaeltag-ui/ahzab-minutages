"""Une passe complète : les sourates d'une récitation, téléchargées puis alignées.

Elle reprend là où elle s'est arrêtée — une sourate déjà alignée en entier est
sautée — et s'arrête d'elle-même quand le budget de temps est épuisé, pour qu'un
serveur de calcul limité à douze heures (Kaggle) ait le temps de sauver ce qui
est fait. Elle tourne aussi sur le PC, lentement.

    py -m outil.passe <récitation> <url du dossier> [--sourates 1-114] [--nom "…"]
                      [--budget-h 10.5] [--refaire] [--garder-audio]

L'URL du dossier se termine par « / » ; le fichier de la sourate s est <dossier>sss.mp3.
"""

from __future__ import annotations

import sys
import time
import traceback

from .aligner import aligner_sourate
from .commun import CACHE, DONNEES, comptes, ecrire_json, http_brut, lire_json
from .qua import _fichier


def plage(texte: str) -> list[int]:
    res: list[int] = []
    for morceau in texte.split(","):
        a, _, b = morceau.partition("-")
        res += list(range(int(a), int(b or a) + 1))
    return res


def complete(rec: str, s: int) -> bool:
    """La sourate a déjà été alignée en entier (pas un extrait)."""
    fiche_f = DONNEES / rec / "recitation.json"
    if not (DONNEES / rec / f"{s:03d}.json").exists() or not fiche_f.exists():
        return False
    p = lire_json(fiche_f).get("passes", {}).get(f"{s:03d}")
    return bool(p) and p.get("versets") == [1, len(comptes("hafs")[s])]


def telecharger(url: str, rec: str, s: int) -> str:
    cible = CACHE / "audio" / rec / f"{s:03d}.mp3"
    if not cible.exists() or cible.stat().st_size == 0:
        cible.parent.mkdir(parents=True, exist_ok=True)
        corps, _ = http_brut(url)
        partiel = cible.with_suffix(".part")
        partiel.write_bytes(corps)
        partiel.replace(cible)
    return str(cible)


def passe(rec: str, dossier: str, sourates: list[int], nom: str | None = None,
          budget_h: float = 10.5, refaire: bool = False, garder_audio: bool = False) -> dict:
    _fichier("surah_info.json")  # les comptes de mots, absents d'un clone neuf
    t0 = time.time()
    bilan = {"faites": [], "sautees": [], "erreurs": {}, "non_commencees": []}
    for k, s in enumerate(sourates):
        if time.time() - t0 > budget_h * 3600:
            bilan["non_commencees"] = sourates[k:]
            print(f"budget de {budget_h} h épuisé : arrêt avant la sourate {s}", flush=True)
            break
        if not refaire and complete(rec, s):
            bilan["sautees"].append(s)
            continue
        url = f"{dossier}{s:03d}.mp3"
        try:
            chemin = telecharger(url, rec, s)
            aligner_sourate(rec, chemin, s, audio_url=url, nom=nom)
            bilan["faites"].append(s)
            if not garder_audio:
                (CACHE / "audio" / rec / f"{s:03d}.mp3").unlink(missing_ok=True)
        except Exception as e:  # une sourate qui échoue n'arrête pas la passe
            bilan["erreurs"][s] = f"{type(e).__name__}: {e}"
            print(f"ÉCHEC sourate {s} : {e}", flush=True)
            traceback.print_exc()
        ecoule = time.time() - t0
        print(f"[{k + 1}/{len(sourates)}] {ecoule / 60:.1f} min écoulées", flush=True)
    print(f"passe {rec} : {len(bilan['faites'])} faites, {len(bilan['sautees'])} sautées, "
          f"{len(bilan['erreurs'])} en échec, {len(bilan['non_commencees'])} non commencées "
          f"— {(time.time() - t0) / 60:.1f} min", flush=True)
    return bilan


def main(args: list[str]) -> None:
    if len(args) < 2:
        print(__doc__)
        return
    rec, dossier = args[0], args[1]
    drapeaux = {a for a in args[2:] if a in ("--refaire", "--garder-audio")}
    reste = [a for a in args[2:] if a not in drapeaux]
    opts = dict(zip(reste[::2], reste[1::2]))
    bilan = passe(rec, dossier, plage(opts.get("--sourates", "1-114")), nom=opts.get("--nom"),
                  budget_h=float(opts.get("--budget-h", "10.5")), refaire="--refaire" in drapeaux,
                  garder_audio="--garder-audio" in drapeaux)
    ecrire_json(CACHE / f"passe-{rec}.json", bilan, compact=False)


if __name__ == "__main__":
    main(sys.argv[1:])
