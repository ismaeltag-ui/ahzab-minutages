"""Faire tourner une passe sur Kaggle (GPU gratuit, environ 30 h par semaine).

Le compte et sa connexion sont faits une fois pour toutes par son titulaire
(`kaggle auth login`) ; ensuite tout se pilote d'ici :

    py -m outil.kaggle preparer <récitation> <url du dossier> [--sourates 1-114] [--nom "…"] [--budget-h 10.5]
    py -m outil.kaggle lancer <récitation>
    py -m outil.kaggle etat <récitation>
    py -m outil.kaggle recuperer <récitation>

`preparer` écrit dans `cache/kaggle/<récitation>/` un script qui, sur le serveur,
installe les dépendances, clone ce dépôt **au commit courant** (il doit donc être
poussé), fait la passe et range les tables dans la sortie du calcul.
`recuperer` rapatrie ces tables dans `donnees/<récitation>/` et fusionne la fiche.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .commun import CACHE, DONNEES, ecrire_json, lire_json

DEPOT = "https://github.com/ismaeltag-ui/ahzab-minutages"
KAGGLE = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
PAQUETS = ["quran-muaalem==0.2.2", "quran-transcript==0.6.4", "transformers==4.57.6", "imageio-ffmpeg"]

MODELE_SCRIPT = r'''# Passe ahzab-minutages sur Kaggle — généré par `py -m outil.kaggle preparer`.
import json, os, shutil, subprocess, sys

P = json.loads(r"""__PARAMS__""")


def lancer(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


lancer("nvidia-smi")
lancer(sys.executable, "-m", "pip", "install", "-q", *P["paquets"], check=True)
src = "/tmp/ahzab-minutages"
if not os.path.exists(src):
    lancer("git", "clone", "-q", P["depot"], src, check=True)
lancer("git", "-C", src, "checkout", "-q", P["commit"], check=True)
env = dict(os.environ, AHZAB_FILS="0", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
lancer(sys.executable, "-m", "outil.passe", P["rec"], P["dossier"], "--sourates", P["sourates"],
       "--nom", P["nom"], "--budget-h", str(P["budget_h"]), cwd=src, env=env)
# ce qui est fait est rangé même si la passe s'est arrêtée en route
sortie = "/kaggle/working"
if os.path.isdir(f"{src}/donnees/{P['rec']}"):
    shutil.copytree(f"{src}/donnees/{P['rec']}", f"{sortie}/{P['rec']}", dirs_exist_ok=True)
if os.path.exists(f"{src}/cache/passe-{P['rec']}.json"):
    shutil.copy(f"{src}/cache/passe-{P['rec']}.json", f"{sortie}/passe.json")
'''


def kaggle(*args: str, capture: bool = True) -> str:
    r = subprocess.run([str(KAGGLE), *args], capture_output=capture, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} : {(r.stderr or r.stdout or '').strip()}")
    return r.stdout or ""


def compte() -> str:
    sortie = kaggle("config", "view")
    m = re.search(r"username:\s*(\S+)", sortie)
    if not m or m.group(1) in ("None", ""):
        raise RuntimeError("compte Kaggle inconnu : le titulaire doit d'abord lancer `kaggle auth login`")
    return m.group(1)


def slug(rec: str) -> str:
    return "ahzab-" + re.sub(r"[^a-z0-9-]", "-", rec.lower())


def dossier(rec: str) -> Path:
    return CACHE / "kaggle" / rec


def commit_courant() -> str:
    racine = Path(__file__).resolve().parent.parent
    etat = subprocess.run(["git", "-C", str(racine), "status", "--porcelain", "outil"],
                          capture_output=True, text=True).stdout.strip()
    if etat:
        raise RuntimeError("des modifications de outil/ ne sont pas commitées : le serveur ne les verrait pas")
    sha = subprocess.run(["git", "-C", str(racine), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    distant = subprocess.run(["git", "-C", str(racine), "branch", "-r", "--contains", sha],
                             capture_output=True, text=True).stdout.strip()
    if not distant:
        raise RuntimeError(f"le commit {sha[:7]} n'est pas poussé : le serveur ne pourrait pas le cloner")
    return sha


def preparer(rec: str, url_dossier: str, sourates: str = "1-114", nom: str | None = None,
             budget_h: float = 10.5) -> Path:
    d = dossier(rec)
    d.mkdir(parents=True, exist_ok=True)
    params = {"rec": rec, "dossier": url_dossier, "sourates": sourates, "nom": nom or rec,
              "budget_h": budget_h, "depot": DEPOT, "commit": commit_courant(), "paquets": PAQUETS}
    (d / "passe_kaggle.py").write_text(MODELE_SCRIPT.replace("__PARAMS__", json.dumps(params, ensure_ascii=False)),
                                       encoding="utf-8", newline="\n")
    ecrire_json(d / "kernel-metadata.json", {
        "id": f"{compte()}/{slug(rec)}", "title": slug(rec), "code_file": "passe_kaggle.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, compact=False)
    print(f"préparé : {d} (commit {params['commit'][:7]}, sourates {sourates}, budget {budget_h} h)")
    return d


def lancer(rec: str) -> None:
    print(kaggle("kernels", "push", "-p", str(dossier(rec))).strip())


def etat(rec: str) -> str:
    s = kaggle("kernels", "status", f"{compte()}/{slug(rec)}").strip()
    print(s)
    return s


def recuperer(rec: str) -> None:
    sortie = dossier(rec) / "sortie"
    if sortie.exists():
        shutil.rmtree(sortie)
    sortie.mkdir(parents=True)
    kaggle("kernels", "output", f"{compte()}/{slug(rec)}", "-p", str(sortie), "--force")
    distant = sortie / rec
    if not distant.is_dir():
        print("aucune table dans la sortie ; journal :", *sorted(p.name for p in sortie.iterdir()))
        return
    local = DONNEES / rec
    local.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(distant.glob("[0-9][0-9][0-9].json")):
        shutil.copy(f, local / f.name)
        n += 1
    # la fiche : on garde la locale et l'on y reporte les passes faites là-bas
    fd = distant / "recitation.json"
    if fd.exists():
        fiche_d = lire_json(fd)
        fl = local / "recitation.json"
        fiche = lire_json(fl) if fl.exists() else fiche_d
        for cle in ("passes", "vitesse"):
            fiche.setdefault(cle, {}).update(fiche_d.get(cle, {}))
            fiche[cle] = dict(sorted(fiche[cle].items()))
        ecrire_json(fl, fiche, compact=False)
    bilan = sortie / "passe.json"
    print(f"{n} sourates rapatriées dans {local}")
    if bilan.exists():
        b = lire_json(bilan)
        print(f"bilan : {len(b['faites'])} faites, {len(b['sautees'])} sautées, "
              f"erreurs {b['erreurs'] or 'aucune'}, non commencées {b['non_commencees'] or 'aucune'}")


def main(args: list[str]) -> None:
    if len(args) < 2:
        print(__doc__)
        return
    cmd, rec = args[0], args[1]
    if cmd == "preparer":
        opts = dict(zip(args[3::2], args[4::2]))
        preparer(rec, args[2], opts.get("--sourates", "1-114"), opts.get("--nom"),
                 float(opts.get("--budget-h", "10.5")))
    elif cmd == "lancer":
        lancer(rec)
    elif cmd == "etat":
        etat(rec)
    elif cmd == "recuperer":
        recuperer(rec)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
