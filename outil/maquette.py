"""Prépare les extraits et les données de la maquette d'écoute.

    py -m outil.maquette <dossier de sortie>

Écrit `<sortie>/audio/…` (extraits MP3 mono 64 kbit/s) et `<sortie>/donnees.json`.
Rien de ceci n'entre dans le dépôt : l'audio appartient à ses éditeurs, la
maquette ne sert qu'à l'écoute privée.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .commun import CACHE, DONNEES, lire_json
from .comparer import stats

AUDIO = CACHE / "audio"
EVERYAYAH = "https://everyayah.com/data"

BASMALAS = [  # dossier everyayah, nom affiché
    ("Husary_128kbps", "Mahmoud Khalil al-Husary"),
    ("Husary_Muallim_128kbps", "Al-Husary, muʿallim"),
    ("Minshawy_Murattal_128kbps", "Muhammad Siddiq al-Minshawi"),
    ("Abdul_Basit_Murattal_192kbps", "ʿAbd al-Basit ʿAbd as-Samad"),
    ("Hudhaify_128kbps", "ʿAli al-Hudhayfi"),
    ("Muhammad_Ayyoub_128kbps", "Muhammad Ayyub"),
    ("Abdullah_Matroud_128kbps", "ʿAbdallah al-Matroud"),
    ("Nasser_Alqatami_128kbps", "Nasser al-Qatami"),
    ("Yasser_Ad-Dussary_128kbps", "Yasser ad-Dussary"),
    ("MaherAlMuaiqly128kbps", "Maher al-Muaiqly"),
    ("Saood_ash-Shuraym_128kbps", "Saʿud ash-Shuraym"),
    ("Ghamadi_40kbps", "Saʿd al-Ghamdi"),
    ("Ali_Jaber_64kbps", "ʿAli Jaber"),
]

PISTES = [  # identifiant, récitant, sourate, titre, table alignée, table publiée (même fichier)
    ("badr-001", "Badr at-Turki", 1, "Al-Fatiha", "aligne-badr", None),
    ("badr-018", "Badr at-Turki", 18, "Al-Kahf, versets 1 à 10", "aligne-badr", None),
    ("badr-078", "Badr at-Turki", 78, "An-Naba", "aligne-badr", None),
    ("humaid-001", "Ahmad Talib ibn Humaid", 1, "Al-Fatiha", "aligne-humaid", "qua-ahmed_talib_bin_humaid_mp3quran"),
    ("humaid-018", "Ahmad Talib ibn Humaid", 18, "Al-Kahf, versets 1 à 10", "aligne-humaid", "qua-ahmed_talib_bin_humaid_mp3quran"),
    ("humaid-078", "Ahmad Talib ibn Humaid", 78, "An-Naba", "aligne-humaid", "qua-ahmed_talib_bin_humaid_mp3quran"),
]

VOIX_HASBUK = {
    "sa-hamed": "Hamed (Arabie saoudite)", "eg-shakir": "Shakir (Égypte)", "ae-hamdan": "Hamdan (Émirats)",
    "kw-fahed": "Fahed (Koweït)", "qa-moaz": "Moaz (Qatar)", "jo-taim": "Taim (Jordanie)",
    "sy-laith": "Laith (Syrie)", "om-abdullah": "Abdullah (Oman)",
}


def telecharger(url: str, cible: Path) -> Path:
    if not cible.exists():
        cible.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-s", "-f", "-o", str(cible), url], check=True)
    return cible


def extrait(source: Path, cible: Path, debut: float = 0.0, fin: float | None = None) -> None:
    cible.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{debut:.3f}"]
    if fin is not None:
        cmd += ["-t", f"{fin - debut:.3f}"]
    cmd += ["-i", str(source), "-ac", "1", "-ar", "44100", "-b:a", "64k", str(cible)]
    subprocess.run(cmd, check=True)


def hauteur(source: Path, debut: float = 0.0, fin: float | None = None) -> float | None:
    """Hauteur médiane de la voix, en Hz : un repère pour « la plus grave »."""
    import librosa

    from .aligner import charger_audio

    a = charger_audio(source, debut, (fin - debut) if fin else None)
    f0, voise, _ = librosa.pyin(a, fmin=60, fmax=300, sr=16000, frame_length=1024)
    f0 = f0[voise & ~np.isnan(f0)]
    return round(float(np.median(f0))) if len(f0) else None


def mots_texte(s: int, a: int) -> list[str]:
    from quran_transcript import Aya

    return Aya(s, a).get().uthmani.split(" ")


def piste(ident: str, recitant: str, s: int, titre: str, aligne: str, publie: str | None, sortie: Path) -> dict:
    ta = lire_json(DONNEES / aligne / f"{s:03d}.json")
    tp = lire_json(DONNEES / publie / f"{s:03d}.json") if publie else None
    versets = [a + 1 for a, v in enumerate(ta["versets"]) if v]
    fin_ms = max(ta["versets"][a - 1][1] for a in versets) + 1200
    source = AUDIO / ("badr" if ident.startswith("badr") else "humaid") / f"{s:03d}.mp3"
    fichier = f"audio/pistes/{ident}.mp3"
    extrait(source, sortie / fichier, 0, fin_ms / 1000)
    mots = []
    for a in versets:
        texte = mots_texte(s, a)
        lies = set(ta.get("lies", {}).get(str(a), []))
        for w, t in enumerate(texte):
            entree = {"v": a, "w": w + 1, "t": t, "lie": w in lies,
                      "al": ta["mots"][a - 1][2 * w: 2 * w + 2]}
            if tp and tp["mots"][a - 1]:
                entree["pu"] = tp["mots"][a - 1][2 * w: 2 * w + 2]
            mots.append(entree)
    return {
        "id": ident, "recitant": recitant, "sourate": s, "titre": titre, "fichier": fichier,
        "ouverture": ta.get("ouverture"), "ouverture_dite": ta.get("ouverture_dite"),
        "versets": [{"v": a, "al": ta["versets"][a - 1], "pu": tp["versets"][a - 1] if tp else None} for a in versets],
        "mots": mots, "reprises": ta.get("reprises", {}),
        "source_publiee": "Quranic Universal Audio" if publie else None,
    }


def mesures() -> dict:
    """Écarts aligneur ↔ QUA sur Ibn Humaid, séparés mots libres / mots liés."""
    res = {}
    for s in (1, 18, 78):
        fa = DONNEES / "aligne-humaid" / f"{s:03d}.json"
        if not fa.exists():
            continue
        a, b = lire_json(fa), lire_json(DONNEES / "qua-ahmed_talib_bin_humaid_mp3quran" / f"{s:03d}.json")
        g = {"debuts_libres": [], "fins_libres": [], "lies": []}
        for i, (ma, mb) in enumerate(zip(a["mots"], b["mots"])):
            if not (ma and mb):
                continue
            L = set(a.get("lies", {}).get(str(i + 1), []))
            for k in range(len(ma) // 2):
                if ma[2 * k] < 0 or mb[2 * k] < 0:
                    continue
                (g["lies"] if (k - 1) in L else g["debuts_libres"]).append(ma[2 * k] - mb[2 * k])
                (g["lies"] if k in L else g["fins_libres"]).append(ma[2 * k + 1] - mb[2 * k + 1])
        res[str(s)] = {k: stats(v) for k, v in g.items()}
    return res


def main(args: list[str]) -> None:
    sortie = Path(args[0])
    d: dict = {}

    # A. Basmala
    bas = []
    for dossier, nom in BASMALAS:
        src = telecharger(f"{EVERYAYAH}/{dossier}/001001.mp3", AUDIO / "basmala" / f"{dossier}.mp3")
        f = f"audio/basmala/{dossier}.mp3"
        extrait(src, sortie / f)
        bas.append({"id": dossier, "nom": nom, "fichier": f, "origine": "everyayah, 001001.mp3", "hz": hauteur(src)})
    for ident, nom, dos in (("humaid", "Ahmad Talib ibn Humaid", "humaid"), ("badr", "Badr at-Turki", "badr")):
        t = lire_json(DONNEES / f"aligne-{ident}" / "001.json")
        d0, d1 = t["versets"][0]
        src = AUDIO / dos / "001.mp3"
        f = f"audio/basmala/{ident}.mp3"
        extrait(src, sortie / f, max(0, d0 - 150) / 1000, (d1 + 250) / 1000)
        bas.append({"id": ident, "nom": nom, "fichier": f, "origine": "sa Fatiha, taillée par l'aligneur",
                    "hz": hauteur(src, max(0, d0 - 150) / 1000, (d1 + 250) / 1000)})
    couture = telecharger(f"{EVERYAYAH}/Minshawy_Murattal_128kbps/067001.mp3", AUDIO / "divers" / "minshawi-067001.mp3")
    extrait(couture, sortie / "audio/divers/minshawi-067001.mp3")
    d["basmala"] = {"voix": bas, "couture": {"fichier": "audio/divers/minshawi-067001.mp3",
                                             "legende": "Al-Mulk, verset 1, par al-Minshawi"}}

    # B. Ḥasbuk
    hasbuk_src = Path(args[1]) if len(args) > 1 else sortie / "audio" / "hasbuk"
    hb = []
    for f in sorted(hasbuk_src.glob("*.mp3")):
        cle, reglage = f.stem.rsplit("-", 1)
        hb.append({"id": f.stem, "voix": VOIX_HASBUK.get(cle, cle), "reglage": reglage,
                   "fichier": f"audio/hasbuk/{f.name}"})
    fin_partie = telecharger(f"{EVERYAYAH}/Husary_128kbps/112004.mp3", AUDIO / "divers" / "husary-112004.mp3")
    extrait(fin_partie, sortie / "audio/divers/husary-112004.mp3")
    d["hasbuk"] = {"voix": hb, "fin_partie": {"fichier": "audio/divers/husary-112004.mp3",
                                              "legende": "Al-Ikhlas, verset 4, par al-Husary"}}

    # C. Mot à mot
    d["pistes"] = [piste(*p, sortie) for p in PISTES if (DONNEES / p[4] / f"{p[2]:03d}.json").exists()]
    d["mesures"] = mesures()
    fiche_b = lire_json(DONNEES / "aligne-badr" / "recitation.json")
    fiche_h = lire_json(DONNEES / "aligne-humaid" / "recitation.json")
    d["vitesse"] = [{"recitant": f["nom"], "sourate": s, **p} for f in (fiche_b, fiche_h) for s, p in f.get("vitesse", {}).items()]

    # D. Continuité : al-Hudhayfi, An-Naba 31-40
    q = lire_json(DONNEES / "qua-ali_al_huthaifi_mp3quran" / "078.json")
    v31, v40 = q["versets"][30], q["versets"][39]
    debut = max(0, v31[0] - 300)
    extrait(AUDIO / "hudhayfi" / "078.mp3", sortie / "audio/continuite/sourate.mp3", debut / 1000, (v40[1] + 600) / 1000)
    fichiers = []
    for a in range(31, 41):
        f = f"audio/continuite/078{a:03d}.mp3"
        extrait(AUDIO / "hudhayfi" / f"078{a:03d}.mp3", sortie / f)
        fichiers.append({"v": a, "fichier": f})
    d["continuite"] = {
        "recitant": "ʿAli al-Hudhayfi", "sourate": 78, "titre": "An-Naba, versets 31 à 40",
        "versets_fichiers": fichiers, "sourate_fichier": "audio/continuite/sourate.mp3",
        "bornes": [{"v": a + 1, "d": q["versets"][a][0] - debut, "f": q["versets"][a][1] - debut} for a in range(30, 40)],
        "textes": {str(a): " ".join(mots_texte(78, a)) for a in range(31, 41)},
    }

    # Audit et inventaire
    d["audit"] = lire_json(DONNEES / "audit-durees.json")
    for rec, a in d["audit"].items():
        fiche = DONNEES / rec / "recitation.json"
        if fiche.exists():
            f = lire_json(fiche)
            nom = {"mp3quran-134": "Muhammad Saayed, Warsh", "mp3quran-137": "Ahmad Talib ibn Humaid"}.get(rec, f["nom"])
            a["nom"] = f"{nom[:1].upper()}{nom[1:]}, table {f['source']['nom'].split(' —')[0].split(',')[0].replace('.net', '')}"
    (sortie / "donnees.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"maquette : {len(bas)} basmalas, {len(hb)} ḥasbuk, {len(d['pistes'])} pistes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
