"""Relever la table au verset d'une voix dans ses fichiers de sourate (niveau « reporté »).

Quand la même prise existe verset par verset (everyayah) et en sourates
entières (mp3quran), chaque fichier de verset est un morceau exact du fichier de
sourate : on le retrouve par corrélation, et ses bornes sont **les coupes de
l'éditeur**. Rien n'est deviné, et ce que la table de mp3quran faisait mal
disparaît d'un coup : une reprise de l'imam (qu'everyayah avait retirée) reste
hors des versets, une frontière ne peut pas être décalée, une fin ne peut pas
être coupée.

Deux façons de retrouver un verset :
- **onde** : la forme d'onde à 8 kHz (corrélation de Pearson, 1 = identique).
  Même prise : 0,9 et plus ; autre prise : sous 0,5.
- **enveloppe** : l'énergie par tranches de 5 ms, en dB. Pour un son retraité
  (Ayyoub, Al-Akhdar : égalisé, débruité) dont la forme d'onde ne se reconnaît
  plus, mais dont le découpage est resté le même.

Chaque verset est cherché après le précédent, dans les trente secondes qui suivent ;
s'il n'y est pas net, dans tout le reste du fichier, puis dans le fichier entier.
En mode onde, la recherche se fait à 2 kHz, puis s'affine à l'échantillon (8 kHz).

    py -m outil.reporter relever <voix> [--sourates 1-114]   # coupes brutes, dans cache/reporter/
    py -m outil.reporter ecrire <voix>                       # tables dans donnees/reporte-<voix>/
    py -m outil.reporter bilan <voix>                        # scores, écarts, reprises
    py -m outil.reporter controler <voix>                    # chaque verset tel que l'application le jouera
    py -m outil.reporter nettoyer <voix>                     # efface l'audio téléchargé

Les voix sont dans `VOIX` : dossier d'everyayah, adresse des sourates chez
mp3quran, numéro de lecture mp3quran, mode.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .aligner import ffmpeg
from .commun import CACHE, DONNEES, comptes, ecrire_json, http_brut, lire_json
from .format import ecrire_fiche, ecrire_sourate, table_sourate

SR = 8000
EVERYAYAH = "https://everyayah.com/data/"
MP3QURAN = "https://cdn.mp3quran.net/audio/"
ICI = CACHE / "reporter"

# id de l'application : (dossier everyayah, dossier mp3quran, lecture mp3quran, nom, mode)
VOIX = {
    "sahl-yassin": ("Sahl_Yassin_128kbps", "sahl-yassin/r1/", 32, "Sahl Yassin", "onde"),
    "fares-abbad": ("Fares_Abbad_64kbps", "fares-abbad/r1/", 81, "Fares Abbad", "onde"),
    "ajamy": ("ahmed_ibn_ali_al_ajamy_128kbps", "ahmad-ajmi/r1/", 5, "Ahmed Al-Ajamy", "onde"),
    "ghamadi": ("Ghamadi_40kbps", "saad-ghamdi/r1/", 30, "Saad Al-Ghamadi", "onde"),
    "ali-jaber": ("Ali_Jaber_64kbps", "ali-jaber/r1/", 76, "Ali Jaber", "onde"),
    "qahtani": ("Khaalid_Abdullaah_al-Qahtaanee_192kbps", "khalid-qahtani/r1/", 21, "Khalid Al-Qahtani", "onde"),
    "suesy": ("Ali_Hajjaj_AlSuesy_128kbps", "ali-souissi/r1/", 77, "Ali Hajjaj As-Suesy", "onde"),
    "neana": ("Ahmed_Neana_128kbps", "ahmad-naina/r1/", 9, "Ahmed Neana", "onde"),
    "ayyoub": ("Muhammad_Ayyoub_128kbps", "muhammad-ayyub/r1/", 109, "Muhammad Ayyoub", "enveloppe"),
    "akhdar": ("Ibrahim_Akhdar_32kbps", "ibrahim-akhdar/r1/", 1, "Ibrahim Al-Akhdar", "enveloppe"),
}

PAS_ENV = 40  # 5 ms à 8 kHz
NET = {"onde": 0.85, "enveloppe": 0.80}  # au-dessus : trouvé sans ambiguïté

# Fichiers d'everyayah que mpg123 raccourcit (trames abîmées jetées) : décodés par ffmpeg.
EA_FFMPEG = {"Ghamadi_40kbps"}


# --- Audio ------------------------------------------------------------------

def _chemin(url: str) -> Path:
    if url.startswith(EVERYAYAH):
        return ICI / "everyayah" / url[len(EVERYAYAH):]
    return ICI / "sourates" / url.split("://", 1)[1].split("/", 1)[1]


def telecharger(url: str) -> tuple[Path, int]:
    f = _chemin(url)
    deja = CACHE / "empreinte" / url.split("://", 1)[1].replace("/", "_")  # relevés de M385
    if not f.exists() and deja.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deja, f)
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        corps, _ = http_brut(url)
        tmp = f.with_suffix(".part")
        tmp.write_bytes(corps)
        tmp.replace(f)
    return f, f.stat().st_size


def decoder(f: Path) -> np.ndarray:
    """Mono, 8 kHz. libsndfile (mpg123) en mémoire : cent fois plus rapide que de
    lancer ffmpeg pour chaque petit fichier, et le même signal à l'échantillon près
    (vérifié sur Fares Abbad 082 : même longueur, décalage nul, corrélation 0,9998).

    Les fichiers de sourate passent par ffmpeg : mpg123 jette les trames abîmées
    (« dequantization failed ») et tout ce qui suit se retrouve avancé — 3,3 s à la
    fin de la sourate 3 d'Al-Ajamy. ffmpeg les garde, et sa durée est celle que
    suppose la table de mp3quran (`controles-m389/longueurs.py`).

    Les fichiers au verset des dossiers de `EA_FFMPEG` aussi : chez Al-Ghamadi
    (40 kb/s), mpg123 en raccourcit 2 869 sur 6 236, de 20 à 190 ms
    (`controles-m389/durees_locales.py`)."""
    if ICI / "sourates" in f.parents or f.parent.name in EA_FFMPEG:
        cmd = [ffmpeg(), "-v", "error", "-i", str(f), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
        return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)
    try:
        import soundfile
        import soxr

        x, sr = soundfile.read(str(f), dtype="float32", always_2d=True)
        return soxr.resample(x.mean(axis=1), sr, SR).astype(np.float32)
    except Exception:
        cmd = [ffmpeg(), "-v", "error", "-i", str(f), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
        return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def enveloppe(x: np.ndarray) -> np.ndarray:
    """Énergie par tranches de 5 ms, en dB, plancher à 60 dB sous le maximum."""
    n = len(x) // PAS_ENV
    e = 10 * np.log10((x[: n * PAS_ENV].astype(np.float64).reshape(n, PAS_ENV) ** 2).mean(axis=1) + 1e-12)
    return np.maximum(e, e.max() - 60) if n else e


# --- Recherche --------------------------------------------------------------

class Piste:
    """Un signal et ses sommes cumulées : la corrélation de Pearson d'un motif à
    chaque position d'une tranche se calcule alors en une convolution."""

    def __init__(self, x: np.ndarray, taux: int) -> None:
        self.x = x.astype(np.float64)
        self.taux = taux
        self.c1 = np.concatenate(([0.0], np.cumsum(self.x)))
        self.c2 = np.concatenate(([0.0], np.cumsum(self.x * self.x)))

    def chercher(self, motif: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
        """(position, score) du meilleur emplacement de `motif` qui commence dans [lo, hi − len)."""
        from scipy.signal import fftconvolve

        lo, hi, n = max(0, lo), min(len(self.x), hi), len(motif)
        m = motif.astype(np.float64) - motif.mean()
        nm = np.linalg.norm(m)
        if hi - lo < n or nm == 0:
            return -1, 0.0
        num = fftconvolve(self.x[lo:hi], m[::-1], mode="valid")
        somme = self.c1[lo + n:hi + 1] - self.c1[lo:hi - n + 1]
        var = (self.c2[lo + n:hi + 1] - self.c2[lo:hi - n + 1]) - somme * somme / n
        sc = num / (np.sqrt(np.maximum(var, 1e-12)) * nm)
        k = int(np.argmax(sc))
        return lo + k, float(sc[k])


def coeur(v: np.ndarray) -> tuple[int, int]:
    """Le verset sans les silences de bord (seuil : 2 % du maximum)."""
    e = np.abs(v) > 0.02 * np.abs(v).max()
    i, j = int(np.argmax(e)), len(e) - int(np.argmax(e[::-1]))
    return (i, j) if j - i > SR // 2 else (0, len(v))


GROS = 2000  # la recherche se fait d'abord à 2 kHz, puis s'affine à 8 kHz


def _reduire(x: np.ndarray) -> np.ndarray:
    import soxr

    return soxr.resample(x, SR, GROS).astype(np.float32)


def relever_sourate(voix: str, s: int) -> dict:
    dossier_ea, dossier_mq, _, _, mode = VOIX[voix]
    n = len(comptes("hafs")[s])
    url_s = f"{MP3QURAN}{dossier_mq}{s:03d}.mp3"
    urls = [f"{EVERYAYAH}{dossier_ea}/{s:03d}{a:03d}.mp3" for a in range(1, n + 1)]

    def charger(url: str) -> np.ndarray:
        return decoder(telecharger(url)[0])

    with ThreadPoolExecutor(max_workers=12) as ex:
        fut_s = ex.submit(telecharger, url_s)
        versets = list(ex.map(charger, urls))
        f_s, octets = fut_s.result()
    so = decoder(f_s)
    if mode == "enveloppe":
        r = SR // PAS_ENV  # trames par seconde
        grossier, fin = Piste(enveloppe(so), r), None
    else:
        r = GROS
        grossier, fin = Piste(_reduire(so), GROS), Piste(so, SR)
    f = SR // r  # échantillons à 8 kHz par point de la piste grossière
    seuil = NET[mode]

    res, pos = [], 0  # pos : en échantillons à 8 kHz
    for a, v in enumerate(versets, start=1):
        if len(v) < SR // 5:  # fichier sans son (Al-Ajamy 33:50 : 2 158 octets d'étiquettes)
            ms = round(pos * 1000 / SR)
            res.append({"a": a, "coupe": [ms, ms], "voix": [ms, ms], "score": 0.0, "recherche": "vide",
                        "duree_ea": round(len(v) * 1000 / SR), "vide": True})
            continue
        i, j = coeur(v)
        motif = enveloppe(v[i:j]) if mode == "enveloppe" else _reduire(v[i:j])
        essais = []
        # D'abord juste après le verset précédent : un passage en double dans le
        # fichier (Sahl Yassin, an-Nisa' 47-64) se retrouve alors dans l'ordre.
        # On ne s'éloigne que si le score **affiné** n'est pas net : à 2 kHz, un
        # décalage d'une fraction d'échantillon fait baisser le score grossier.
        for nom, lo, hi in (("fenetre", pos, pos + (j - i) + 30 * SR), ("suite", pos, len(so)), ("tout", 0, len(so))):
            k, sc = grossier.chercher(motif, lo // f, -(-hi // f))
            p = k * f
            if fin is not None and k >= 0:  # affinage à l'échantillon, à ±25 ms
                p, sc = fin.chercher(v[i:j], p - SR // 40, p + (j - i) + SR // 40)
            essais.append((nom, p, sc))
            if sc >= seuil:
                break
        recherche, p, sc = max(essais, key=lambda x: x[2])
        debut = p - i  # le fichier d'everyayah commence `i` échantillons avant son cœur
        res.append({
            "a": a,
            "coupe": [round(debut * 1000 / SR), round((debut + len(v)) * 1000 / SR)],
            "voix": [round(p * 1000 / SR), round((p + (j - i)) * 1000 / SR)],
            "score": round(sc, 4),
            "recherche": recherche,
            "duree_ea": round(len(v) * 1000 / SR),
        })
        if sc >= 0.5:
            pos = max(0, p + (j - i) - SR // 10)
    return {
        "sourate": s,
        "audio": url_s,
        "octets": octets,
        "duree": round(len(so) * 1000 / SR),
        "mode": mode,
        "versets": res,
    }


def relever(voix: str, sourates: list[int]) -> None:
    for s in sourates:
        f = ICI / "brut" / voix / f"{s:03d}.json"
        if f.exists():
            continue
        t0 = time.time()
        r = relever_sourate(voix, s)
        ecrire_json(f, r, compact=False)
        scores = [v["score"] for v in r["versets"]]
        hors = [v["a"] for v in r["versets"] if v["score"] < NET[r["mode"]]]
        print(f"{voix} {s:03d} : {len(scores)} versets, score min {min(scores):.3f}, médian "
              f"{np.median(scores):.3f}{f', douteux {hors}' if hors else ''} ({time.time() - t0:.0f} s)",
              flush=True)


# --- Examen ---------------------------------------------------------------

# Versets dont le fichier d'everyayah n'est pas le bon verset, vérifié en lisant
# les phonèmes des deux côtés (`controles-m389/phonemes.py`, modèle muaalem) :
# leurs bornes sont les coupes de leurs voisins (fin du précédent, début du
# suivant, ou fin du fichier), et ce qu'on y lit est le verset attendu.
VOISINS: dict[str, dict[tuple[int, int], str]] = {
    "fares-abbad": {
        (7, 7): "everyayah 007007.mp3 est identique à l'octet à 007008.mp3 (le verset 8) ; "
                "entre les coupes de 6 et de 8 : فَلَنَقُصَّنَّ عَلَيْهِم بِعِلْمٍ وَمَا كُنَّا غَائِبِينَ",
        (75, 40): "everyayah 075040.mp3 dit la basmala ; après la coupe de 39, jusqu'à la fin du "
                  "fichier : أَلَيْسَ ذَٰلِكَ بِقَادِرٍ عَلَىٰ أَن يُحْيِيَ الْمَوْتَىٰ",
    },
    "ajamy": {
        (29, 29): "everyayah 029029.mp3 n'a pas de son ; entre les coupes de 28 et de 30 (l'imam y redit "
                  "« إِلَّا أَن قَالُوا ائْتِنَا بِعَذَابِ اللَّهِ ») : أَئِنَّكُمْ لَتَأْتُونَ الرِّجَالَ … إِن كُنتَ مِنَ الصَّادِقِينَ",
        (33, 50): "everyayah 033050.mp3 n'a pas de son ; entre les coupes de 49 et de 51 (l'imam y redit "
                  "plusieurs passages) : يَا أَيُّهَا النَّبِيُّ إِنَّا أَحْلَلْنَا لَكَ أَزْوَاجَكَ … وَكَانَ اللَّهُ غَفُورًا رَّحِيمًا",
        (34, 29): "everyayah 034029.mp3 n'a pas de son ; entre les coupes de 28 et de 30 : وَيَقُولُونَ مَتَىٰ "
                  "هَٰذَا الْوَعْدُ إِن كُنتُمْ صَادِقِينَ",
        (37, 20): "everyayah 037020.mp3 n'a pas de son ; entre les coupes de 19 et de 21 : وَقَالُوا يَا وَيْلَنَا "
                  "هَٰذَا يَوْمُ الدِّينِ",
    },
    "neana": {
        (3, 80): "everyayah 003080.mp3 : la seconde moitié est la même prise (0,99), la première moins "
                 "(0,72), et le fichier déborde de 1,3 s sur la coupe de 81 ; entre les coupes de 79 et "
                 "de 81 : وَلَا يَأْمُرَكُمْ أَن تَتَّخِذُوا الْمَلَائِكَةَ … بَعْدَ إِذْ أَنتُم مُّسْلِمُونَ",
        (4, 24): "everyayah 004024.mp3 : la seconde moitié est la même prise (1,00), la première moins "
                 "(0,82) ; entre les coupes de 23 et de 25 : وَالْمُحْصَنَاتُ مِنَ النِّسَاءِ … إِنَّ اللَّهَ "
                 "كَانَ عَلِيمًا حَكِيمًا",
    },
    "ghamadi": {
        (16, 120): "everyayah 016120.mp3 : la seconde moitié est la même prise (0,99), la première moins "
                   "(0,51) — une pause intérieure raccourcie ; entre les coupes de 119 et de 121 : إِنَّ "
                   "إِبْرَاهِيمَ كَانَ أُمَّةً قَانِتًا لِّلَّهِ حَنِيفًا وَلَمْ يَكُ مِنَ الْمُشْرِكِينَ",
        (17, 69): "everyayah 017069.mp3 : chaque moitié est la même prise (0,92 et 0,98), le verset entier "
                  "non (0,50) — une pause intérieure raccourcie ; entre les coupes de 68 et de 70 (l'imam y "
                  "redit « ثُمَّ لَا تَجِدُوا ») : أَمْ أَمِنتُمْ أَن يُعِيدَكُمْ فِيهِ تَارَةً أُخْرَىٰ … "
                  "ثُمَّ لَا تَجِدُوا لَكُمْ عَلَيْنَا بِهِ تَبِيعًا",
        (22, 18): "everyayah 022018.mp3 : chaque moitié est la même prise (0,92 et 0,94), le verset entier "
                  "moins (0,82) — le bruit du codec à 40 kbit/s ; entre les coupes de 17 et de 19 (l'imam y "
                  "redit plusieurs passages) : أَلَمْ تَرَ أَنَّ اللَّهَ يَسْجُدُ لَهُ … إِنَّ اللَّهَ يَفْعَلُ "
                  "مَا يَشَاءُ",
    },
}

# Versets dont aucun fichier d'everyayah ne donne la coupe (un fichier qui dit deux
# versets, un autre vide) : bornes posées à la main, en millisecondes du fichier de
# sourate, au milieu d'un silence (`controles-m389/silences.py`), et vérifiées en
# lisant les phonèmes de chaque morceau.
COUPES: dict[str, dict[tuple[int, int], tuple[tuple[int, int], str]]] = {
    "ajamy": {
        (74, 40): ((268042, 275487), "everyayah 074040.mp3 n'a pas de son, et l'imam dit le verset deux "
                   "fois entre les coupes de 39 et de 41 : la première, jusqu'au creux d'énergie (-39 dB) "
                   "qui les sépare : فِي جَنَّاتٍ يَتَسَاءَلُونَ ; la reprise reste hors du verset"),
    },
    "neana": {
        (18, 61): ((1319105, 1333155), "everyayah 018061.mp3 dit les versets 61 et 62 : sa coupe, "
                   "jusqu'au milieu du silence 1332445-1333865 ms : فَلَمَّا بَلَغَا مَجْمَعَ بَيْنِهِمَا "
                   "نَسِيَا حُوتَهُمَا فَاتَّخَذَ سَبِيلَهُ فِي الْبَحْرِ سَرَبًا"),
        (18, 62): ((1333155, 1348754), "la suite de la coupe de 018061.mp3 : فَلَمَّا جَاوَزَا قَالَ "
                   "لِفَتَاهُ آتِنَا غَدَاءَنَا لَقَدْ لَقِينَا مِن سَفَرِنَا هَٰذَا نَصَبًا"),
        (18, 63): ((1348754, 1372630), "everyayah 018062.mp3 dit le verset 63, et 018063.mp3 n'a "
                   "que 0,26 s de son : la coupe de 018062.mp3 : قَالَ أَرَأَيْتَ إِذْ أَوَيْنَا إِلَى "
                   "الصَّخْرَةِ … وَاتَّخَذَ سَبِيلَهُ فِي الْبَحْرِ عَجَبًا"),
    },
}


# Deux fichiers d'everyayah qui se recouvrent de plus que la lecture d'un trait
# n'enjambe (330 ms) : chez Al-Ghamadi, le fichier de l'un porte, fondue dans sa
# traîne ou sa tête, une seconde du silence et de la voix du voisin. Les deux voix
# ne s'y touchent pas : la coupe est posée au milieu du silence qui les sépare
# (`controles-m389/jonctions.py`), et chaque morceau, lu aux phonèmes, est son
# verset et lui seul. (sourate, verset) → (coupe en ms entre lui et le suivant, raison).
JONCTIONS: dict[str, dict[tuple[int, int], tuple[int, str]]] = {
    "ghamadi": {
        (2, 25): (422605, "les fichiers de 25 et 26 se recouvrent de 1 358 ms ; silence 422350-422860 ms "
                          "(−60 dB) ; lus : 0,99 et 1,00"),
        (2, 74): (1414505, "recouvrement de 679 ms ; silence 1414400-1414610 ms (−51 dB) ; lus : 0,99 et 0,85 "
                           "(l'imam redit « وَقَدْ كَانَ فَرِيقٌ مِّنْهُمْ يَسْمَعُونَ كَلَامَ اللَّهِ » dans 75)"),
        (2, 141): (2931142, "recouvrement de 2 188 ms ; silence 2930800-2931485 ms (−52 dB) ; lus : 0,98 et 0,99"),
        (3, 152): (2926612, "recouvrement de 366 ms ; silence 2926510-2926715 ms (−49 dB) ; lus : 0,99 et 0,88 "
                            "(l'imam redit « وَالرَّسُولُ يَدْعُوكُمْ فِي أُخْرَاكُمْ » et « لِّكَيْلَا تَحْزَنُوا » "
                            "dans 153)"),
        (10, 70): (1424468, "recouvrement de 548 ms ; silence 1424245-1424690 ms (−50 dB) ; lus : 0,85 et 0,96 "
                            "(l'imam redit « ثُمَّ نُذِيقُهُمُ الْعَذَابَ الشَّدِيدَ » dans 70)"),
        (10, 89): (1753922, "recouvrement de 339 ms ; silence 1753735-1754110 ms (−49 dB) ; lus : 0,99 et 0,99"),
        (16, 50): (722408, "recouvrement de 365 ms ; silence 721965-722850 ms (−60 dB) ; lus : 1,00 et 0,99"),
    },
}

# Jour entre deux fichiers d'everyayah en deçà duquel un verset reconnu à son
# enveloppe est tenu pour accolé au précédent (`examiner`) : 100 ms, plus là où
# chaque jour jusqu'à 2 s a été lu aux phonèmes et ne contient rien
# (`controles-m389/jonctions.py --phonemes`).
JOUR_ACCOLE: dict[str, int] = {"ghamadi": 2000}  # 91 jours de 100 ms à 2 s, tous vides aux phonèmes

# Versets qu'everyayah a tirés d'un autre fichier de sourate que le leur : la
# basmala d'al-Fâtiha (1:1), que le fichier d'al-Fâtiha de mp3quran ne contient
# pas dans cette prise. Absents de la table ; l'application les lit dans ce fichier,
# à ces bornes (`lib/recitationAudio.ts`). (sourate, verset) → (audio, fenêtre en ms, raison).
AILLEURS: dict[str, dict[tuple[int, int], tuple[str, tuple[int, int], str]]] = {
    "ghamadi": {
        (1, 1): ("https://cdn.mp3quran.net/audio/saad-ghamdi/r1/004.mp3", (3937, 9000),
                 "001.mp3 s'ouvre sur l'isti'adha et une autre basmala (0,05 avec 001001.mp3) ; 001001.mp3 "
                 "est la basmala qui ouvre an-Nisâ', retrouvée dans 004.mp3 à 3 937 ms (0,96) : la voix "
                 "attaque à 3 940 ms et se tait vers 8 750, يَا أَيُّهَا النَّاسُ attaque à 9 200 ; la fenêtre "
                 "s'arrête au milieu de ce silence. Lu aux phonèmes : بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"),
    },
}


def hors_reference(voix: str) -> set[tuple[int, int]]:
    """Les versets qu'on ne peut pas comparer à leur fichier d'everyayah."""
    return set(VOISINS.get(voix, {})) | set(COUPES.get(voix, {})) | set(AILLEURS.get(voix, {}))


ENVELOPPE_MEME_PRISE = 0.90  # en mode onde, un verset retraité (fondu de fin…) se reconnaît encore là


def _env_au_meme_endroit(voix: str, s: int, v: dict, so: np.ndarray) -> float:
    dossier_ea = VOIX[voix][0]
    x = decoder(telecharger(f"{EVERYAYAH}{dossier_ea}/{s:03d}{v['a']:03d}.mp3")[0])
    i, j = coeur(x)
    p = v["voix"][0] * SR // 1000 // PAS_ENV
    return Piste(enveloppe(so), SR // PAS_ENV).chercher(enveloppe(x[i:j]), p - 100, p + (j - i) // PAS_ENV + 100)[1]


def examiner(voix: str) -> dict[str, str]:
    """Le verdict de chaque verset qui n'a pas été retrouvé net : « même prise »
    (l'enveloppe le reconnaît au même endroit, accolé au précédent), « voisins »
    (vérifié aux phonèmes, `VOISINS`), « coupe fixée » (`COUPES`), sinon
    « douteux » ou « doublon », qui
    bloquent la voix. Gardé dans `cache/reporter/brut/<voix>/examen.json`."""
    mode = VOIX[voix][4]
    f_ex = ICI / "brut" / voix / "examen.json"
    deja = lire_json(f_ex) if f_ex.exists() else {}
    verdicts: dict[str, str] = {}
    for s in range(1, 115):
        r = lire_json(ICI / "brut" / voix / f"{s:03d}.json")
        vs = r["versets"]
        so = None
        for k, v in enumerate(vs):
            cle = f"{s}:{v['a']}"
            # deux versets sur la même coupe : un fichier d'everyayah en double, sauf si
            # l'autre est déjà borné autrement (`VOISINS`, `COUPES`)
            doublon = not v.get("vide") and any(
                abs(w["coupe"][0] - v["coupe"][0]) < 100 and not w.get("vide") and (s, w["a"]) not in hors_reference(voix)
                for w in vs[max(0, k - 1):k] + vs[k + 1:k + 2])
            if (s, v["a"]) in VOISINS.get(voix, {}):
                verdicts[cle] = "voisins"
            elif (s, v["a"]) in COUPES.get(voix, {}):
                verdicts[cle] = "coupe fixée"
            elif (s, v["a"]) in AILLEURS.get(voix, {}):
                verdicts[cle] = "ailleurs"
            elif v.get("vide"):
                verdicts[cle] = "douteux (fichier d'everyayah sans son)"
            elif doublon:
                verdicts[cle] = "doublon"
            elif v["score"] >= NET[mode]:
                continue
            elif mode == "onde":
                if cle in deja and deja[cle].startswith("même prise"):
                    verdicts[cle] = deja[cle]
                    continue
                so = decoder(telecharger(r["audio"])[0]) if so is None else so
                env = _env_au_meme_endroit(voix, s, v, so)
                # accolé au précédent : au plus 100 ms de jour (ou `JOUR_ACCOLE`), un
                # recouvrement que la lecture d'un trait enjambe sans reculer (`controler`),
                # ou une jonction posée dans le silence (`JONCTIONS`)
                accole = (k == 0 or -330 <= v["coupe"][0] - vs[k - 1]["coupe"][1] <= JOUR_ACCOLE.get(voix, 100)
                          or (s, vs[k - 1]["a"]) in JONCTIONS.get(voix, {}))
                verdicts[cle] = (f"même prise (enveloppe {env:.2f})" if env >= ENVELOPPE_MEME_PRISE and accole
                                 else f"douteux (onde {v['score']:.2f}, enveloppe {env:.2f})")
            else:
                verdicts[cle] = f"douteux (enveloppe {v['score']:.2f})"
    ecrire_json(f_ex, verdicts, compact=False)
    return verdicts


# --- Tables -----------------------------------------------------------------

def bornes(r: dict, voisins: set[int] = frozenset(),
           fixees: dict[int, tuple[int, int]] | None = None,
           jonctions: dict[int, int] | None = None) -> list[tuple[int, int]]:
    """Les bornes écrites dans la table, pour qu'un verset joué seul soit **exactement
    le fichier d'everyayah**.

    L'application (`lib/verseTiming.ts`, `windowsFromTable`) part 30 ms avant le
    début écrit (sans repasser la fin du verset précédent) et, pour un verset
    joué seul, s'arrête 150 ms après la fin écrite. On écrit donc la coupe de
    l'éditeur décalée d'autant : début + 30 ms (sans dépasser l'attaque de la
    voix), fin − 150 ms. Une reprise de l'imam qu'everyayah avait retirée reste
    hors du verset ; la lecture d'un trait, elle, enchaîne jusqu'au verset
    suivant et la joue, comme le fichier.

    `voisins` : les versets (numéros) dont la coupe est celle de leurs voisins ;
    `fixees` : ceux dont la coupe est posée à la main (`COUPES`) ;
    `jonctions` : verset → coupe entre lui et le suivant (`JONCTIONS`), qui
    remplace la fin de l'un et le début de l'autre.
    """
    vs = r["versets"]
    out = []
    for k, v in enumerate(vs):
        if fixees and v["a"] in fixees:
            c0, c1 = fixees[v["a"]]
            v0 = c0 + 30
        elif v["a"] in voisins:
            c0 = vs[k - 1]["coupe"][1] if k else 0
            c1 = vs[k + 1]["coupe"][0] if k + 1 < len(vs) else r["duree"]
            v0 = c0 + 30
        else:
            (c0, c1), v0 = v["coupe"], v["voix"][0]
        if jonctions and v["a"] in jonctions:
            c1 = jonctions[v["a"]]
        if jonctions and v["a"] - 1 in jonctions:
            c0 = jonctions[v["a"] - 1]
        c0 = max(0, c0)
        c1 = min(c1, r["duree"])  # le fichier d'everyayah peut finir sur un silence de plus
        debut = max(c0, min(c0 + 30, v0))
        out.append((debut, max(debut + 100, c1 - 150)))
    return out


def ecrire(voix: str) -> None:
    dossier_ea, dossier_mq, lecture, nom, mode = VOIX[voix]
    rec = f"reporte-{voix}"
    verdicts = examiner(voix)
    bloquants = {c: x for c, x in verdicts.items() if x.startswith(("douteux", "doublon"))}
    if bloquants:
        print(f"{rec} : pas de table, {len(bloquants)} verset(s) à résoudre :")
        for c, x in bloquants.items():
            print(f"   {c} : {x}")
        return
    for s in range(1, 115):
        r = lire_json(ICI / "brut" / voix / f"{s:03d}.json")
        b = bornes(r, {int(c.split(":")[1]) for c, x in verdicts.items() if x == "voisins" and int(c.split(":")[0]) == s},
                   {a: c for (s2, a), (c, _) in COUPES.get(voix, {}).items() if s2 == s},
                   {a: c for (s2, a), (c, _) in JONCTIONS.get(voix, {}).items() if s2 == s})
        n = comptes("hafs")[s]
        ailleurs = {a for (s2, a) in AILLEURS.get(voix, {}) if s2 == s}  # absents de la table (`AILLEURS`)
        t = table_sourate(r["audio"], r["octets"], n, {a: x for a, x in enumerate(b, start=1) if a not in ailleurs}, {})
        p = next(k for k in range(len(b)) if k + 1 not in ailleurs)  # le premier verset de la table
        c0 = max(0, r["versets"][p]["coupe"][0])
        t["ouverture"] = [0, c0] if c0 > 0 else None  # ce qui précède le premier fichier d'everyayah
        sauts = [x for x in sauts_de(r, b) if x not in ailleurs and x + 1 not in ailleurs]
        if sauts:
            t["sauts"] = sauts
        ecrire_sourate(rec, s, t)
    ecrire_fiche(rec, {
        "recitation": rec,
        "nom": nom,
        "riwaya": "hafs",
        "comptage": "kufi",
        "source": {
            "audio": f"mp3quran.net, {MP3QURAN}{dossier_mq} (lecture {lecture})",
            "reference": f"fichiers au verset de la même prise, {EVERYAYAH}{dossier_ea}/",
        },
        "niveau": "reporte",
        "production": (
            "chaque fichier au verset retrouvé dans le fichier de sourate par corrélation "
            + ("de la forme d'onde (2 kHz, affinée à 8 kHz)" if mode == "onde"
               else "de l'énergie par tranches de 5 ms (son retraité)")
            + " ; bornes = coupes de l'éditeur, début + 30 ms et fin − 150 ms (marges que l'application ajoute)"
        ),
        "examen": {c: x for c, x in verdicts.items()},
        "voisins": {f"{s}:{a}": x for (s, a), x in VOISINS.get(voix, {}).items()},
        **({"coupes": {f"{s}:{a}": {"coupe": list(c), "raison": x} for (s, a), (c, x) in COUPES[voix].items()}}
           if voix in COUPES else {}),
        **({"jonctions": {f"{s}:{a}-{a + 1}": {"coupe": c, "raison": x} for (s, a), (c, x) in JONCTIONS[voix].items()}}
           if voix in JONCTIONS else {}),
        **({"ailleurs": {f"{s}:{a}": {"audio": u, "fenetre": list(c), "raison": x}
                         for (s, a), (u, c, x) in AILLEURS[voix].items()}} if voix in AILLEURS else {}),
        "couverture": {"versets": 6236, "versets_avec_mots": 0},
    })
    print(f"{rec} : 114 sourates écrites", flush=True)


SOUFFLE_MS = 3000  # `MAX_BREATH_MS` de hooks/use-verse-audio.ts : au-delà, la lecture se déplace


def sauts_de(r: dict, b: list[tuple[int, int]]) -> list[int]:
    """Les versets après lesquels le fichier contient autre chose que le verset
    suivant (reprise de l'imam, passage copié deux fois) : plus d'un souffle entre
    la fin de leur coupe et le début de la suivante. L'application les fait sauter
    à la lecture d'un trait (`sauts`, `lib/verseTiming.ts`)."""
    return [k + 1 for k in range(len(b) - 1) if b[k + 1][0] - (b[k][1] + 150) > SOUFFLE_MS]


def fenetres_app(versets: list, ouverture, sauts=()) -> tuple[list[tuple[int, int]], list[int]]:
    """Les fenêtres que l'application tire d'une table : `windowsFromTable` et
    `voiceEndsFromTable` de `lib/verseTiming.ts`, recopiés à l'identique (sans `mots`)."""
    fin_ouv = ouverture[1] if ouverture else 0
    debuts = []
    for i, v in enumerate(versets):
        if v is None:  # verset absent de la table (lu ailleurs, `AILLEURS`) : -1, comme l'application
            debuts.append(-1)
            continue
        prec = fin_ouv if i == 0 else (versets[i - 1][1] if versets[i - 1] else None)
        debuts.append(v[0] - min(30, max(0, 30 if prec is None else v[0] - prec)))
    spans = []
    for i, v in enumerate(versets):
        if v is None:
            spans.append((-1, -1))
            continue
        suiv = debuts[i + 1] if i + 1 < len(versets) and (i + 1) not in sauts else -1
        spans.append((debuts[i], max(v[1], suiv) if suiv >= 0 else v[1] + 150))
    fins = [-1 if v is None else min(v[1] + 150, spans[i][1]) for i, v in enumerate(versets)]
    return spans, fins


def controler(voix: str) -> None:
    """Chaque verset tel que l'application le jouera, contre le fichier d'everyayah.

    - seul (Jeux, Mémorisation) : [début de fenêtre, fin de la voix] doit tomber
      sur la coupe de l'éditeur, et durer ce que dure le fichier ;
    - d'un trait : les fenêtres doivent se suivre, sans trou ni recouvrement de
      plus de 150 ms (au-delà, `continues` de hooks/use-verse-audio.ts recule).
    La durée de référence est celle du décodage du fichier d'everyayah téléchargé.

    Deux fichiers d'everyayah qui se recouvrent (versets enchaînés sans pause : la
    fin de l'un est le début de l'autre) : le premier s'arrête où le second
    commence, au plus 150 ms avant la fin de son fichier. C'est compté à part.
    """
    rec = f"reporte-{voix}"
    voisins = {f"{s}:{a}" for s, a in hors_reference(voix)}
    jonctions = JONCTIONS.get(voix, {})
    ecarts, decales, trous, recouvre, n, nsauts, joints = [], [], [], [], 0, 0, []
    for s in range(1, 115):
        t = lire_json(DONNEES / rec / f"{s:03d}.json")
        r = lire_json(ICI / "brut" / voix / f"{s:03d}.json")
        spans, fins = fenetres_app(t["versets"], t.get("ouverture"), t.get("sauts", []))
        nsauts += len(t.get("sauts", []))
        for i, (v, (d, f), fv) in enumerate(zip(r["versets"], spans, fins)):
            n += 1
            if f"{s}:{v['a']}" in voisins:
                continue
            c0, c1 = max(0, v["coupe"][0]), min(v["coupe"][1], r["duree"])
            # une jonction posée (`JONCTIONS`) remplace la coupe de l'éditeur de ce côté-là
            pose = [x for x in ((s, v["a"]), (s, v["a"] - 1)) if x in jonctions]
            if (s, v["a"]) in jonctions:
                c1 = jonctions[(s, v["a"])][0]
            if (s, v["a"] - 1) in jonctions:
                c0 = jonctions[(s, v["a"] - 1)][0]
            if pose:
                joints.append(f"{s}:{v['a']} ({(v['coupe'][1] - v['coupe'][0] - (c1 - c0)) / 1000:.2f} s de moins)")
            w = r["versets"][i + 1] if i + 1 < len(r["versets"]) else None
            # le verset s'arrête où commence la fenêtre du suivant (son amorce de 30 ms comprise)
            if (not pose and w and not w.get("vide") and f"{s}:{w['a']}" not in voisins and w["coupe"][0] < c1 - 1
                    and w["coupe"][0] - 30 <= fv <= c1 and c1 - fv <= 150 and abs(d - c0) <= 30):
                recouvre.append(f"{s}:{v['a']} ({w['coupe'][0] - c1:+d} ms)")
                continue
            if abs(d - c0) > 30 or abs(fv - c1) > 60:
                decales.append(f"{s}:{v['a']} ({d - c0:+d}/{fv - c1:+d} ms)")
            joue = fv - d
            attendu = v["duree_ea"]
            if pose:  # le fichier d'everyayah moins ce qui dépasse la jonction
                attendu = c1 - c0
            if abs(joue - attendu) > 120 and not (i == len(spans) - 1 and c1 < v["coupe"][1]):
                ecarts.append(f"{s}:{v['a']} ({attendu / 1000:.2f} → {joue / 1000:.2f} s)")
            if i + 1 < len(spans) and (i + 1) not in t.get("sauts", []) and not 0 <= f - spans[i + 1][0] <= 150:
                trous.append(f"{s}:{v['a']} ({spans[i + 1][0] - f:+d} ms)")
    print(f"{rec} : {n} versets contrôlés, dont {len(voisins)} bornés par leurs voisins ou à la main (hors comparaison) ; "
          f"{nsauts} saut(s) à la lecture d'un trait")
    for titre, l in (("fichiers d'everyayah qui se recouvrent (le verset s'arrête où commence le suivant)", recouvre),
                     ("versets bornés d'un côté par une jonction posée au milieu du silence (`JONCTIONS`)", joints),
                     ("joué seul, décalé de la coupe de l'éditeur (début/fin)", decales),
                     ("joué seul, durée différente du fichier d'everyayah (> 120 ms)", ecarts),
                     ("d'un trait, fenêtres qui ne se suivent pas", trous)):
        print(f"  {titre} : {len(l)}{' — ' + ', '.join(l[:30]) if l else ''}{' …' if len(l) > 30 else ''}")


# --- Bilan ------------------------------------------------------------------

def bilan(voix: str) -> None:
    mode = VOIX[voix][4]
    douteux, desordre, chevauche, reprises, hors = [], [], [], [], []
    scores = []
    for s in range(1, 115):
        f = ICI / "brut" / voix / f"{s:03d}.json"
        if not f.exists():
            continue
        r = lire_json(f)
        vs = r["versets"]
        for k, v in enumerate(vs):
            scores.append(v["score"])
            if v["score"] < NET[mode]:
                douteux.append(f"{s}:{v['a']} ({v['score']:.2f})")
            if v["voix"][1] > r["duree"] or v["voix"][0] < 0:
                hors.append(f"{s}:{v['a']}")
            precs = [w for w in vs[:k] if not w.get("vide")]
            if precs and not v.get("vide"):
                p = precs[-1]
                ecart = v["coupe"][0] - p["coupe"][1]
                if v["coupe"][0] <= p["coupe"][0]:
                    desordre.append(f"{s}:{v['a']}")
                elif ecart < -60:  # au-delà de deux trames MP3 de remplissage
                    chevauche.append(f"{s}:{v['a']} ({ecart} ms)")
                elif ecart > 2000:
                    reprises.append(f"{s}:{p['a']}-{v['a']} (+{ecart / 1000:.1f} s)")
    sc = np.array(scores)
    print(f"{voix} : {len(sc)} versets, score min {sc.min():.3f}, 1er centile {np.percentile(sc, 1):.3f}, "
          f"médian {np.median(sc):.3f}")
    for titre, l in (("douteux", douteux), ("dans le désordre", desordre), ("coupes qui se chevauchent", chevauche),
                     ("voix hors du fichier", hors), ("écarts de plus de 2 s entre deux coupes (reprises retirées)", reprises)):
        print(f"  {titre} : {len(l)}{' — ' + ', '.join(l[:40]) if l else ''}{' …' if len(l) > 40 else ''}")


def nettoyer(voix: str) -> None:
    dossier_ea, dossier_mq = VOIX[voix][:2]
    for d in (ICI / "everyayah" / dossier_ea, ICI / "sourates" / "audio" / dossier_mq):
        if d.exists():
            shutil.rmtree(d)
            print(f"effacé : {d}")


def _sourates(texte: str) -> list[int]:
    out: list[int] = []
    for part in texte.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(args: list[str]) -> None:
    if len(args) < 2 or args[1] not in VOIX:
        print(__doc__)
        return
    cmd, voix = args[0], args[1]
    opts = dict(zip(args[2::2], args[3::2]))
    if cmd == "relever":
        relever(voix, _sourates(opts.get("--sourates", "1-114")))
    elif cmd == "ecrire":
        ecrire(voix)
    elif cmd == "controler":
        controler(voix)
    elif cmd == "bilan":
        bilan(voix)
        for c, x in examiner(voix).items():
            print(f"  examen {c} : {x}")
    elif cmd == "nettoyer":
        nettoyer(voix)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
