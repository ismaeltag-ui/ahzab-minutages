"""Alignement forcé d'une récitation sur le texte connu (niveau « aligné »).

Pour ce que personne ne publie (Badr at-Turki, le mot à mot d'Ibn Humaid…).
Principe : le modèle muaalem (obadx/muaalem-model-v3_2, MIT) donne pour chaque
trame de 40 ms la probabilité de chaque phonème ; `quran-transcript` (MIT)
transcrit l'uthmani en ces mêmes phonèmes, en gardant pour chaque phonème la
lettre — donc le mot — d'où il vient. L'alignement cherche le chemin le plus
probable qui dit **tout le texte, dans l'ordre** : il ne devine jamais une borne
d'après un silence, il fait correspondre des sons au texte. Les silences ne
servent ensuite qu'à poser la frontière **à l'intérieur** de l'écart que
l'alignement a déjà laissé entre deux mots (un modèle CTC signale un phonème
sur une ou deux trames, pas sur toute sa durée).

Le fichier se traite par fenêtres d'une minute environ, le texte débordant
toujours l'audio (fin libre) : on garde les versets finis bien avant le bord,
on repart du suivant. Le coût est donc linéaire, et une sourate de deux heures
passe comme une courte.

    py -m outil.aligner essai cache/audio/humaid/001.mp3 1
    py -m outil.aligner sourate <récitation> <fichier ou url> <sourate> [--jusqua 120] [--versets 1-10]
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .commun import CACHE, DONNEES, comptes, ecrire_json, lire_json
from .format import ecrire_fiche, ecrire_sourate, table_sourate

MODELE = "obadx/muaalem-model-v3_2"
SR = 16000
TRAME = 0.04  # s par trame de sortie (pas de 10 ms, puis deux réductions par 2)
BASMALA = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
ISTIADHA = "أَعُوذُ بِٱللَّهِ مِنَ ٱلشَّيْطَانِ ٱلرَّجِيمِ"
NEG = -1e30


# --- Audio ------------------------------------------------------------------

def charger_audio(source: str | Path, debut: float = 0.0, duree: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{debut}"]
    if duree:
        cmd += ["-t", f"{duree}"]
    cmd += ["-i", str(source), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def energie_db(audio: np.ndarray, pas: int = 160) -> np.ndarray:
    """Énergie par tranche de 10 ms, en dB."""
    n = len(audio) // pas
    x = audio[: n * pas].reshape(n, pas)
    return 10 * np.log10(np.mean(x * x, axis=1) + 1e-10)


# --- Modèle -----------------------------------------------------------------

_modele = None


def modele(fils: int = 4):
    global _modele
    if _modele is None:
        import torch
        from transformers import AutoFeatureExtractor
        from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC

        torch.set_num_threads(fils)  # laisser respirer la machine
        m = Wav2Vec2BertForMultilevelCTC.from_pretrained(MODELE, dtype=torch.float32)
        m.eval()
        _modele = (m, AutoFeatureExtractor.from_pretrained(MODELE))
    return _modele


def emissions(audio: np.ndarray, cle: str, fenetre: float = 20.0, marge: float = 2.0) -> np.ndarray:
    """Log-probabilités des phonèmes, (trames, 43), mises en cache par morceau.
    Morceaux de `fenetre` s avec `marge` s de contexte de part et d'autre."""
    import torch

    dossier = CACHE / "emissions" / cle
    dossier.mkdir(parents=True, exist_ok=True)
    total = int(len(audio) / SR / TRAME)
    sortie = np.zeros((total, 43), dtype=np.float32)
    pas = int(fenetre * SR)
    for k, debut in enumerate(range(0, len(audio), pas)):
        f = dossier / f"{k:05d}.npy"
        a, b = max(0, debut - int(marge * SR)), min(len(audio), debut + pas + int(marge * SR))
        t0 = round(debut / SR / TRAME)
        t1 = min(total, round(min(len(audio), debut + pas) / SR / TRAME))
        if f.exists():
            morceau = np.load(f).astype(np.float32)
        else:
            m, fe = modele()
            with torch.no_grad():
                x = fe(audio[a:b], sampling_rate=SR, return_tensors="pt")
                sortie_m = m(**x, return_dict=False)[0]["phonemes"][0]
                lp = torch.log_softmax(sortie_m.float(), dim=-1).numpy()
            decal = round((debut - a) / SR / TRAME)
            morceau = lp[decal: decal + (t1 - t0)]
            np.save(f, morceau.astype(np.float16))
        n = min(len(morceau), t1 - t0)
        sortie[t0: t0 + n] = morceau[:n]
        if n < t1 - t0:  # dernière trame manquante : on recopie
            sortie[t0 + n: t1] = morceau[n - 1] if n else 0
    return sortie


# --- Texte ------------------------------------------------------------------

_VOCAB: dict[str, int] | None = None


def vocab() -> dict[str, int]:
    global _VOCAB
    if _VOCAB is None:
        from huggingface_hub import hf_hub_download
        import json

        with open(hf_hub_download(MODELE, "vocab.json"), encoding="utf-8") as f:
            _VOCAB = json.load(f)["phonemes"]
    return _VOCAB


def moshaf():
    from quran_transcript import MoshafAttributes

    return MoshafAttributes(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4,
                            madd_mottasel_waqf=4, madd_aared_len=4)


@dataclass
class Unite:
    """Un morceau de texte dit d'une traite puis suivi d'un arrêt : un verset,
    la basmala ou l'istiʿādha."""
    genre: str  # "verset" | "basmala" | "istiadha"
    verset: int
    mots: list[str]
    jetons: list[int] = field(default_factory=list)  # ids de phonèmes
    mot_du_jeton: list[int] = field(default_factory=list)
    lies: list[int] = field(default_factory=list)  # i : les mots i et i+1 se prononcent d'un bloc


def phonetiser(genre: str, verset: int, uthmani: str) -> Unite:
    from quran_transcript import quran_phonetizer

    o = quran_phonetizer(uthmani, moshaf(), remove_spaces=False)
    mots = uthmani.split(" ")
    mot_de_lettre = []
    for i, m in enumerate(mots):
        mot_de_lettre += [i] * len(m) + [i]  # l'espace qui suit compte avec le mot
    proprio = [-1] * len(o.phonemes)
    lies = []
    for j, mp in enumerate(o.mappings or []):
        w = mot_de_lettre[j] if j < len(mot_de_lettre) else len(mots) - 1
        if uthmani[j] == " " and mp.deleted and w + 1 < len(mots):
            lies.append(w)  # l'espace a disparu à la prononciation : idghām, iqlāb…
        if mp.deleted:
            continue
        for p in range(mp.pos[0], mp.pos[1]):
            if p < len(proprio):
                proprio[p] = w
    v = vocab()
    u = Unite(genre, verset, mots, lies=lies)
    dernier = 0
    for p, ch in enumerate(o.phonemes):
        if proprio[p] >= 0:
            dernier = proprio[p]
        if ch == " ":
            continue
        if ch not in v:
            raise ValueError(f"phonème inconnu du modèle : {ch!r} ({genre} {verset})")
        u.jetons.append(v[ch])
        u.mot_du_jeton.append(dernier)
    return u


def unites_sourate(s: int, versets: list[int], prefixe: tuple[str, ...]) -> list[Unite]:
    from quran_transcript import Aya

    res = []
    if "istiadha" in prefixe:
        res.append(phonetiser("istiadha", 0, ISTIADHA))
    if "basmala" in prefixe:
        res.append(phonetiser("basmala", 0, BASMALA))
    for a in versets:
        res.append(phonetiser("verset", a, Aya(s, a).get().uthmani))
    return res


# --- Viterbi CTC ------------------------------------------------------------

def viterbi(lp: np.ndarray, cibles: list[int], fin_libre: bool,
            mots: list[tuple[int, int]] | None = None, penalite: float = -30.0,
            recul: int = 20) -> tuple[np.ndarray, float, np.ndarray]:
    """État (0..2L) de chaque trame, score, et les trames où le chemin revient en
    arrière. États pairs = blanc.

    `mots` ([premier jeton, dernier jeton] de chaque mot) autorise les **reprises** :
    depuis la fin d'un mot, le chemin peut revenir au début de l'un des `recul`
    mots précédents (ou du même), au prix de `penalite`. Un récitant qui redit
    « wa-yubashshira l-muminin » avant de poursuivre est ainsi suivi, au lieu que
    la reprise soit avalée par le mot d'avant. La pénalité interdit d'inventer
    une reprise : un mot redit qu'on ne suivrait pas coûte des dizaines de
    trames mal appariées, bien plus que 30."""
    T, L = len(lp), len(cibles)
    S = 2 * L + 1
    ext = np.zeros(S, dtype=np.int64)
    ext[1::2] = cibles
    saut = np.zeros(S, dtype=bool)
    if L > 1:
        saut[3::2] = ext[3::2] != ext[1:-2:2]
    em = lp[:, ext]
    dp = np.full(S, NEG)
    dp[0], dp[1] = em[0, 0], em[0, 1]
    choix = np.zeros((T, S), dtype=np.int8)
    if mots:
        s_deb = np.array([2 * a + 1 for a, _ in mots])
        s_fin = np.array([2 * b + 1 for _, b in mots])
        s_blc = np.minimum(s_fin + 1, S - 1)
        nW = len(mots)
        rang = np.arange(nW)
        source = np.zeros((T, nW), dtype=np.int32)
        mot_de_debut = {int(s): j for j, s in enumerate(s_deb)}
    for t in range(1, T):
        b = np.concatenate(([NEG], dp[:-1]))
        c = np.concatenate(([NEG, NEG], dp[:-2]))
        c[~saut] = NEG
        meilleur = dp.copy()
        ch = np.zeros(S, dtype=np.int8)
        m = b > meilleur
        meilleur[m], ch[m] = b[m], 1
        m = c > meilleur
        meilleur[m], ch[m] = c[m], 2
        if mots:
            e_tok, e_blc = dp[s_fin], dp[s_blc]
            e = np.maximum(e_tok, e_blc)
            e_src = np.where(e_tok >= e_blc, s_fin, s_blc)
            best = np.full(nW, NEG)
            arg = np.zeros(nW, dtype=np.int64)
            for d in range(min(recul + 1, nW)):  # de la fin du mot j+d au début du mot j
                cand = np.full(nW, NEG)
                cand[: nW - d] = e[d:]
                mm = cand > best
                best[mm] = cand[mm]
                arg[mm] = (rang + d)[mm]
            val = best + penalite
            mm = val > meilleur[s_deb]
            if mm.any():
                cibles_s = s_deb[mm]
                meilleur[cibles_s] = val[mm]
                ch[cibles_s] = 3
                source[t, mm] = e_src[arg[mm]]
        dp = meilleur + em[t]
        choix[t] = ch
    if fin_libre:
        s = int(np.argmax(dp))
    else:
        s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    score = float(dp[s])
    etats = np.zeros(T, dtype=np.int64)
    retours = np.zeros(T, dtype=bool)
    for t in range(T - 1, -1, -1):
        etats[t] = s
        c = int(choix[t, s])
        if c == 3:
            retours[t] = True
            s = int(source[t, mot_de_debut[s]])
        else:
            s -= c
    return etats, score, retours


# --- Alignement par fenêtres -------------------------------------------------

@dataclass
class Mot:
    """Une occurrence d'un mot dans l'enregistrement (un mot redit en a deux)."""
    unite: int
    mot: int
    debut: int  # première trame du premier phonème
    fin: int  # dernière trame du dernier phonème, incluse
    score: float


def _occurrences(unites: list[Unite], idx: list[int], etats: np.ndarray, retours: np.ndarray,
                 t0: int, lp: np.ndarray) -> list[Mot]:
    """Les occurrences de mots dans l'ordre du temps. `lp` : les trames de la fenêtre."""
    carte = []
    for ui in idx:
        for j, w in enumerate(unites[ui].mot_du_jeton):
            carte.append((ui, w, unites[ui].jetons[j]))
    res: list[Mot] = []
    scores: list[list[float]] = []
    for t, s in enumerate(etats):
        if s % 2 == 0:
            continue
        ui, w, jeton = carte[s // 2]
        if res and (res[-1].unite, res[-1].mot) == (ui, w) and not retours[t]:
            res[-1].fin = t0 + t
            scores[-1].append(float(lp[t, jeton]))
        else:
            res.append(Mot(ui, w, t0 + t, t0 + t, 0.0))
            scores.append([float(lp[t, jeton])])
    for m, sc in zip(res, scores):
        m.score = float(np.mean(sc))
    return res


def _bornes_mots(unites: list[Unite], idx: list[int]) -> list[tuple[int, int]]:
    res, base = [], 0
    for ui in idx:
        u = unites[ui]
        for w in range(len(u.mots)):
            pos = [base + j for j, x in enumerate(u.mot_du_jeton) if x == w]
            if pos:
                res.append((pos[0], pos[-1]))
        base += len(u.jetons)
    return res


def aligner(lp: np.ndarray, unites: list[Unite], fenetre_s: float = 60.0, fin_ouverte: bool = False,
            journal=print) -> list[Mot]:
    """Toutes les occurrences, dans l'ordre du temps.
    `fin_ouverte` : l'audio s'arrête avant la fin du texte (extrait) ; on ne
    garde alors que les unités finies avant le bord."""
    T = len(lp)
    W = int(fenetre_s / TRAME)
    rythme = 2.2  # trames par phonème, réestimé après chaque fenêtre
    pos, ui, res = 0, 0, []
    while ui < len(unites):
        reste_audio = T - pos
        # texte : assez pour déborder 1,6 fois la fenêtre audio
        idx, n_jetons = [], 0
        while ui + len(idx) < len(unites) and (n_jetons * rythme < 1.6 * min(W, reste_audio) or len(idx) < 2):
            idx.append(ui + len(idx))
            n_jetons += len(unites[idx[-1]].jetons)
        derniere = idx[-1] == len(unites) - 1
        finale = derniere or W >= reste_audio
        if finale and not fin_ouverte:
            # tout le texte restant tient dans ce qui reste d'audio : dernière passe
            idx = list(range(ui, len(unites)))
            fen, fin_libre = reste_audio, False
        else:
            fen, fin_libre = min(W, reste_audio), True
        cibles = [j for i in idx for j in unites[i].jetons]
        fenetre = lp[pos: pos + fen]
        etats, _, retours = viterbi(fenetre, cibles, fin_libre, _bornes_mots(unites, idx))
        occ = _occurrences(unites, idx, etats, retours, pos, fenetre)
        n_rep = int(retours.sum())
        note = f", {n_rep} reprise(s)" if n_rep else ""
        if not fin_libre:
            res += occ
            journal(f"  fenêtre {pos * TRAME:7.1f}-{(pos + fen) * TRAME:7.1f} s : unités {idx[0]}..{idx[-1]} (fin){note}")
            break
        # garder les unités finies avant le bord (3 s) et qui ne sont pas la dernière du texte
        limite = pos + fen - int(3.0 / TRAME)
        gardees = []
        for i in (idx if finale else idx[:-1]):
            oi = [m for m in occ if m.unite == i]
            if oi and len({m.mot for m in oi}) == len(set(unites[i].mot_du_jeton)) and oi[-1].fin < limite:
                gardees.append(i)
            else:
                break
        if finale and fin_ouverte:
            res += [m for m in occ if m.unite in gardees]
            journal(f"  fenêtre {pos * TRAME:7.1f}-{(pos + fen) * TRAME:7.1f} s : fin de l'extrait, "
                    f"{len(gardees)} unités gardées{note}")
            break
        if not gardees:
            W = min(int(W * 1.5), reste_audio)
            journal(f"  fenêtre trop courte à {pos * TRAME:.1f} s, élargie à {W * TRAME:.0f} s")
            continue
        garde = [m for m in occ if m.unite in gardees]
        # ce qui suit la dernière occurrence gardée appartient à la fenêtre suivante
        dern = garde[-1].fin
        premier_suivant = next((m.debut for m in occ if m.debut > dern and m.unite not in gardees), dern + 1)
        res += garde
        jetons_g = sum(len(unites[i].jetons) for i in gardees)
        rythme = max(1.2, (dern - pos) / max(1, jetons_g))
        journal(f"  fenêtre {pos * TRAME:7.1f}-{(pos + fen) * TRAME:7.1f} s : unités {gardees[0]}..{gardees[-1]} gardées{note}")
        pos = (dern + premier_suivant) // 2 + 1
        ui = gardees[-1] + 1
        W = int(fenetre_s / TRAME)
    return res


def choisir_prefixe(lp: np.ndarray, s: int, premiers: list[int]) -> tuple[str, ...]:
    """Istiʿādha et basmala sont dites ou non selon l'enregistrement : on essaie
    chaque ouverture sur la première minute et l'on garde la plus probable."""
    candidats = [(), ("istiadha",)] if s in (1, 9) else [(), ("basmala",), ("istiadha", "basmala")]
    if s == 9:
        candidats = [(), ("istiadha",)]
    fen = lp[: int(60 / TRAME)]
    meilleur, score_m = candidats[0], NEG
    for c in candidats:
        u = unites_sourate(s, premiers, c)
        _, sc, _ = viterbi(fen, [j for x in u for j in x.jetons], True)
        if sc > score_m:
            meilleur, score_m = c, sc
    return meilleur


# --- Frontières fines ---------------------------------------------------------

def affiner(mots: list[Mot], db: np.ndarray) -> list[tuple[int, int]]:
    """Bornes en ms. Entre deux mots, l'alignement laisse un écart (fin du dernier
    phonème signalé → début du premier phonème suivant) : s'il contient un
    silence d'au moins 80 ms, le premier mot finit où le silence commence et le
    suivant commence où il finit ; sinon la frontière est posée au creux
    d'énergie de l'écart. Les silences ne décident jamais d'une frontière hors
    de cet écart.

    Avant un arrêt, la fin du mot suit la décroissance jusqu'à ce qu'elle rejoigne
    le bruit de fond (seuil bas), sans dépasser la pause : couper dès que la voix
    tombe tronquait le « -ā » final des fins de versets, surtout dans une mosquée
    où la réverbération prolonge chaque mot (Ibn Humaid, An-Naba, 2026-09-25)."""
    plancher = np.percentile(db, 5)
    etendue = np.percentile(db, 95) - plancher
    seuil = plancher + 0.35 * etendue
    seuil_bas = plancher + 0.15 * etendue
    silence = db < seuil
    ms = lambda trame: int(round(trame * TRAME * 1000))
    bornes = [[ms(m.debut), ms(m.fin + 1)] for m in mots]

    def cherche(a_ms: int, b_ms: int):
        a, b = max(0, a_ms // 10), min(len(db), b_ms // 10)
        if b <= a:
            return None, (a + b) // 2 * 10
        seg = silence[a:b]
        meilleur, cur, deb_m = (0, 0), 0, 0
        for i, x in enumerate(seg):
            cur = cur + 1 if x else 0
            if cur > meilleur[0]:
                meilleur = (cur, i - cur + 1)
        if meilleur[0] >= 8:
            return (a + meilleur[1]) * 10, (a + meilleur[1] + meilleur[0]) * 10
        return None, (a + int(np.argmin(db[a:b]))) * 10

    for k in range(len(mots) - 1):
        a, b = bornes[k][1] - 40, bornes[k + 1][0] + 40
        deb_sil, fin = cherche(a, b)
        if deb_sil is not None:
            i, j = deb_sil // 10, max(deb_sil // 10, fin // 10 - 5)
            while i < j and db[i] >= seuil_bas:
                i += 1
            bornes[k][1], bornes[k + 1][0] = i * 10, fin
        else:
            bornes[k][1] = bornes[k + 1][0] = fin
    # tout début et toute fin : on étend jusqu'au silence voisin (au plus 400 ms)
    if bornes:
        i = bornes[0][0] // 10
        j = max(0, i - 40)
        while i > j and not silence[i - 1]:
            i -= 1
        bornes[0][0] = i * 10
        i = min(len(db), bornes[-1][1] // 10)
        j = min(len(db), i + 40)
        while i < j and db[i] >= seuil_bas:
            i += 1
        bornes[-1][1] = i * 10
    return [tuple(b) for b in bornes]


# --- Commandes ---------------------------------------------------------------

def aligner_sourate(rec: str, source: str, s: int, audio_url: str | None = None,
                    jusqua: float | None = None, versets: tuple[int, int] | None = None,
                    nom: str | None = None) -> dict:
    t_debut = time.time()
    audio = charger_audio(source, 0, jusqua)
    cle = hashlib.sha1(f"{source}|{jusqua}".encode()).hexdigest()[:16]
    lp = emissions(audio, cle)
    t_em = time.time() - t_debut
    n = len(comptes("hafs")[s])
    a1, a2 = versets or (1, n)
    liste = list(range(a1, (n if jusqua else a2) + 1))  # extrait : le texte déborde l'audio
    prefixe = choisir_prefixe(lp, s, liste[:3]) if a1 == 1 else ()
    unites = unites_sourate(s, liste, prefixe)
    occ = aligner(lp, unites, fin_ouverte=jusqua is not None)
    occ = [m for m in occ if unites[m.unite].genre != "verset" or unites[m.unite].verset <= a2]
    bornes = affiner(occ, energie_db(audio))
    # Un mot redit garde sa **dernière** occurrence, celle qui mène à la suite ;
    # le verset, lui, couvre tout ce qui a été dit de lui, reprises comprises.
    derniere: dict[tuple[int, int], int] = {}
    for k, m in enumerate(occ):
        derniere[(m.unite, m.mot)] = k
    par_verset: dict[int, dict[int, tuple[int, int]]] = {}
    etendue: dict[int, list[int]] = {}
    reprises: dict[int, list[list[int]]] = {}
    ouverture = None
    scores: dict[int, list[float]] = {}
    for k, (m, (d, f)) in enumerate(zip(occ, bornes)):
        u = unites[m.unite]
        if u.genre != "verset":
            ouverture = [0 if ouverture is None else ouverture[0], f]
            continue
        e = etendue.setdefault(u.verset, [d, f])
        e[0], e[1] = min(e[0], d), max(e[1], f)
        if derniere[(m.unite, m.mot)] != k:
            r = reprises.setdefault(u.verset, [])
            if r and r[-1][1] >= d - 50:
                r[-1][1] = f
            else:
                r.append([d, f])
            continue
        par_verset.setdefault(u.verset, {})[m.mot + 1] = (d, f)
        scores.setdefault(u.verset, []).append(m.score)
    vbornes = {a: tuple(e) for a, e in etendue.items()}
    mots = [m for k, m in enumerate(occ) if derniere[(m.unite, m.mot)] == k]
    table = table_sourate(audio_url or str(source), None, comptes("hafs")[s], vbornes, par_verset)
    table["ouverture"] = ouverture
    table["ouverture_dite"] = list(prefixe)
    table["lies"] = {str(u.verset): u.lies for u in unites if u.genre == "verset" and u.lies}
    table["confiance"] = {str(a): round(float(np.mean(v)), 3) for a, v in scores.items()}
    table["reprises"] = {str(a): r for a, r in reprises.items()}
    ecrire_sourate(rec, s, table)
    duree = len(audio) / SR
    fiche_f = DONNEES / rec / "recitation.json"
    fiche = lire_json(fiche_f) if fiche_f.exists() else {
        "recitation": rec, "nom": nom or rec, "riwaya": "hafs", "comptage": "kufi",
        "source": {"nom": "alignement forcé ahzab-minutages", "modele": MODELE,
                   "texte": "quran-transcript (uthmani, Hafs, madd 4)"},
        "niveau": "aligne", "production": "alignement CTC sur le texte, frontières affinées dans l'écart", "passes": {}}
    fiche["passes"][f"{s:03d}"] = {"versets": [a1, a2], "audio_s": round(duree, 1),
                                   "calcul_s": round(time.time() - t_debut, 1), "emissions_s": round(t_em, 1),
                                   "ouverture": list(prefixe)}
    if t_em > 20:  # émissions calculées, pas relues du cache : une vraie mesure de vitesse
        fiche.setdefault("vitesse", {})[f"{s:03d}"] = {"audio_s": round(duree, 1), "calcul_s": round(time.time() - t_debut, 1)}
    ecrire_fiche(rec, fiche)
    print(f"{rec} {s:03d} : {len(par_verset)} versets, {len(mots)} mots ; ouverture {prefixe or 'aucune'} ; "
          f"audio {duree:.0f} s, calcul {time.time() - t_debut:.0f} s", flush=True)
    return table


def main(args: list[str]) -> None:
    if not args:
        print(__doc__)
        return
    if args[0] == "sourate":
        rec, source, s = args[1], args[2], int(args[3])
        opts = dict(zip(args[4::2], args[5::2]))
        v = opts.get("--versets")
        aligner_sourate(rec, source, s, audio_url=opts.get("--url"),
                        jusqua=float(opts["--jusqua"]) if "--jusqua" in opts else None,
                        versets=tuple(int(x) for x in v.split("-")) if v else None,
                        nom=opts.get("--nom"))


if __name__ == "__main__":
    main(sys.argv[1:])
