"""Préparer les tables que l'application lit directement, par jsDelivr.

L'application (`lib/verseTiming.ts` de CoranRevise) télécharge une sourate à la
fois à l'adresse

    https://cdn.jsdelivr.net/gh/ismaeltag-ui/ahzab-minutages@<commit>/donnees/<table>/<sss>.json

et ne s'en sert que si la **taille du MP3** qu'elle mesure (requête HEAD) est
celle que la table a notée dans `octets` : un minutage ne veut rien dire hors du
fichier qui l'a produit. Les tables de Quranic Universal Audio arrivent sans
cette taille ; ce module la relève et l'écrit, pour chaque table listée dans
`APPLICATION`.

Les tables de tiers sont ignorées par git (voir `.gitignore`) : celles que
l'application lit y sont réintégrées une à une, avec leur licence dans
`recitation.json` (CC BY 4.0 pour QUA — attribution à Wider Community).

    py -m outil.publier            # relève les tailles manquantes
    py -m outil.publier --verifier # signale les tables dont une taille a changé
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

from .commun import DONNEES, ecrire_json, lire_json, taille_distante

# Les tables que l'application lit, et pourquoi.
APPLICATION = {
    # Hafs : lecture continue de voix publiées au verset (même prise vérifiée par
    # empreinte, donnees/continuite.json).
    "qua-mahmoud_ali_al_banna_qdc": "Al-Banna, lecture continue",
    "qua-mohammed_siddiq_al_minshawi_mujawwad_mp3quran": "Al-Minshawi mujawwad, lecture continue",
    "aligne-jibreel": "Muhammad Jibreel, lecture continue",
    "aligne-alili": "Aziz Alili, lecture continue",
    "aligne-husary-muallim": "Al-Husary muʿallim, lecture continue",
    # Hafs : voix nouvelles.
    "aligne-badr": "Badr at-Turki",
    "qua-mustafa_ismail_mp3quran": "Mustafa Ismail, murattal",
    # Warsh.
    "qua-mohammed_saayed_warsh_mp3quran": "Mohammad Saayed (remplace la table mp3quran 134, d'un autre enregistrement)",
    "qua-abdulbasit_abdulsamad_warsh_qdc": "AbdelBasset en Warsh, lecture continue",
    "qua-saber_abdulhakam_warsh_way2quran": "Saber Abdulhakam en Warsh",
    # Qālūn.
    "qua-ahmed_deban_qalon_mp3quran": "Ahmed Deban",
    "qua-walid_al_naihi_qalon_mp3quran": "Walid an-Naihi",
    "qua-saber_abdulhakam_qalon_way2quran": "Saber Abdulhakam en Qālūn",
    # Shuʿba.
    "qua-abdur_rashid_sufi_shubah_qdc": "Abdur-Rashid Sufi",
    # Saber Abdulhakam en Shuʿba attend : son serveur nomme 70 sourates « 97.mp3 »
    # et les autres « 097.mp3 », quand l'application ne fabrique que la seconde
    # forme. Sa table reste versionnée, prête pour le jour où elle lira `audio`.
}


def _relever(rec: str, s: int, verifier: bool) -> str | None:
    f = DONNEES / rec / f"{s:03d}.json"
    if not f.exists():
        return f"{rec} {s:03d} : table absente"
    t = lire_json(f)
    if t.get("octets") and not verifier:
        return None
    taille = taille_distante(t["audio"])
    if taille is None:
        return f"{rec} {s:03d} : taille illisible ({t['audio']})"
    if t.get("octets") and t["octets"] != taille:
        return f"{rec} {s:03d} : taille changée ({t['octets']} → {taille})"
    if not t.get("octets"):
        t["octets"] = taille
        ecrire_json(f, t)
    return None


def main(args: list[str]) -> None:
    verifier = "--verifier" in args
    recs = [a for a in args if not a.startswith("--")] or list(APPLICATION)
    for rec in recs:
        if not (DONNEES / rec).exists():
            print(f"{rec} : pas encore de données", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=8) as ex:
            notes = [n for n in ex.map(lambda s: _relever(rec, s, verifier), range(1, 115)) if n]
        print(f"{rec} : {'rien à signaler' if not notes else f'{len(notes)} problème(s)'}", flush=True)
        for n in notes[:10]:
            print("   ", n, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
