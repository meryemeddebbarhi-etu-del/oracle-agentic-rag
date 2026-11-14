#!/usr/bin/env python
"""Point d'entrée en ligne de commande du module Agent IA (Membre 3).

Exemples :
    python main.py
    python main.py --method hybrid
    python main.py --backend mock --method auto
    python main.py --question "Comment fonctionne DHCP ?" --json

Compatible Windows (aucune dépendance à un shell Unix).
"""

from __future__ import annotations

import argparse
import json
import sys

from agent import RAGAgent


def _afficher_reponse(reponse: dict, en_json: bool) -> None:
    if en_json:
        print(json.dumps(reponse, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 70)
    print(f"Question : {reponse['question']}")
    print("-" * 70)
    if reponse.get("error"):
        print(f"[Erreur] {reponse['error']}")
    else:
        print(reponse["answer"])
    if reponse.get("ambiguous"):
        print("\n(Remarque : question potentiellement ambiguë.)")
    if reponse.get("sources"):
        print("\nSources :")
        for s in reponse["sources"]:
            print(f"  - {s['source']} (méthode : {s['search_method']}, score : {s['score']})")
    print(f"\nMéthode de recherche utilisée : {reponse['search_method']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent IA — Oracle Agentic RAG (Membre 3)")
    parser.add_argument("--backend", choices=["mock", "real"], default=None, help="Moteur de recherche à utiliser (défaut : .env SEARCH_BACKEND, 'mock' par défaut).")
    parser.add_argument("--method", choices=["auto", "vector", "keyword", "hybrid"], default=None, help="Méthode de recherche (défaut : .env DEFAULT_SEARCH_METHOD, 'auto' par défaut).")
    parser.add_argument("--top-k", type=int, default=None, help="Nombre de résultats par recherche.")
    parser.add_argument("--question", type=str, default=None, help="Pose une seule question et quitte (sinon, mode interactif).")
    parser.add_argument("--json", action="store_true", help="Affiche la réponse au format JSON brut.")
    args = parser.parse_args()

    agent = RAGAgent(backend=args.backend, top_k=args.top_k)

    if args.question:
        reponse = agent.ask(args.question, method=args.method, top_k=args.top_k)
        _afficher_reponse(reponse, args.json)
        sys.exit(0)

    print("Agent IA — Oracle Agentic RAG (Membre 3)")
    print("Tapez votre question, ou 'quit' pour quitter.\n")
    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break
        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir.")
            break
        if not question:
            continue
        reponse = agent.ask(question, method=args.method, top_k=args.top_k)
        _afficher_reponse(reponse, args.json)


if __name__ == "__main__":
    main()
