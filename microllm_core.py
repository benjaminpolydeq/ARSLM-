# microllm_core.py
from typing import List, Dict
import time

class MicroLLMCore:
    """
    MicroLLM Core — version légère pour MVP/Studio demo.
    Comportement :
      - Réponses rapides basées sur règles + reformulation simple.
      - Fournit un score de confiance fictif et suggestions de follow-up.
    Objectif : prototype fonctionnel sans dépendances lourdes (torch, transformers).
    """

    def __init__(self, name: str = "MicroLLM-lite"):
        self.name = name

    def _simple_summary(self, text: str) -> str:
        # Résumé très simple : première phrase + longueur
        s = text.strip()
        first_sentence = s.split(".")[0]
        words = len(s.split())
        chars = len(s)
        return f"{first_sentence.strip()[:200]}... ({words} mots, {chars} caractères)"

    def generate_reply(self, user_text: str, history: List[Dict]) -> Dict:
        """
        Retourne un dict {text, meta} où text est la réponse, meta contient info.
        """
        # heuristiques simples
        low = user_text.lower()
        if any(g in low for g in ("bonjour", "salut", "hello", "hi")):
            reply = "Bonjour ! Comment puis-je t'aider aujourd'hui ?"
            confidence = 0.95
        elif "résumé" in low or "résumer" in low or "summary" in low:
            reply = "Voici un résumé court : " + self._simple_summary(user_text)
            confidence = 0.85
        elif "invest" in low or "investisseur" in low or "investissement" in low:
            reply = "Je peux préparer une version synthétique pour investisseurs : veux-tu un pitch court (1 paragraphe) ou long (1 page) ?"
            confidence = 0.80
        else:
            # reformulation utile + suggestion
            summary = self._simple_summary(user_text)
            reply = (
                f"MicroLLM (demo) — reformulation courte :\n\n"
                f"{summary}\n\n"
                f"Suggestion : préciser l'objectif ou demander un exemple concret pour une réponse plus ciblée."
            )
            confidence = 0.72

        # petite latence simulée (optionnel)
        time.sleep(0.15)

        return {
            "text": reply,
            "meta": {
                "model": self.name,
                "confidence": round(confidence, 2),
                "tokens_used": len(user_text.split())
            }
        }