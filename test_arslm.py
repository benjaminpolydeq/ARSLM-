# test_arslm.py
from arslm.arslm import ARSLM
from datetime import datetime

def main():
    print("🧠 Test ARSLMCore - Adaptive Reasoning Semantic Language Model\n")

    # Initialiser le modèle
    print("Initialisation du modèle ARSLM...")
    model = ARSLM(use_custom_model=True, device="cpu")  # ou "cuda" si tu as GPU

    # Test 1 : génération simple
    prompt = "Bonjour, explique-moi l'intelligence artificielle simplement."
    print("\n=== Test 1 : Génération simple ===")
    response = model.generate(prompt, max_length=50)
    print(f"Prompt : {prompt}")
    print(f"Réponse : {response}")

    # Vérifier l'historique
    print("\nHistorique après Test 1 :")
    for exchange in model.get_history():
        print(f"- User: {exchange['user']}")
        print(f"  Assistant: {exchange['assistant']}")

    # Test 2 : multi-tours
    print("\n=== Test 2 : Multi-tours ===")
    multi_prompts = [
        "Salut, peux-tu me donner un exemple d'application de l'IA ?",
        "Quels sont les avantages pour les petites entreprises ?",
        "Et les risques potentiels ?"
    ]

    for i, msg in enumerate(multi_prompts):
        print(f"\nTour {i+1} - User: {msg}")
        resp = model.generate(msg, max_length=50)
        print(f"Assistant: {resp}")

    # Historique final
    print("\nHistorique final :")
    for exchange in model.get_history():
        timestamp = exchange.get('timestamp', 'N/A')
        print(f"[{timestamp}] User: {exchange['user']}")
        print(f"Assistant: {exchange['assistant']}\n")

if __name__ == "__main__":
    main()