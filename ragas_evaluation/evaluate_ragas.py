"""
Script d'évaluation Ragas du système RAG (Puls-Events).

Charge le jeu de test annoté par un humain (test_ragas_dataset.json),
fait tourner le VRAI pipeline RAG (chatbot.py) sur chaque question pour
récupérer la réponse réelle et le contexte réellement retrouvé dans FAISS,
puis calcule les métriques Ragas (faithfulness, answer_relevancy,
context_precision, context_recall) en comparant tout ça aux réponses
annotées (ground_truth).

Prérequis :
    pip install ragas datasets nest_asyncio

Utilisation :
    python evaluate_ragas.py
"""
from ragas.run_config import RunConfig
import json
import pandas as pd
import nest_asyncio
from dotenv import load_dotenv
from datasets import Dataset

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

# On réutilise directement les vraies briques du chatbot, pas une version
# simplifiée : c'est ça qui rend l'évaluation représentative du système réel.
from chatbot import search_events, rag_chain

load_dotenv()
nest_asyncio.apply()

TEST_FILE = "test_ragas_dataset.json"
RESULTS_FILE = "ragas_evaluation_results.csv"


def build_evaluation_dataset(test_file: str = TEST_FILE) -> Dataset:
    """
    Charge le jeu de test annoté, puis exécute le VRAI pipeline RAG
    (recherche FAISS + génération Mistral) sur chaque question.

    Pour chaque question, on récupère séparément :
      - les contextes réellement retrouvés par search_events (la liste
        des textes de chunks, nécessaire à Ragas pour juger le retrieval) ;
      - la réponse réellement générée par rag_chain (nécessaire à Ragas
        pour juger la génération).

    Retourne un objet Dataset prêt pour ragas.evaluate().
    """
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for case in test_cases:
        question = case["question"]
        ground_truth = case["ground_truth"]

        if ground_truth.startswith("__A_COMPLETER__"):
            print(f"  Question {case.get('id')} ignorée : ground_truth pas encore annoté par un humain.")
            continue

        print(f"→ [{case.get('difficulte', '?')}] {question}")

        # 1. Récupération : les mêmes chunks que ceux utilisés par le chatbot
        docs = search_events(question)
        contexts = [doc.page_content for doc in docs]

        # 2. Génération : la vraie réponse du chatbot, via la chaîne complète
        answer = rag_chain.invoke(question)

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

    print(f"\n {len(questions)} questions annotées traitées sur {len(test_cases)} au total.\n")

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def main():
    print(" Construction du jeu d'évaluation à partir du chatbot réel...\n")

    evaluation_dataset = build_evaluation_dataset()

    if len(evaluation_dataset) == 0:
        print(
            " Aucune question annotée trouvée. "
            f"Complète les champs ground_truth dans {TEST_FILE}."
        )
        return

    print(" Initialisation du juge Ragas avec Mistral...")

    mistral_llm = ChatMistralAI(
        model="mistral-medium-latest",
        temperature=0.0,
        max_retries=2,
        timeout=120,
    )

    mistral_embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    metrics = [
        faithfulness,
        context_precision,
        context_recall
    ]

    print("Métriques calculées : faithfulness, answer_relevancy, context_precision, context_recall\n")

    run_config = RunConfig(
        max_workers=1,
        timeout=300,
        max_retries=5,
        max_wait=60,
    )   
    

    print("🧪 Lancement de l'évaluation Ragas...\n")

    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=mistral_llm,
        embeddings=mistral_embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    results_df = results.to_pandas()

    if "question" not in results_df.columns:
        results_df.insert(
            0,
            "question",
            evaluation_dataset["question"]
        )

    expected_columns = [
        "question",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    for column in expected_columns:
        if column not in results_df.columns:
            results_df[column] = None

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 160)

    print("\n--- Résultats détaillés par question ---")
    print(
        results_df[expected_columns]
        .to_string(index=False)
    )

    score_columns = [
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    print("\n--- Scores moyens ---")
    print(
        results_df[score_columns]
        .mean(numeric_only=True)
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\n Résultats complets sauvegardés dans {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
