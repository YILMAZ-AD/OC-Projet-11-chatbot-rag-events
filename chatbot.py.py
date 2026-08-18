"""
Script du Chatbot RAG (Puls-Events).
Ce script charge l'index vectoriel FAISS existant et utilise l'API Mistral AI 
pour répondre aux questions de l'utilisateur en se basant uniquement sur les 
événements culturels de Brest trouvés dans la base de données.
"""

import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Chargement des variables d'environnement (clé API Mistral)
load_dotenv()

# 1. Chargement de la base vectorielle construite à l'étape 3
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Nécessaire et sécurisé pour recharger un index local qu'on a créé
)


# 2. Fonctions de recherche et de formatage
def search_events(query, k=5, fetch_k=15):
    """
    Recherche les événements les plus pertinents dans la base vectorielle
    en éliminant les doublons (chunks du même événement).
    
    Arguments :
        query (str) : La question de l'utilisateur.
        k (int) : Le nombre de résultats uniques à conserver finaux.
        fetch_k (int) : Le nombre de résultats bruts à extraire avant le filtrage.
        
    Retourne :
        unique_results (list) : Une liste de documents LangChain uniques.
    """
    results = vector_store.similarity_search(query, k=fetch_k)
    seen_uids = set()
    unique_results = []
    for doc in results:
        uid = doc.metadata.get("uid")
        if uid not in seen_uids:
            seen_uids.add(uid)
            unique_results.append(doc)
        if len(unique_results) == k:
            break
    return unique_results


def format_events(docs):
    """
    Formate les documents trouvés en une seule chaîne de caractères 
    pour les injecter proprement dans le contexte du LLM.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# 3. Le prompt : Instructions strictes de l'IA + contexte + question
prompt = ChatPromptTemplate.from_template(
    """Tu es l'assistant de recommandation d'événements culturels de Puls-Events.
Réponds à la question UNIQUEMENT à partir des événements listés ci-dessous.

Règles de comportement :
- Si aucun événement ne correspond à la demande, dis-le clairement plutôt que d'inventer une réponse.
- Sois concis, poli et précise toujours le lieu et la date de chaque événement recommandé.
- Si l'utilisateur pose une question hors sujet ou sur une autre ville (comme Paris), aiguille-le poliment vers le bon fonctionnement du bot en utilisant par exemple cette phrase : « Je ne suis spécialisé que dans les événements concernant Brest. N'hésitez pas à me dire si vous en cherchez un. »

Événements disponibles :
{context}

Question de l'utilisateur : {question}

Réponse :"""
)

# 4. Le LLM Mistral : Configuration du modèle
# Utilisation de mistral-small-latest et d'une température basse (0.2) pour garantir la factualité (pas d'hallucination)
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)


# 5. La chaîne complète (LangChain Expression Language - LCEL)
rag_chain = (
    {
        "context": RunnableLambda(lambda q: format_events(search_events(q))),
        "question": RunnableLambda(lambda q: q),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# 6. Boucle de test interactive (Interface utilisateur)
def start_chat():
    """
    Lance l'interface de discussion interactive dans le terminal.
    Conforme au cahier des charges du POC (pas d'historique de conversation conservé).
    """
    print("\n👋 Bienvenue ! Je suis votre assistant Puls-Events (spécial Brest) 😁")
    print("Comment puis-je vous aider aujourd'hui ?")
    print("— Tapez 'quit' ou 'exit' pour arrêter le chatbot —\n")
    
    while True:
        question = input("Toi : ")
        if question.lower() in ("quit", "exit"):
            print("\n🤖 Au revoir et à bientôt sur Puls-Events !\n")
            break
        
        reponse = rag_chain.invoke(question)
        print(f"\n🤖 {reponse}\n")


if __name__ == "__main__":
    start_chat()