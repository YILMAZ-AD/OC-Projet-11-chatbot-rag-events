🇬🇧 English | readme en français 👉 [🇫🇷 Français](README.md)

# Chatbot Assistant with Integrated RAG for Cultural Event Recommendations (Brest)

## Project Context

Finding something to do by searching online is often a hassle. You have to dig through event calendars full of stuff that doesn't even interest you.

**Puls-Events** is a chatbot designed to make it easier to access culture in the city of Brest. The idea is simple: instead of searching a website yourself, you chat directly with the bot. You tell it what you want, and it gives you precise recommendations.

## Technical Goal (Proof of Concept)

The problem with typical AI is that it sometimes makes up answers when it doesn't actually know (what's called "hallucinations").

To avoid this, this project uses a technique called **RAG (Retrieval-Augmented Generation)**. Basically, the system works like this:
1. The code fetches real, up-to-date events from the public **OpenAgenda** API.
2. It feeds this info to the AI (**Mistral AI**) with a strict instruction: it's only allowed to answer using these events, without inventing anything else.

## Project Structure

The pipeline is split into 3 main scripts, to be run in this order:

1. **`data_extract.py`**: Connects to the OpenAgenda API, filters Brest events from the last year, cleans the data, and saves everything to `evenements_brest.json`.
2. **`vectorize_data.py`**: Splits event descriptions into chunks, turns them into mathematical vectors using Mistral, and builds a local database in the `faiss_index/` folder.
3. **`chatbot.py`**: The final interface. This script connects the database and the Mistral model to chat with the user and provide precise recommendations.

## How to Run the Project

### 1. Set Up the Environment

Clone this repo, then create and activate a Python virtual environment:

```bash
python -m venv env_rag
# On Windows: env_rag\Scripts\activate
# On Mac/Linux: source env_rag/bin/activate
```

### 2. Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Configure the API Key

Create a `.env` file at the root of the project and add your Mistral AI key:

```plaintext
MISTRAL_API_KEY=your_secret_key_here
```

### 4. Run the Pipeline

Run the scripts one by one:

```bash
python data_extract.py
python vectorize_data.py
python chatbot.py
```

## Test and Evaluate the Bot

Once the project is running, you can check that everything works well and that the bot is reliable:

- **Run a quick test:** You can run the `test_pipeline.py` script to do a test run and see how the data flows through the system end-to-end.

- **Score the bot with Ragas:** I also set up an automatic evaluation system with the `evaluate_ragas.py` script. Basically, the **Ragas** tool acts like a teacher grading the bot's exam. It checks whether the bot made anything up (*Faithfulness*) and whether its answer actually matches the question asked (*Answer Relevancy*). At the end, it outputs a small results table to prove the bot isn't talking nonsense.

## Tech Stack

Here are the technologies and libraries used to build this project:

- **Python**: The main language of the project.
- **LangChain**: Framework used to orchestrate the RAG pipeline (text splitting, prompt management, and connecting the AI to the database).
- **FAISS**: Local vector database (developed by Meta) enabling ultra-fast similarity search.
- **Mistral AI (API)**: Uses the `mistral-embed` model for vectorization and `mistral-large-latest` for text generation (LLM).
- **Pandas & Requests**: Querying the OpenAgenda API, cleaning and handling tabular data (pre-processing).
- **python-dotenv**: Keeping API keys secure and out of the source code.

## Author

**YAD95**
Project completed as part of the Data Engineer training program — OpenClassrooms
