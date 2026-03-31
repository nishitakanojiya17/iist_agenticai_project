"""
llm_agent.py — Thin wrapper around Ollama (Llama 3).

This file is the integration point.  If you already have your own llm_agent.py,
replace the body of ask_llm() with your existing implementation — just make sure
the function signature stays the same:

    ask_llm(context: str, question: str) -> str

query.py imports this automatically.
"""

import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"          # change to e.g. "llama3:8b" if needed

SYSTEM_PROMPT = (
    "You are a precise and helpful assistant. "
    "Answer questions ONLY using the provided context. "
    "If the answer cannot be found in the context, say: "
    "'I could not find relevant information in the document.' "
    "Be concise and factual."
)


def ask_llm(context: str, question: str) -> str:
    """
    Send context + question to Ollama and return the answer string.

    Parameters
    ----------
    context  : Retrieved document chunks joined as a single string.
    question : The user's original question.

    Returns
    -------
    str : The model's answer.
    """
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "---\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,   # low temp → factual, less hallucination
                    "num_predict": 512,   # max tokens in answer
                },
            },
            timeout=120,  # seconds — increase for slow machines
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot reach Ollama. Is it running?  Try:  ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "Ollama took too long to respond. "
            "Try a smaller model or increase the timeout."
        )
