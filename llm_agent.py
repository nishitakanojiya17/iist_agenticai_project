import ollama

class LLMAgent:
    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def run(self, docs, question):

        context = "\n\n".join(docs)

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the given context.
If answer not found say "Not found".

Context:
{context}

Question:
{question}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        return {
            "answer": answer,
            "source": "Chunk number used: Chunk 1",
            "found": True
        }