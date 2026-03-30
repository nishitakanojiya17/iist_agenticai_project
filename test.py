from agents.llm_agent import LLMAgent

llm = LLMAgent()

docs = [
    "Computer Science exam will be held on 25 March 2026.",
    "Venue: Block A"
]

question = "When is the exam?"

result = llm.run(docs, question)

print(result)