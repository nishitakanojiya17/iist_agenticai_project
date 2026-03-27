# LLM Agent (Ollama)

## Install

pip install -r requirements.txt

Install Ollama:
https://ollama.com

Pull model:
ollama pull llama3.1:8b

## Run test

python test_llm.py

## Usage

from agents.llm_agent import LLMAgent

llm = LLMAgent()
response = llm.run(docs, question)