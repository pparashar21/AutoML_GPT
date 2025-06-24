import pandas as pd
import numpy as np
import os
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv
from src.utils import *
load_dotenv(find_dotenv())
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.utils import *

import warnings
warnings.filterwarnings("ignore")

client = QdrantClient(":memory:")
OPENAI_API_KEY: str | None = os.getenv("OPEN_AI_API")

#Defining our LLM configuration 
llm = OpenAI(model_name = OPENAI_MODEL_VERSION, temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS, top_p=OPENAI_TOP_P, openai_api_key=OPENAI_API_KEY)

def load_docs():
    # directory = MODEL_DOCS_DIR
    for dirpath, _, filenames in os.walk(MODEL_DOCS_DIR):
        for filename in filenames:
            if filename.endswith('.txt'):
                with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as file:
                    documents = file.readlines()
                documents = [doc.strip() for doc in documents]
                client.add(collection_name="knowledge-base", documents=documents)
                
    return 1

#Define prompt template using langchain
def rag(chat_history: list[str], question: str, n_points: int = 3) -> str:
    results = client.query(
        collection_name="knowledge-base",
        query_text=question,
        limit=n_points,
    )
    
    context = "\n".join(r.document for r in results)
    context = context.strip()
    
    # Define the prompt template with input variables
    prompt_template = PromptTemplate(
        input_variables=["question", "chat_history", "context"],
        template="""
            ### ROLE
            You are **AutoML-GPT**, an expert assistant who can  
            (1) answer conceptual questions about classical ML models, and  
            (2) build / run those models on request.

            ### MEMORY & CONTEXT
            • Always consult **{chat_history}** first - it overrides any other source.  
            • Use **{context}** (the technical docs) when you need factual model details.  
            If a fact is missing, admit it: “Thats all I know.”  
            • Ignore anything that contradicts the above two sources.

            ### INTERACTION MODES  
            You operate in two modes — decide which one fits the user's message.

            1. **Info-Mode** (explain / compare / theory)  
            - Simply answer the question in plain English.

            2. **Builder-Mode** (JSON spec or run request)  
            a. If the user says **“build / train / run”** and supplies a complete JSON →  
                • Return the JSON unchanged.  
            b. If required keys are missing →  
                • Ask concise follow-up questions.  
            c. If the user says **“run”** (or similar) after a valid JSON exists →  
                • Reply with **`-1`** to signal the host code to execute the pipeline.

            ### JSON SPEC FORMAT  
            Return exactly one JSON object (no extraneous text) with these keys:

            ```json
            [
            "filename": "<file>.csv",
            "model_name": "<svm | decision_tree | logistic_regression>",
            "param": [ "<name>": <value_or_list> ],
            "target_variable": "<column>",
            "split": 0.2,
            "flag": 0
            ]

            Rules
            • model_name: map synonyms - “support vector machine” → svm, “lr” → logistic_regression, etc.
            • only input the parameter values set by the user, if the user didnt ask you to set any other value, dont include it in the json file
            • flag = 1 only when the user asks for hyper-parameter tuning or supplies multiple values.
            • For tuning, store each parameters values as a list.
            • Default any omitted parameter sensibly, then tell the user what you used.

            MULTIPLE MODELS & COMPARISON

            Store successive specs as model 1, model 2… in memory.
            When asked, compare their accuracy or other metrics you were given.

        USER QUESTION: {question}
    """
    )

    # Create the Langchain instance
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    response = chain.run(question=question, chat_history=chat_history, context = context)
    return response
