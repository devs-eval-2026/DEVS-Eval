import os
import logging
import sys
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

class Retriever:
    def __init__(self, stored_index = None, path = None, api_version="2023-03-15-preview"):
        self.embed_model, self.llm = self.setup_api(api_version)
        self.index = self.retrieve_documents(stored_index, path)

    def setup_api(self, api_version):
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Enter OpenAI API key:")
            os.environ["OPENAI_API_KEY"] = api_key

        api_key = os.environ["OPENAI_API_KEY"]
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002", api_key=api_key
        )

        llm = OpenAI(api_key=api_key, model="gpt-3.5-turbo")

        return embed_model, llm

    def retrieve_documents(self, stored_index = None, path = None):
        # if stored_index exists, load it
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        if os.path.exists(stored_index):
            print("Loading index from storage")
            storage_context = StorageContext.from_defaults(persist_dir=stored_index)
            index = load_index_from_storage(storage_context)
        else: 
            print("Building index from scratch")
            nodes = self.read_docs(path)
            index = self.build_index(nodes)
            index.storage_context.persist(persist_dir=stored_index)
        return index 

    def read_docs(self, path = "rodada2/docs/r"):
        
        documents = SimpleDirectoryReader(
            path
        ).load_data()

        # parser = MarkdownNodeParser()
        
        splitter = TokenTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            separator=" ",
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Read {len(documents)} documents")
        print(f"Extracted {len(nodes)} nodes")
        return nodes

    def build_index(self, nodes):
        index = VectorStoreIndex(nodes)
        return index

    def generate_prompt_for_index(self, query, num_queries=10):
        # The prompt is not necessary 
        QUERY_GEN_PROMPT = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
        )
        formatted_prompt = QUERY_GEN_PROMPT.format(num_queries=num_queries, query=query)

        query_engine = self.index.as_query_engine()
        response = query_engine.query(formatted_prompt)
        questions = response.response.split('\n')

        print(questions) 
        return questions
 
    def query_documents(self, questions):
        retriever = self.index.as_retriever()

        context = set()
        for q in questions:
            if not isinstance(q, str) or q == "":
                continue
            nodes = retriever.retrieve(q)
            context.add(nodes[0].text)
        return context

if __name__ == "__main__":
    retriever = Retriever(stored_index='devs-index', path='../docs/r')
    query = '''simulates a manufacturing assembly line with multiple workstations, job arrivals, processing times, machine failures and repairs, and calculates throughput, cycle time, and resource utilization.'''
    questions = retriever.generate_prompt_for_index(query)
    context = retriever.query_documents(questions)
    for i, c in enumerate(context):
        print(f'Context {i}:')
        print(c)

    