from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from os.path import exists
import json
from pathlib import Path

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    # metadata["summary"] = record.get("feedbackText")
    metadata["timestamp"] = record.get("createdDate")
    metadata["totalRevenue"] = record.get("totalRevenue")
    metadata["productNames"] = record.get("productNames")

    return metadata

class ChatCSV:
    vector_store = None
    retriever = None
    memory = None
    chain = None

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.

        This constructor sets up the following components:
        - A ChatOllama model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """
        # Initialize the ChatOllama model.
        # self.model = ChatOllama(model="llama2:7b-chat")
        self.model = ChatOllama(model="mistral:latest")

        # Initialize the RecursiveCharacterTextSplitter with specific chunk settings.
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        # Initialize the PromptTemplate with a predefined template for constructing prompts.
        # self.prompt = PromptTemplate.from_template(
        #     """
        #     <s> [INST] You are an assistant for summarizing feedback. Use the following pieces of retrieved context 
        #     to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
        #      maximum and keep the answer concise. [/INST] </s> 
        #     [INST] Question: {question} 
        #     Context: {context} 
        #     Answer: [/INST]
        #     """
        # )
        # Your tone should be professional and informative

        # self.prompt = PromptTemplate.from_template(
        #     """
        #     <s> [INST]You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative. [/INST] </s> 
        #     [INST] Human Question: {question} 
        #     Context: {context}
        #     Previous conversation: {chat_history}
        #     Answer: [/INST]

        #     """
        # )

        self.prompt = ChatPromptTemplate.from_messages(
            messages=[
                # SystemMessagePromptTemplate.from_template("You are a knowledgeable chatbot, here to help summarize users response. Your tone should be professional and informative."),
                SystemMessagePromptTemplate.from_template("You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
                # ("context", "{context}")
            ]
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question", output_key="answer",return_messages=True)
        # self.prompt.format(product="virtualizaion software")
        # self.prompt = prompt_template.format_prompt(product="virtualizaion software").to_messages()


    def ingest(self, ingest_path: str, index: bool, type: str):
        '''
        Ingests data from a CSV file containing resumes, process the data, and set up the
        components for further analysis.

        Parameters:
        - csv_file_path (str): The file path to the CSV file.

        Usage:
        obj.ingest("/path/to/data.csv")

        This function uses a CSVLoader to load the data from the specified CSV file.

        Args:
        - file.path (str): The path to the CSV file.
        - encoding (str): The character encoding of the file (default is 'utf-8').
        - source_column (str): The column in the CSV containing the data (default is "Resume").
        '''        
        # loader = CSVLoader(
        #     file_path=csv_file_path,
        #     # file_path='/home/chris/projects/csv-chatbot-local-llm/data/looper.csv'
        #     encoding='utf-8',
        #     source_column="summary"
        #     )
        # loader = TextLoader(
        #     file_path=csv_file_path,
        #     # file_path='/home/chris/projects/csv-chatbot-local-llm/data/looper.csv'
        #     encoding='utf-8'
        #     # source_column="summary"
        #     )
        embeddings=FastEmbedEmbeddings()
        if index:
            print("loading indexes")
            vector_store = FAISS.load_local("./data",  embeddings)
            print("index loaded")
        else:
            if type == "web":
                #loader = WebBaseLoader(['https://grocycle.com/about-us/', 'https://grocycle.com/mushroom-cultivation-courses/', 'https://grocycle.com/blog/', 'https://grocycle.com/mushroom-growing-hub/', 'https://grocycle.com/mushroom-education/', 'https://grocycle.com/microgreens-hub/', 'https://grocycle.com/permaculture/', 'https://grocycle.com/homesteading/', 'https://grocycle.com/grocycle-youtube/', 'https://grocycle.com/farmtour/', 'https://grocycle.com/subscribe/', 'https://grocycle.com/workshop/', 'https://grocycle.com/about-us/', 'https://grocycle.com/mushroom-cultivation-courses/', 'https://grocycle.com/blog/', 'https://grocycle.com/mushroom-growing-hub/', 'https://grocycle.com/mushroom-education/', 'https://grocycle.com/microgreens-hub/', 'https://grocycle.com/permaculture/', 'https://grocycle.com/homesteading/', 'https://grocycle.com/grocycle-youtube/', 'https://grocycle.com/farmtour/', 'https://grocycle.com/subscribe/', 'https://grocycle.com/workshop/', 'https://grocycle.com/mushroom-cultivation-courses/', 'https://grocycle.com/mushroom-growing-bags/', 'https://grocycle.com/mushroom-growing-bags/', 'https://grocycle.com/how-to-make-grain-spawn/', 'https://grocycle.com/how-to-make-grain-spawn/', 'https://grocycle.com/how-to-make-liquid-culture/', 'https://grocycle.com/how-to-make-liquid-culture/', 'https://grocycle.com/pf-tek/', 'https://grocycle.com/pf-tek/', 'https://grocycle.com/how-to-grow-morel-mushrooms/', 'https://grocycle.com/how-to-grow-morel-mushrooms/', 'https://grocycle.com/mushroom-inoculation/', 'https://grocycle.com/mushroom-inoculation/', 'https://grocycle.com/how-much-can-you-sell-mushrooms-for/', 'https://grocycle.com/how-much-can-you-sell-mushrooms-for/', 'https://grocycle.com/do-mushrooms-need-light/', 'https://grocycle.com/do-mushrooms-need-light/', 'https://grocycle.com/is-mushroom-growing-sustainable/', 'https://grocycle.com/is-mushroom-growing-sustainable/', 'https://grocycle.com/growing-mushrooms-in-bottles/', 'https://grocycle.com/growing-mushrooms-in-bottles/', 'https://grocycle.com/mushroom-growing-terminology/', 'https://grocycle.com/mushroom-growing-terminology/', 'https://grocycle.com/common-mistakes-mushroom-growers-make/', 'https://grocycle.com/common-mistakes-mushroom-growers-make/', 'https://grocycle.com/how-to-pasteurize-mushroom-substrate/', 'https://grocycle.com/how-to-pasteurize-mushroom-substrate/', 'https://grocycle.com/how-to-sterilize-mushroom-substrate/', 'https://grocycle.com/how-to-sterilize-mushroom-substrate/', 'https://grocycle.com/cold-water-lime-bath-pasteurization-for-mushroom-substrate/', 'https://grocycle.com/cold-water-lime-bath-pasteurization-for-mushroom-substrate/', 'https://grocycle.com/harvest-mushrooms/', 'https://grocycle.com/harvest-mushrooms/', 'https://grocycle.com/mushroom-grow-kits/', 'https://grocycle.com/mushroom-grow-kits/', 'https://grocycle.com/what-conditions-are-needed-for-a-mushroom-to-grow/', 'https://grocycle.com/what-conditions-are-needed-for-a-mushroom-to-grow/', 'https://grocycle.com/mushroom-pinning/', 'https://grocycle.com/mushroom-pinning/', 'https://grocycle.com/grow-mushrooms-indoors/', 'https://grocycle.com/grow-mushrooms-indoors/', 'https://grocycle.com/monotub-fruiting-chamber/', 'https://grocycle.com/monotub-fruiting-chamber/', 'https://grocycle.com/how-to-make-a-mushroom-spore-print/', 'https://grocycle.com/how-to-make-a-mushroom-spore-print/', 'https://grocycle.com/shotgun-fruiting-chamber/', 'https://grocycle.com/shotgun-fruiting-chamber/', 'https://grocycle.com/mushroom-grow-tent/', 'https://grocycle.com/mushroom-grow-tent/', 'https://grocycle.com/coco-coir-substrate/', 'https://grocycle.com/coco-coir-substrate/', 'https://grocycle.com/how-to-build-a-martha-grow-tent/', 'https://grocycle.com/how-to-build-a-martha-grow-tent/', 'https://grocycle.com/what-is-masters-mix/', 'https://grocycle.com/what-is-masters-mix/', 'https://grocycle.com/best-mushroom-growing-books/', 'https://grocycle.com/best-mushroom-growing-books/', 'https://grocycle.com/how-to-clone-mushrooms/', 'https://grocycle.com/how-to-clone-mushrooms/', 'https://grocycle.com/how-to-grow-mushrooms-on-straw/', 'https://grocycle.com/how-to-grow-mushrooms-on-straw/', 'https://grocycle.com/growing-mushrooms-on-sawdust-blocks/', 'https://grocycle.com/growing-mushrooms-on-sawdust-blocks/', 'https://grocycle.com/growing-mushrooms-on-a-book/', 'https://grocycle.com/growing-mushrooms-on-a-book/', 'https://grocycle.com/how-to-sell-mushrooms-to-restaurants/', 'https://grocycle.com/how-to-sell-mushrooms-to-restaurants/', 'https://grocycle.com/mushroom-growing-supplies-list/', 'https://grocycle.com/mushroom-growing-supplies-list/', 'https://grocycle.com/how-long-does-it-take-to-grow-mushrooms/', 'https://grocycle.com/how-long-does-it-take-to-grow-mushrooms/', 'https://grocycle.com/grow-reishi-mushrooms/', 'https://grocycle.com/grow-reishi-mushrooms/', 'https://grocycle.com/growing-mushrooms-in-buckets/', 'https://grocycle.com/growing-mushrooms-in-buckets/', 'https://grocycle.com/mushroom-growing-business-plan/', 'https://grocycle.com/mushroom-growing-business-plan/', 'https://grocycle.com/grow-mushrooms-on-cardboard/', 'https://grocycle.com/grow-mushrooms-on-cardboard/', 'https://grocycle.com/grow-mushrooms-in-a-bag/', 'https://grocycle.com/grow-mushrooms-in-a-bag/', 'https://grocycle.com/easiest-mushrooms-to-grow/', 'https://grocycle.com/easiest-mushrooms-to-grow/', 'https://grocycle.com/growing-lions-mane/', 'https://grocycle.com/growing-lions-mane/', 'https://grocycle.com/mushroom-substrate/', 'https://grocycle.com/mushroom-substrate/', 'https://grocycle.com/how-to-build-a-mushroom-fruiting-chamber/', 'https://grocycle.com/how-to-build-a-mushroom-fruiting-chamber/', 'https://grocycle.com/how-to-grow-mushrooms-outdoors/', 'https://grocycle.com/how-to-grow-mushrooms-outdoors/', 'https://grocycle.com/diy-mushroom-spawn/', 'https://grocycle.com/diy-mushroom-spawn/', 'https://grocycle.com/how-to-grow-mushrooms/', 'https://grocycle.com/how-to-grow-mushrooms/', 'https://grocycle.com/how-to-grow-shiitake-mushrooms/', 'https://grocycle.com/how-to-grow-shiitake-mushrooms/', 'https://grocycle.com/how-to-grow-mushrooms-on-logs-the-ultimate-guide/', 'https://grocycle.com/how-to-grow-mushrooms-on-logs-the-ultimate-guide/', 'https://grocycle.com/growing-mushrooms-in-coffee-grounds/', 'https://grocycle.com/growing-mushrooms-in-coffee-grounds/', 'https://grocycle.com/how-to-grow-oyster-mushrooms/', 'https://grocycle.com/how-to-grow-oyster-mushrooms/', 'https://grocycle.com/5-things-i-wish-i-knew-before-i-started-growing-mushrooms-for-a-living/', 'https://grocycle.com/5-things-i-wish-i-knew-before-i-started-growing-mushrooms-for-a-living/', 'https://grocycle.com/how-to-set-up-a-low-tech-mushroom-farm/', 'https://grocycle.com/how-to-set-up-a-low-tech-mushroom-farm/', 'https://grocycle.com/how-to-grow-mushrooms/', 'https://grocycle.com/how-to-grow-mushrooms/', 'https://grocycle.com/how-to-grow-mushrooms/', 'https://grocycle.com/how-to-grow-mushrooms-on-logs-the-ultimate-guide/', 'https://grocycle.com/how-to-grow-mushrooms-on-logs-the-ultimate-guide/', 'https://grocycle.com/how-to-grow-mushrooms-on-logs-the-ultimate-guide/', 'https://grocycle.com/how-to-set-up-a-low-tech-mushroom-farm/', 'https://grocycle.com/how-to-set-up-a-low-tech-mushroom-farm/', 'https://grocycle.com/how-to-set-up-a-low-tech-mushroom-farm/', 'https://grocycle.com/most-profitable-crops-to-grow/', 'https://grocycle.com/most-profitable-crops-to-grow/', 'https://grocycle.com/most-profitable-crops-to-grow/'])
                # loader = WebBaseLoader(['https://ezmushroom.com/how-to-grow-mushrooms-at-home/', 'https://ezmushroom.com/recommended-equipment/', 'https://ezmushroom.com/knowledge/', 'https://ezmushroom.com/shop/', 'https://ezmushroom.com/', 'https://ezmushroom.com/', 'https://ezmushroom.com/cart/', 'https://ezmushroom.com/', 'https://ezmushroom.com/how-to-grow-mushrooms-at-home/', 'https://ezmushroom.com/recommended-equipment/', 'https://ezmushroom.com/knowledge/', 'https://ezmushroom.com/shop/', 'https://ezmushroom.com/knowledge/what-does-mycelium-look-like/', 'https://ezmushroom.com/knowledge/what-does-mycelium-look-like/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/white-fuzz-on-mushrooms-safe-to-eat-growing-advice/', 'https://ezmushroom.com/knowledge/white-fuzz-on-mushrooms-safe-to-eat-growing-advice/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/puffball-mushrooms-guide/', 'https://ezmushroom.com/knowledge/puffball-mushrooms-guide/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/liquid-culture-vs-spore-syringe/', 'https://ezmushroom.com/knowledge/liquid-culture-vs-spore-syringe/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/reishi-mushroom-growing-profile/', 'https://ezmushroom.com/knowledge/reishi-mushroom-growing-profile/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/lions-mane-mushroom-growing-profile/', 'https://ezmushroom.com/knowledge/lions-mane-mushroom-growing-profile/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/king-oyster-mushroom-growing-profile/', 'https://ezmushroom.com/knowledge/king-oyster-mushroom-growing-profile/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/blue-oyster-mushroom-growing-profile/', 'https://ezmushroom.com/knowledge/blue-oyster-mushroom-growing-profile/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/5-easiest-mushrooms-to-grow-indoors/', 'https://ezmushroom.com/knowledge/5-easiest-mushrooms-to-grow-indoors/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/do-turkey-tail-mushrooms-grow-in-the-uk/', 'https://ezmushroom.com/knowledge/do-turkey-tail-mushrooms-grow-in-the-uk/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/does-growing-mushrooms-indoor-smell/', 'https://ezmushroom.com/knowledge/does-growing-mushrooms-indoor-smell/', 'https://ezmushroom.com/author/mrmush/', 'https://ezmushroom.com/knowledge/does-lions-mane-improve-gaming-performance/', 'https://ezmushroom.com/knowledge/does-lions-mane-improve-gaming-performance/', 'https://ezmushroom.com/author/mrmush/'])
                loader = WebBaseLoader(ingest_path)
            elif type == "json":
                loader = JSONLoader(
                    file_path=ingest_path,
                    # jq_schema=".data[].feedbackText",
                    jq_schema=".[].summary, .[].service",
                    # content_key="summary",
                    # metadata_func=metadata_func
                    )
            # loads the data
            data = loader.load()
            print(data)
            # splits the documents into chunks
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            # all_splits = text_splitter.split_documents(data)
            # vector_store = FAISS.from_documents(all_splits, embeddings)
            # # creates a vector store using embedding
            # if exists('./data/index.faiss'):
            #     vector_store_i = FAISS.load_local("./data",  embeddings)
            #     # vector_store_i = FAISS.from_documents(all_splits, embeddings)
            #     vector_store.merge_from(vector_store_i)
            # # else:
            # #     vector_store = FAISS.from_documents(all_splits, embeddings)
            # vector_store.save_local("./data")
        # sets up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # Define a processing chain for handling a question-answer scenario.
        # The chain consists of the following components:
        # 1. "context" from the retriever
        # 2. A passthrough for the "question"
        # 3. Processing with the "prompt"
        # 4. Interaction with the "model"
        # 5. Parsing the output using the "StrOutputParser"

        # self.memory.load_memory_variables({})
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"]))
            )
                | self.prompt
                | self.model
                | StrOutputParser())
        self.chain = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough(), "chat_history": RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history")}
        ).assign(answer=rag_chain_from_docs)

    def ask(self, query: str):
        """
        Asks a question using the configured processing chain.

        Parameters:
        - query (str): The question to be asked.

        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """
        # if not self.chain:
        #     return "Please, add a website first."
        # load memory for history
        self.memory.load_memory_variables({})
        response = self.chain.invoke(query)
        print(response)
        query = {"question": query}
        self.memory.save_context(query, {"answer": response["answer"]})
        return response
        # return self.chain({"question": query, "chat_history": self.chat_history})

    def clear(self):
        """
        Clears the components in the question-answering system.

        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        # Set the vector store to None.
        self.vector_store = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None