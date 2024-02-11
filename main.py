from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

load_dotenv()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

embeddings = OpenAIEmbeddings()

# Creating a Chroma instance and want to calculate embeddings for all the documents inside there
# Reaching out to OpenAI and calculating embeddings for each of the texts
# They are stored in a directory called "emb" and can be reused later on
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

results = db.similarity_search(
    "What is an interesting fact about the English Language?",
)

for result in results:
    print("\n")
    print(result.page_content)
