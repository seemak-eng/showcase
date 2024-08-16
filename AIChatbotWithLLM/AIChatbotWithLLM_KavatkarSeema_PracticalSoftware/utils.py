#region Utils
#region Import
import os
from dotenv import load_dotenv
# Pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Open AI
import openai
from openai import OpenAI

#Langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Utilities
from tqdm.auto import tqdm
import time

#endregion


#region Initialization

## Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

# pinecone_index = os.environ.get('PINECONE_INDEX')
pinecone_region = os.environ.get('PINECONE_REGION')
pinecone_cloud = os.environ.get('PINECONE_CLOUD')

## Initialize Pinecone client with API key
pc = Pinecone(api_key=pinecone_api_key)

# ## Initialize Pinecone Index
# if pinecone_index in pc.list_indexes().names():
#     print("Initialize Pinecone Index..")
#     index = pc.Index(pinecone_index)
# else:
#     pass

## dimensionality of text-embedding-ada-002
dimension = 1536

## OpenAI embedding model
embed_model = "text-embedding-ada-002"

## The maximum character limit for the combined contexts in the prompt
limit = 3750

## Top 'k'
k=2

#endregion


#region Creating Index and Upsert Dataset

def read_doc(directory):
    """
    Reads documents from a specified directory.

    Parameters:
    - directory (str): Path to the directory containing documents.

    Returns:
    - documents list[str]: List of document contents.
    """
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents



def chunk_data(docs, chunk_size=800, chunk_overlap = 50):
    """
    Splits the loaded documents into chunks.

    Parameters:
    - docs (list[str]): List of document contents.
    - chunk_size (int): Size of each chunk. Default is 800.
    - chunk_overlap (int): Overlap between chunks. Default is 50.

    Returns:
    - list[str]: List of chunked document contents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap = chunk_overlap)
    split_doc = text_splitter.split_documents(docs)
    return split_doc


def create_pinecone_index(pinecone_index):
    """
    Creates a Pinecone index if it doesn't exist already.

    Parameters:
    - pinecone_index (str): Name of the Pinecone index.

    Returns:
    - Pinecone.Index: Initialized Pinecone index object.
    """
    cloud_spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)

    # check if index already exists (it shouldn't if this is first time)
    if pinecone_index not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            pinecone_index,
            dimension=dimension,
            metric='cosine',
            spec=cloud_spec
        )
    else:
        pass
    # connect to index
    index = pc.Index(pinecone_index)
    return index

def load_upsert_dataset(pinecone_index):
    """
    Loads documents, chunks them, and upserts the embeddings into the Pinecone index.

    Parameters:
    - pinecone_index (str): Name of the Pinecone index.

    Returns:
    - int: Flag indicating success (1) or failure (0) of the operation.
    """
    flag = 0
    docs = read_doc('dataset/')
    chunks = chunk_data(docs = docs)

    index = create_pinecone_index(pinecone_index)

    # # Extract text content from PDF documents
    document_texts = [chunk.page_content for chunk in chunks]

    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(document_texts), batch_size)):
        # set end position of batch
        i_end = min(i+batch_size, len(document_texts))
        # get batch of lines and IDs
        lines_batch = document_texts[i: i+batch_size]
        ids_batch = [str(n) for n in range(i, i_end)]
        # create embeddings
        res = openai.embeddings.create(input=lines_batch, model=embed_model)
        embeds = [record.embedding for record in res.data]
        # prep metadata and upsert batch
        meta = [{'text': line} for line in lines_batch]
        to_upsert = zip(ids_batch, embeds, meta)
        # upsert to Pinecone
        index_ref = index.upsert(vectors=list(to_upsert))
        if index_ref is not None:
            print("Document embeddings uploaded to Pinecone index successfully.")
            flag = 1
        else:
            print("Error: Upsert operation failed.")
    return flag
#endregion


#region Retrieval

def create_embeddings(query_text):
    """
    Creates embeddings for a given query text.

    Parameters:
    - query_text (str): Text to be embedded.

    Returns:
    - list[float]: Embeddings for the input text.
    """
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    res = client.embeddings.create(input = [query_text], model=embed_model).data[0].embedding
    return res

def retrieve_query_result(query_text, pinecone_index):
    """
    Retrieves relevant query results from a Pinecone index.

    Parameters:
    - query_text (str): Text of the query.
    - pinecone_index (str): Name of the Pinecone index to query.

    Returns:
    - Dict[str, Any]: Query results including relevant contexts and metadata.
    """
    ## Initialize Pinecone client with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index)

    query_text = query_text.replace("\n", " ")
    query_vector = create_embeddings(query_text) 
  
    # get relevant contexts (including the questions)
    context = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return context

def build_prompt(query_text):
    """
    Builds a prompt for question answering based on retrieved contexts.

    Parameters:
    - query_text (str): Text of the query.

    Returns:
    - str: Prompt for question answering.
    """
    pinecone_index = os.environ.get('PINECONE_CUSTOM_INDEX')

    # get relevant contexts from pinecone
    contexts = []
    time_waited = 0
    while (len(contexts) < k and time_waited < 60 * 5): # Timeout after 60sec * 5= 5 mins
        cntx = retrieve_query_result(query_text, pinecone_index)
        contexts = contexts + [
            r['metadata']['text'] for r in cntx['matches']
        ]
        print(f"Retrieved {len(contexts)} contexts, sleeping for 5 seconds...")
        time.sleep(3)
        time_waited += 3

    if time_waited >= 60 * 5:
        print("Timed out waiting for contexts to be retrieved.")
        contexts = ["Sorry, unable to retrieved the contexts!"]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query_text}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt


def generate_response(prompt):
    """
    Generates a response based on the provided prompt.

    Parameters:
    - prompt (str): Prompt for generating the response.

    Returns:
    - Optional[str]: Generated response or None if the response is invalid.
    """
    # Instructions
    sys_prompt = "You are a helpful assistant that always answers questions."

    res = openai.chat.completions.create(
        model='gpt-3.5-turbo-0613',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # Extracting content value directly from the string
    res = str(res)
    content_start_index = res.find("content='") + len("content='")
    content_end_index = res.find("'", content_start_index)
    content_value = res[content_start_index:content_end_index]

    # Check if response is "letion(id=", if yes, return None
    if content_value.startswith("letion(id="):
        return None
    
    return content_value


#endregion


#endregion