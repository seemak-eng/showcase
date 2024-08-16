from utils import *

pinecone_index = os.environ.get('PINECONE_CUSTOM_INDEX')
success = load_upsert_dataset(pinecone_index)
if success == 1:
    print("Dataset is successfully upserted in Pinecone")