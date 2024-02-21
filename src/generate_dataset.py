from argparse import ArgumentParser
import os, sys, json, glob
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lmqg import TransformersQG
import pandas as pd
from tqdm import tqdm

def create_question(context):
  outputs = model.generate_qa(context)

  return [{
      'question':question,
      'answer': answer,
  } for (question, answer) in outputs]



parser = ArgumentParser(
    prog="""
        This is simple way to generate question-answer from list of context(i.e paragraphs)
        .Please It's just a simple and worst to try
        You should use chatgpt instead of.
    """
)
parser.add_argument(
    "--folder-text-file",
    type=str,
    help="Provide folder contatin many files .txt",
    required=True
)
parser.add_argument(
    "--output-csv",
    type=str,
    help="Provide path save dataset after generated",
    required=True
)

if __name__ == '__main__':
    args = parser.parse_args()
    folder = args.folder_text_file 
    DATA_PATH = glob.glob(os.path.join(folder, "*.txt"))
    documents = [open(i).read() for i in DATA_PATH]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=256)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    # Split raw-text to many paragraphs
    docs = text_splitter.create_documents(documents)
    print(f"Total len docs = {len(docs)}")

    model = TransformersQG('lmqg/t5-large-squad-qag', language="en")
    
    x = []
    for i in tqdm(range(len(docs))):
        dicts = create_question(docs[i].page_content)
        x.extend([{
            'question':y['question'],
            'answer':y['answer'],
            'content':docs[i].page_content
        } for y in dicts])
    pd.DataFrame(x).to_csv(
       args.output_csv, index=False)