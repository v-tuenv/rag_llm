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
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}
Question: {question}
Answer: """




import glob, os, sys
from argparse import ArgumentParser
import torch

parser = ArgumentParser(
    prog="""
        This is demo question-answer
    """
)
parser.add_argument(
    "--folder-text-file",
    type=str,
    help="Provide folder contatin many files .txt",
    required=True
)
parser.add_argument(
    "--model-path-or-name",
    type=str,
    default="Qwen/Qwen1.5-4B",
    help="Provide folder contatin model generation from step train_text_generation",
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

    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    db = FAISS.from_documents(docs, embeddings)



    ## Define model transformer generator

    # Specify the model name you want to use
    model_name = args.model_path_or_name
    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   
    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        model=model_name, 
        tokenizer=tokenizer,
        return_tensors='pt',
        torch_dtype=torch.float16,
        max_new_tokens=512,
        device='cuda',
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
    
    )

    custom_rag_prompt = PromptTemplate.from_template(template)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})


    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    while True:
        question = input("Provide question to model provide 'STOP' if you don't want predict: question: ")
        if question=="STOP":
            print("You provide stop ->> done exit program")

        answer = rag_chain.invoke(question)
        print(f'\nquestion={question}\nanswer={answer}\n')
        
        
