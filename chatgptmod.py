import os
import pinecone
import tempfile
import PyPDF2
import time
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st


# Check to see if there is an environment variable with your API keys, if not, use what you put below
OPENAI_API_KEY = st.secrets["openai_api"]
PINECONE_API_KEY = st.secrets["pinecone_api"]
PINECONE_API_ENV = st.secrets["pinecone_env"]

api_key = PINECONE_API_KEY
pinecone.init(
    api_key=api_key,
    environment=PINECONE_API_ENV
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def query_find(query):
    
    docsearch = Pinecone.from_existing_index(index_name='intelpdf', embedding=embeddings)
    # Perform the similarity search and get the documents
    docs = docsearch.similarity_search(query=query, k=5)

    # Initialize OpenAI model and perform question-answering
    llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Assuming 'docs' is the list containing your documents or elements
    pg_ref = [doc.metadata['page'] + 1 for doc in docs]

    result = chain.run(input_documents=docs, question=query)
    
    if pg_ref:
        page_number = pg_ref[0]
        return result, page_number
    else:
        print('Nothing to print')
    

def merge_pdfs(input_paths, output_path):

    merge_file = PyPDF2.PdfFileMerger(strict=False)
    for path in input_paths:
        merge_file.append(PyPDF2.PdfFileReader(path, 'rb'))

    with open(output_path, 'wb') as output_file:
        merge_file.write(output_file)

# ... (previous code)

# Function to perform the document search and return the results
def perform_search(pdf_paths):
    # Check if the file paths exist
    if not all(os.path.exists(path) for path in pdf_paths):
        st.warning("One or more PDF files do not exist.")
        return 'Nothing to Print 1', 'Nothing to Pring 2'

    # Create a temporary directory to store the uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Merge the PDF files into a single file
        merged_file_path = os.path.join(temp_dir, "MergedPDF.pdf")
        merge_pdfs(pdf_paths, merged_file_path)

        # Load the merged PDF file
        loader = PyPDFLoader(merged_file_path)
        data = loader.load()

        # Split the PDF into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        # Upload the contents and metadata to Pinecone
        metadata = [t.metadata for t in texts]
        page = [t.page_content for t in texts]
        if not pinecone.list_indexes():
            pinecone.create_index(dimension=1536, name='intelpdf', metric='cosine')
            docsearch = Pinecone.from_texts(page, embeddings, metadatas=metadata, index_name='intelpdf')
            time.sleep(2)
        else:

            #pinecone.delete_index('intelpdf')
            index = pinecone.Index("intelpdf")
            vectorstore = Pinecone(index, embeddings.embed_query,'text')
            vectorstore.add_texts(page,metadata)
            time.sleep(2)


# Streamlit app
def main():
    st.title("PDF Document Search and Question Answering")
    
    # File Upload
    uploaded_files = st.file_uploader("Choose multiple PDF files to merge and search:", type=["pdf"], accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) >= 1:
        # Create a temporary directory to store the uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_paths = []
            for i, file in enumerate(uploaded_files):
                # Save the uploaded PDF files to the temporary directory
                temp_path = os.path.join(temp_dir, f"uploaded_pdf_{i}.pdf")
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(temp_path)

            if st.button("Done", key="done_button"):
                st.text("Uploading files...")
                perform_search(pdf_paths)
                st.session_state.perform_search_done = True  # Set the session state variable to indicate files are uploaded

    # Wait for files to be uploaded and perform_search to be done before showing the query input
    if "perform_search_done" in st.session_state and st.session_state.perform_search_done:
        query = st.text_input("Enter your question:")
    
        if query:
            con, pgn = query_find(query)
            if con is not None and pgn is not None:
                st.subheader("Answer:")
                st.write(con)
                st.subheader("Page Number:")
                st.write(pgn)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

