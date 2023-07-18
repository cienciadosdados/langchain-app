import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# Adicionar a imagem no cabeçalho
image_url = "https://cienciadosdados.com/images/CINCIA_DOS_DADOS_4.png"
st.image(image_url, use_column_width=True)

# Adicionar o nome do aplicativo
st.subheader("Q&A com IA - PLN usando LangChain")

# Componentes interativos
file_input = st.file_uploader("Upload a PDF file")
openaikey = st.text_input("Enter your OpenAI API Key", type='password')
prompt = st.text_area("Enter your questions", height=160)
run_button = st.button("Run!")

select_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Função de perguntas e respostas
def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    return result

# Função para exibir o resultado no Streamlit
def display_result(result):
    st.markdown("### Result:")
    st.write(result["result"])
    st.markdown("### Relevant source text:")
    for doc in result["source_documents"]:
        st.markdown("---")
        st.markdown(doc.page_content)

# Execução do app
if run_button and file_input and openaikey and prompt:
    with st.spinner("Running QA..."):
        # Salvar o arquivo PDF em um local temporário
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp.pdf")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_input.read())

        # Configurar a chave de API do OpenAI
        os.environ["OPENAI_API_KEY"] = openaikey

        # Executar a função de perguntas e respostas
        result = qa(temp_file_path, prompt, select_chain_type, select_k)

        # Exibir o resultado
        display_result(result)
