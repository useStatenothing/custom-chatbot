import streamlit as st
from dotenv import load_dotenv as env
from langchain import document_loaders, embeddings, llms
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import os
env()

# loading the unstructured data in thr form of doc

loader = document_loaders.TextLoader("./content.txt")
document = loader.load()

#converting the corpus into chunks
text_splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap=0)
chunk_doc = text_splitter.split_documents(document)


#embedding the chunk and dumb in the vectorstore (faiss) , where faiss used to search the relevant patterns
embeddings = embeddings.HuggingFaceEmbeddings()


db = FAISS.from_documents(chunk_doc, embeddings)


llm = llms.HuggingFaceHub(repo_id= "tiiuae/falcon-7b-instruct")
chain = load_qa_chain(llm, chain_type="stuff")



st.header("AI Chat powered by Falcon-7b", divider=True)
st.text("The model's data is derived from P2f Semiconductors' social media handles, including")
st.text("their website, LinkedIn, and other relevant social media platforms and webpages.")

st.subheader("Please stand by; the process will require some time.")
st.text("Kindly provide a high-level prompt to enhance accuracy in generating responses.")
st.text("If the response is unclear, please try again for better assistance.")
user_input = st.text_input("" ,placeholder= "Type here....", key="inputs")

if st.button("Submit", type="primary"):
    st.markdown("_______________")
    response = st.empty()

    doc = db.similarity_search(user_input)
    try:
        result = chain.run(input_documents= doc, question= user_input)
    except :
        
        result = "Something went wrong"

    if result is None:
        result  = "submit again "
    print(result)
    response.write(result)

st.header("")
st.header("")
st.header("")
st.header("")
st.header("")
st.header("",divider=True)

st.text("As previously stated, the model's data is sourced from the internet. Consequently,")
st.text("the output may not achieve high accuracy and could benefit from improvement by ")
st.text("incorporating additional relevant data pertaining to the P2F semi. ")

