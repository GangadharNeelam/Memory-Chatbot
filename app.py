import streamlit as st
import pickle
import os
import openai
from PIL import Image


from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain

from langchain.chains import RetrievalQA
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain


from langchain.callbacks import get_openai_callback

    
# Extract content from PDF
@st.cache_data
def content_extractor(pdf) -> str:
    
    """
    Extract text from PDF file.
    
    Args:
        pdf (file): PDF file

    Returns:
        text (string): text extracted from PDF
    """    
    
    pdf_reader = PdfReader(pdf)
    text = ""
    for i in pdf_reader.pages:
        text += i.extract_text()
    return text

# Split content into chunks
@st.cache_data
def text_splitter(text: str) -> list:
    
    """
    Split text into chunks of 1000 characters with 200 characters overlap. 
    Because OpenAI API has a limit of 2048 tokens per request.
    
    Args:
        text (string): text to be split into chunks
        
    Returns:
        chunks (list): list of chunks of text
    """    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Embed chunks
@st.cache_data
def embed_chunks(chunks: list, store_name: str, api: str):
    """
    Check if embeddings are already generated or not. If yes load embeddings from disk.
    Otherwise generate embeddings and save to disk for future use.

    Args:
        chunks (list): chunks of text
        store_name (str): name of the file to save the embeddings
        
    returns:
        VectorStore (object): embeddings of the chunks
    """            
    store_name = store_name
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        # st.write("Embeddings loaded from the disk")
        return VectorStore
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api)
        VectorStore = FAISS.from_texts(chunks, embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
        # st.write("Embeddings generated and saved to disk")
        return VectorStore


    
def main():
    # Main page title
    st.header("Chat with PDF")   
    
    
    # Upload pdf file
    pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf:
        
        text = content_extractor(pdf=pdf)
        chunks = text_splitter(text=text)
        
        if chunks:
            with st.expander("PDF Content", expanded=False):
                page_select = st.number_input(label = "select page", 
                                              step=1, min_value=1,
                                              max_value=len(chunks))
                chunks[page_select-1]
        
        api = st.text_input("Enter your OpenAI API key", type="password",
                            placeholder="sk-", help="https://platform.openai.com/account/api-keys")
        
        if api:
            
            if "memory" not in st.session_state:  
                st.session_state["memory"] = ConversationBufferMemory(memory_key = "chat_history")
                
            # Q&A chain
            store_name = pdf.name[:-4]
            VectorStore = embed_chunks(chunks=chunks, store_name=store_name, api=api)
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=api),
                chain_type="stuff",
                retriever=VectorStore.as_retriever()
                )
            
            # Tool for the agent
            tools = [
                Tool(
                    name="State of Union QA System",
                    func=qa.run,
                    description="useful for when you need to answer questions about current events. Input may be a partial or fully formed question.",
                )
            ]
            
            prefix = """
            Have a conversation with a human, answering the following questions as best you can based on the context and memory available.
            You have access to the single tool
            """
            suffix = """Begin!

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

            prompt_template = ZeroShotAgent.create_prompt(
                tools, prefix=prefix, suffix=suffix, input_variables=["input", "chat_history", "agent_scratchpad"]
            )
            
            # Initialize the session state memory 
            # st.session_state["memory"] = {}
            # if "memory" not in st.session_state:
            #     st.session_state["memory"] = ConversationBufferMemory(memory_key = "chat_history")
            
            # "st.session stae object", st.session_state

            
            llm = OpenAI(model_name = "gpt-3.5-turbo", openai_api_key=api)
            llm_chain = LLMChain(
                llm=llm,
                prompt=prompt_template
                )        
            
            
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent, 
                    tools=tools, 
                    verbose=True,
                    memory = st.session_state.memory
            )
            
            
            # Accept user questions/queries
            query = st.text_input("Ask questions about the PDF")
            # if query:
            #     res = agent_executor.run(query)
            #     st.write(res)
            if query:
                try:
                    # Use the agent_executor to get a response
                    res = agent_executor.run(query)
                    
                    # Display the response using st.write
                    st.write("Agent Response:")
                    st.write(res)
                except Exception as e:
                    # Handle exceptions gracefully
                    st.error(f"An error occurred: {str(e)}")
                
            # docs = VectorStore.similarity_search(query, k=2)
            # with st.expander("Citiation (relevant pages))"):
            #     st.write(docs)            

            
        # Allow the user to view the conversation history and other information stored in the agent's memory
        with st.expander("History/Memory"):
            st.session_state.memory


        
        
# Sidebar contents
with st.sidebar:
    
    st.title("PDF Chat-Bot")
    with st.expander("Steps", expanded=True):
        st.markdown("""
                    1. Upload a PDF file
                    2. Ask questions about the PDF
                    3. Get answers from the PDF
                    """)       

        
    with st.expander("About"):
        
        st.markdown("""
            This is an LLM powered chatbot build using:
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
            - [OpenAI](https://openai.com/)     
        """)
        
        st.markdown("""
                    #### How does it work?
                    #### Text Extraction: 
                     - Extract text content from the provided PDF file.
                    #### Text Chunking: 
                     - Split the extracted text into smaller, overlapping chunks of 1000 characters with a 200-character overlap between chunks.
                     - Because OpenAI API has a limit of 2048 tokens per request.
                    #### Embedding Generation:
                     - Generate embeddings (numerical representations) for each text chunk by using OpenAI API.
                     - These embeddings capture the semantic meaning of the text.
                    #### Similarity Search: 
                     - Compare the embeddings of text chunks with the user's query using a similarity metric.
                     - Rank the text chunks based on their similarity to the query.
                    #### Output:
                     - Utilize a Language Model (LLM) to process the ranked results.
                     - Provide the user with relevant and ranked information based on the query and similarity search results.
                    """)
    
    with st.expander("Architecture"):
        image = Image.open('Architecture.jpg')
        st.image(image, caption='Architecture')
        image2 = Image.open('Architecture2.jpg')
        st.image(image2, caption='Internal Architecture')
        
    add_vertical_space(3)
    st.write("Made by Gangadhar😊")
    st.markdown("""
                ### Connect with me:
                 - [LinkedIn](https://www.linkedin.com/in/gangadhar-neelam/)
                 - [GitHub](https://github.com/GangadharNeelam)
                
                """)
    
    
if __name__ == "__main__":
    main()