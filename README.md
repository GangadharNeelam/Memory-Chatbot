## Memory Chat-Bot

## Steps
1. Upload a PDF file
2. Ask questions about the PDF
3. Get answers from the PDF

## About
This is an LLM powered chatbot build using:
 - [Streamlit](https://streamlit.io/)
 - [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
 - [OpenAI](https://openai.com/)

## Architecture
![Architecture](https://github.com/GangadharNeelam/Memory-Chat-Bot/assets/93145713/679b2af6-acf4-4de1-bf4c-e599544b549c)
![Architecture2](https://github.com/GangadharNeelam/Memory-Chat-Bot/assets/93145713/76891f0c-99e1-4a2a-ae3c-a49eb0cdc951)


## How does it work?
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
