from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #to breakdown huge transcripts into smaller chunks
from langchain_community.llms import openai
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import faiss
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()
video_url = "https://www.youtube.com/watch?v=-Osca2Zax4Y"
def create_vector_db_from_youtube_url(video_url: str) -> faiss:
    # load yt video from the url
    loader = YoutubeLoader.from_youtube_url(video_url)
    # convert loaded video into transcript 
    transcript = loader.load()
    
    # split the whole transcript into chunks madde of 1000 words each 
    # chunk_size - how much each chunk will contain 
    # chunk_overlap - last number words chunk 1 that will be the first words in chunk 2 etc
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = faiss.from_documents(docs, embeddings)
    return db
# print(create_vector_db_from_youtube_url(video_url))

def get_response_from_query(db, query, k=4):
    # text-davinci can handle 4097 tokens
    # perform similarity check of the query asked on the db
    # since each doc/chunk has 1000 tokens, we use 4 chunks 
    docs = db.similarity_search(query, k=k) 
    # join the 4 chunks to get approx 4000 tokens to be sent at once to the api
    doc_page_content = "".join([d.page_content for d in docs])

    llm = openai(model='text-davinci-003')
    prompt = PromptTemplate(
        input_variables=['quetion', 'docs'],
        template=""" 
        You are a very helpful YouTube assistant that can answer questions about 
        videos based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know". 

        Your answers should be detailed.
        """,
    )
    chain = LLMChain(llm=llm)
    response = chain.run(question=query, docs=doc_page_content)
    response = response.replace("\n", "")
    return response
