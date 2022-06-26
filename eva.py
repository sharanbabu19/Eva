## Link AutoBot

# Necessary Imports
import streamlit as st
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util

# Set page configs
st.set_page_config(page_title="Eva", page_icon="🤖")

# Set a title for the webpage
st.title("Eva 🤖")

# Function to retreive text content from given URL
def get_text_from_url(url):
    # Get the text from the URLs
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    # Split the text into paragraphs
    paragraphs = text.split('\n')
    return paragraphs

# Process paragraph text and populate helpful data structures
@st.cache
def get_paragraphs(url):
	"""
	Given a URL, retrieve all paragraphs from it
	"""
	response = requests.get(url)
	soup = BeautifulSoup(response.text, 'html.parser')
	paragraphs = soup.find_all('p')
	processed_paras = []
	all_sentences = []
	sentence_to_para = {}

	i = 0
	for paragraph in paragraphs:
		cleaned_para = paragraph.text
		processed_paras.append(cleaned_para)
		smaller_sentences = cleaned_para.split(".")
		for sentence in smaller_sentences:
			all_sentences.append(sentence)
			sentence_to_para[sentence] = i
		i += 1	
	    

	return processed_paras, all_sentences, sentence_to_para

# Load sentence similarity model 
@st.cache(allow_output_mutation=True)
def load_model():
	# Load Semantic Search Model
	embedder =  SentenceTransformer('msmarco-distilbert-base-tas-b') 
	return embedder   

# Take user inputs
url = st.text_input("Enter URL")
st.caption("Ex: https://srikarsamudrala.github.io/srikarsamudrala11/")

st.write("")
st.write("")

query = st.text_input("Enter Query")
st.caption("Ex: Who is srikar samudrala?")

if st.button("Submit"):

	# Loading gif
	gif_runner = st.image("processing.gif")

	processed_paras, all_sentences, sentence_to_para = get_paragraphs(url)
	#st.write(processed_paras)

	# Perform Similarity Check 
	embedder = load_model()
	query_embedding = embedder.encode(query)
	passage_embeddings = embedder.encode(all_sentences)

	# Perform semantic search over text corpus
	# Return top 3 passage matches
	hits = util.semantic_search(query_embedding, passage_embeddings, top_k=2)
	hits = hits[0]
	output = []

	# Create Response Object
	for hit in hits:
		output.append({"sentence":str(all_sentences[hit['corpus_id']]),"score":hit['score'], "para_id": sentence_to_para[all_sentences[hit['corpus_id']]]})

	# Empty gif after completion
	gif_runner.empty()

	# Display results
	for num, result in enumerate(output):
		main_sentence = result['sentence']
		#st.write(f"OYYYY: {main_sentence}")
		main_passage = processed_paras[result['para_id']]
		st.subheader(f"Result {num+1}:")
		#st.write(f"Main Passage is: {main_passage}")
		main_passage_list = main_passage.split(".")
		final_list = []
		for x in main_passage_list:
			if x != main_sentence:
				final_list.append(x)
			else:
				final_list.append(f"<font style='color:red'><b>{x}</b></font>")
		final_string = "".join(final_list)			
		st.markdown(f"{final_string}",unsafe_allow_html=True)