# Se selecciona el metodo de creacion de embeddings (open AI y hugging face) y se pregunta sobre un documento que subimos


import os
from PyPDF2 import PdfReader
import re
import openai
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
#import qdrant_client
from qdrant_client.http import models
from dotenv import load_dotenv
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import time
from transformers import pipeline
import spacy
from openai import OpenAI
import fitz
import joblib
import pickle


# Inicializa el modelo y el tokenizador una sola vez
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def extract_and_clean_text_from_pdf(uploaded_file):
    """Extrae y limpia el texto de un archivo PDF cargado como UploadedFile en Streamlit."""
    text = ""
    try:
        # Usa 'stream' para abrir el documento PDF directamente desde el buffer
        doc = fitz.open(stream=uploaded_file.getvalue())
        for page in doc:
            text += str(page.get_text())
        clean_text = " ".join(text.split('\n'))
        return clean_text
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
        return ""


def get_text_chunks(text):
    chunks = []
    while len(text) > 500:
        last_period_index = text[:500].rfind('.')
        if last_period_index == -1:
            last_period_index = 500
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)
    return chunks


def qdrant_connection(embedding_method):
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection_name = "demo_hf" if embedding_method == "Hugging Face" else "demo_openai"
    vector_size = 384 if embedding_method == "Hugging Face" else 1536

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    return qdrant_client, collection_name


def generate_embeddings(chunks):
    points = []
    i = 1
    for chunk in chunks:
        i += 1
        
        embeddings = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding

        points.append(PointStruct(id=i, vector=embeddings, payload={"text": chunk}))
    
    return points

def generate_embeddings_hf(chunks):
    points = []
    i = 1
    for chunk in chunks:
        i += 1
        # Tokeniza el texto
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # Genera los embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        points.append(PointStruct(id=i, vector=embeddings.tolist(), payload={"text": chunk}))
    
    return points

def index_embeddings(points, qdrant_client, collection_name):
    operation_info = qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

    return operation_info


def create_answer_with_context(query, qdrant_client, collection_name):
    embeddings = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embeddings, 
        limit=3
    )

    prompt = """You are a helpful HR assistant who answers 
                questions in brief based on the context below.
                Context:\n"""
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    completion = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


def create_answer_with_context_hf(query, qdrant_client, collection_name):
    # Tokeniza y genera embeddings para la consulta
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embeddings.tolist(), 
        limit=3
    )

    prompt = """You are a helpful HR assistant who answers 
                questions in brief based on the context below.
                Context:\n"""
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    completion = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


def classify_text(text):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    categories = ["Accounting", "Lawyer", "Agriculture", "Textile Industry", "Art", "Automotive", "Aviation", "Banking", "BPO", "Business Development", "Chef", "Construction", "Consultant", "Designer", "Digital - Media", "Engineering", "Finance", "Fitness", "Health", "Human Resources", "Information Technology", "Public Relations", "Sales", "Teacher"] 
    classification = classifier(text, candidate_labels=categories)
    return classification['labels'][0]


def classify_text_pkl(text):
    # Carga el modelo preentrenado
    model_path = './CV-classification/RandomForestClassifier_best_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    # Cargar el vectorizador y el encoder
    word_vectorizer = joblib.load('./CV-classification/word_vectorizer.joblib')
    encoder = joblib.load('./CV-classification/encoder.joblib')

    vectorized_text = word_vectorizer.transform([text])
    
    # Realiza la predicción
    predicted_label = model.predict(vectorized_text)[0]
    
    category = encoder.inverse_transform([predicted_label])[0]
    
    return category


def extract_named_entities_hf(text):
    ner_pipeline = pipeline("ner", model="reyhanemyr/bert-base-NER-finetuned-cv")
    #ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = ner_pipeline(text)
    # Transformar las entidades a un formato común
    transformed_entities = []
    for entity in entities:
        transformed_entities.append({
            "text": text[entity['start']:entity['end']],
            "label": entity['entity']
        })
    return transformed_entities


def extract_named_entities_spacy(text):
    nlp = spacy.load('./content/output/model-best')
    #nlp = spacy.load('./CV-Parsing-using-Spacy-3/output/model-best')
    #nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    tokens = [token.text for token in doc]
    return {"entities": entities, "tokens": tokens}


def generate_summary_hf(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def generate_summary_openai(text):
    try:
        # Crear una solicitud de completación de chat para generar el resumen

        prompt = "Please summarize the following text:\n" + text

        completion = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error al generar el resumen con OpenAI: {e}")
        return None


# APLICACION PRINCIPAL
def main():
    load_dotenv()

    st.title('CV Insights')

    uploaded_file = st.file_uploader("Carga un CV en formato PDF", type="pdf")

    if uploaded_file is not None:
        action = st.radio("¿Qué quieres hacer?", ("", "Hacer preguntas sobre el Curriculum", "Clasificar el Curriculum", "Extracción de Entidades Nombradas", "Resumen del Curriculum"))

        with st.spinner('Extrayendo texto del PDF...'):
            text = extract_and_clean_text_from_pdf(uploaded_file)
        if text:  # Asegura que el texto ha sido extraído correctamente
            st.success("Texto extraído correctamente.")


        if action == "Hacer preguntas sobre el Curriculum":
                # Selecciona el método de generación de embeddings
            embedding_method_options = ["Seleccione una opción...", "OpenAI", "Hugging Face"]
            embedding_method = st.selectbox("Seleccione el método para generar embeddings:", embedding_method_options)

            if embedding_method != embedding_method_options[0]:
                qdrant_client, collection_name = qdrant_connection(embedding_method)

                with st.spinner('Extrayendo texto del PDF...'):
                    text = extract_and_clean_text_from_pdf(uploaded_file)
                st.success("Texto extraído correctamente.")

                chunks = get_text_chunks(text)
                st.write(f"Se han extraído {len(chunks)} fragmentos del texto.")

                start_time = time.time()
                with st.spinner(f'Generando embeddings usando {embedding_method}...'):
                    if embedding_method == "OpenAI":
                        points = generate_embeddings(chunks)
                    else:
                        points = generate_embeddings_hf(chunks)
                end_time = time.time()
                st.success(f"Embeddings generados en {end_time - start_time:.2f} segundos utilizando el método {embedding_method}.")

                with st.spinner('Indexando embeddings en Qdrant...'):
                    index_embeddings(points, qdrant_client, collection_name)
                st.success("Los embeddings se han indexado correctamente.")

                user_question = st.text_input("Haz una pregunta sobre el Curriculum:")
                if user_question:
                    with st.spinner('Buscando respuestas...'):
                        if embedding_method == "OpenAI":
                            answer = create_answer_with_context(user_question, qdrant_client, collection_name)
                        else:
                            answer = create_answer_with_context_hf(user_question, qdrant_client, collection_name)
                    st.write(f"Respuesta: {answer}")

        elif action == "Clasificar el Curriculum":
            classification_method_options = ["Seleccione una opción...", "Zero-shot-classification", "Modelo Preentrenado"]
            classification_method = st.selectbox("Selecciona el método de clasificación:", classification_method_options)

            if classification_method != classification_method_options[0]:  # Asegura que se haya seleccionado una opción válida
                with st.spinner('Clasificando el contenido del documento...'):
                    text = extract_and_clean_text_from_pdf(uploaded_file)
                    
                    if classification_method == "Zero-shot-classification":
                        category = classify_text(text)
                        st.success(f"Categoría principal del documento: {category}")
                    elif classification_method == "Modelo Preentrenado":
                        category = classify_text_pkl(text)
                        st.success(f"Categoría del documento según el modelo preentrenado: {category}")
           
        elif action == "Extracción de Entidades Nombradas":
            ner_method_options = ["Seleccione una opción...", "spaCy", "Hugging Face"]
            ner_method = st.selectbox("Seleccione el método para la extracción de entidades nombradas:", ner_method_options)

            if ner_method != ner_method_options[0]:
                with st.spinner('Extrayendo texto del PDF...'):
                    if ner_method == "Hugging Face":
                        with st.spinner('Extrayendo entidades nombradas con Hugging Face...'):
                            entities = extract_named_entities_hf(text)
                    elif ner_method == "spaCy":
                        with st.spinner('Extrayendo entidades nombradas con spaCy...'):
                            result = extract_named_entities_spacy(text) 
                            entities = result["entities"]

                    if entities:
                        st.write("Entidades encontradas:")
                        for entity in entities:
                            st.write(f"Texto: {entity['text']}, Tipo: {entity['label']}")
                    else:
                        st.write("No se encontraron entidades.")
        
        elif action == "Resumen del Curriculum":
            summary_method_options = ["Seleccione una opción...", "Hugging Face", "OpenAI"]
            summary_method = st.selectbox("Seleccione el método para generar el resumen:", summary_method_options)

            if summary_method != summary_method_options[0]:
                text = extract_and_clean_text_from_pdf(uploaded_file)

                if summary_method == "Hugging Face":
                    summary = generate_summary_hf(text)  
                    st.write("Resumen (Hugging Face):", summary)
                elif summary_method == "OpenAI":
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    summary = generate_summary_openai(text)  
                    st.write("Resumen (OpenAI):", summary)

if __name__ == '__main__':
    main()




# def get_pdf_text(uploaded_file):
#     """Extrae el texto de un archivo PDF cargado como UploadedFile en Streamlit."""
#     text = ""
#     try:
#         # No es necesario abrir el archivo con 'open', PdfReader puede manejar el buffer directamente
#         pdf_reader = PdfReader(uploaded_file)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#         text = re.sub(r'\s+', ' ', text).strip()  # Normaliza y limpia el texto
#     except Exception as e:
#         st.error(f"Error al leer el archivo: {e}")
#     return text