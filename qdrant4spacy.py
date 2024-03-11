import spacy
import streamlit as st
import fitz  # PyMuPDF

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

def extract_named_entities_spacy(text):
    """Extrae las entidades nombradas usando spaCy."""
    # Reemplaza el nombre del modelo por el modelo específico de spaCy que quieras usar
    # Por ejemplo, "en_core_web_sm" para inglés.
    nlp = spacy.load('./CV-Parsing-using-Spacy-3/output/model-best')
    #nlp = spacy.load("en_core_web_sm")  # Asegúrate de tener este modelo descargado
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"entities": entities}

def main():
    st.title('Extracción de Entidades Nombradas con spaCy')

    uploaded_file = st.file_uploader("Carga un CV en formato PDF", type="pdf")
    if uploaded_file is not None:
        with st.spinner('Extrayendo y limpiando texto del PDF...'):
            clean_text = extract_and_clean_text_from_pdf(uploaded_file)
            st.success("Texto procesado correctamente.")

        if clean_text:
            with st.spinner('Extrayendo entidades nombradas con spaCy...'):
                result = extract_named_entities_spacy(clean_text)
                entities = result["entities"]

            if entities:
                st.write("Entidades encontradas:")
                for entity in entities:
                    st.write(f"Texto: {entity['text']}, Tipo: {entity['label']}")
            else:
                st.write("No se encontraron entidades.")

if __name__ == '__main__':
    main()
