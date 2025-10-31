import streamlit as st
import os
import sys
import json
import time
import re
from openai import OpenAI
import PyPDF2
import chromadb

# --- Configuración de la Página ---
st.set_page_config(page_title="RAG-App", layout="wide", page_icon="🤖")

# ==============================================================================
# SECCIÓN 1: CONFIGURACIONES GLOBALES
# ==============================================================================

# Directorio para guardar los archivos de texto procesados
VAULT_DIRECTORY = "Vaults"

CONFIG_OLLAMA = {
    "type": "ollama",
    "host": "http://148.206.82.35:11434",
    "chat_model": "mixtral:latest",
    "embedding_model": "bge-m3:latest",
    "api_key": "ollama"
}

CONFIG_OPENAI = {
    "type": "openai",
    "chat_model": "gpt-4o",
    "embedding_model": "text-embedding-3-small",
    "api_key": None # Será solicitado al usuario
}

# ==============================================================================
# SECCIÓN 2: LÓGICA DE PROCESAMIENTO DE ARCHIVOS (Sin cambios)
# ==============================================================================

@st.cache_data
def chunk_text(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    max_chunk_size = 1000
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence if current_chunk else sentence)
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_file_logic(file_path: str, output_file_handle):
    text_content = ""
    try:
        if file_path.lower().endswith(".pdf"):
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content += extracted + " "
        elif file_path.lower().endswith(".txt"):
            with open(file_path, 'r', encoding="utf-8") as txt_file:
                text_content = txt_file.read()
        elif file_path.lower().endswith(".json"):
            with open(file_path, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
                text_content = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            st.warning(f"Archivo omitido (extensión no soportada): {os.path.basename(file_path)}")
            return False
        if not text_content.strip():
            st.warning(f"Archivo omitido (no se pudo extraer contenido): {os.path.basename(file_path)}")
            return False
        chunks = chunk_text(text_content)
        for chunk in chunks:
            output_file_handle.write(chunk + "\n")
        return True
    except Exception as e:
        st.error(f"Error al procesar el archivo {file_path}: {e}")
        return False

# ==============================================================================
# SECCIÓN 3: LÓGICA RAG (SIMPLIFICADA)
# ==============================================================================

def open_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
    except Exception as e:
        st.error(f"Error al leer el archivo vault: {e}")
        return None

@st.cache_resource(ttl=3600)
def initialize_vector_store(_vault_file_path, vault_content, _client, config):
    """
    Inicializa y carga una colección de ChromaDB. Si está vacía, genera los embeddings
    usando la API configurada (Ollama u OpenAI) y los guarda.
    """
    db_client = chromadb.PersistentClient(path="./chroma_db")
    
    base_name = _vault_file_path.replace('.', '_').replace(os.sep, '_')
    model_name_safe = config['embedding_model'].replace(':', '_').replace('/', '_')
    collection_name = f"vault_{base_name}_{model_name_safe}"
    
    collection = db_client.get_or_create_collection(name=collection_name)
    
    if collection.count() == 0:
        st.info(f"Colección '{collection_name}' vacía. Generando y guardando embeddings...")
        with st.spinner(f"Usando el modelo de embeddings '{config['embedding_model']}'..."):
            documents_list = [c.strip() for c in vault_content if c.strip()]
            ids_list = [f"doc_{base_name}_line_{i}" for i in range(len(documents_list))]
            embeddings_list = []
            
            embedding_model_name = config['embedding_model']
            
            st.info(f"Generando embeddings con API ({config['type']})...")
            for i, content in enumerate(documents_list):
                try:
                    response = _client.embeddings.create(input=content, model=embedding_model_name)
                    embeddings_list.append(response.data[0].embedding)
                except Exception as e:
                    st.error(f"Error generando embedding para el fragmento {i}: {e}")
                    # Añadimos un embedding vacío para mantener la consistencia de los arrays
                    # La dimensionalidad de nomic-embed-text es 768.
                    embeddings_list.append([0.0] * 768) 

            if ids_list and embeddings_list and len(ids_list) == len(embeddings_list):
                collection.add(embeddings=embeddings_list, documents=documents_list, ids=ids_list)
                st.success(f"{len(ids_list)} embeddings guardados en ChromaDB (Colección: {collection_name})")
            else:
                st.warning("No se generaron embeddings o hubo un error de consistencia.")
    else:
        st.success(f"Colección de ChromaDB '{collection_name}' cargada ({collection.count()} items).")
        
    return collection

def get_relevant_context(query, collection, client, config, top_k=3):
    """
    Genera un embedding para la consulta y busca los documentos más relevantes en ChromaDB.
    """
    try:
        embedding_model_name = config['embedding_model']
        
        response = client.embeddings.create(input=query, model=embedding_model_name)
        input_embedding = response.data[0].embedding
        
        if not input_embedding:
            st.error("No se pudo generar el embedding para la consulta.")
            return []

        query_results = collection.query(query_embeddings=[input_embedding], n_results=top_k)
        return query_results['documents'][0]
    except Exception as e:
        st.error(f"Error al consultar ChromaDB: {e}")
        return []

def rewrite_query(user_input, conversation_history, client, chat_model):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Reescriba la siguiente consulta incorporando el contexto pertinente proveniente del historial de conversación.
La consulta reescrita debe:
- Conservar la intención principal y el significado del planteamiento original
- Ampliar y precisar la consulta para que sea más específica e informativa
- NO RESPONDER JAMÁS la consulta original, sino centrarse exclusivamente en reformularla con el fin de evitar ambiguedades.
Devuelva ÚNICAMENTE el texto de la consulta reescrita.
Historial de conversación:
{context}
Consulta original: [{user_input}]
Consulta reescrita:
"""
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200, n=1, temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"No se pudo reescribir la consulta, usando la original. Error: {e}")
        return user_input

# ==============================================================================
# SECCIÓN 4: INTERFAZ DE STREAMLIT (SIMPLIFICADA)
# ==============================================================================

st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.radio("Selecciona una sección:", ("Inicio", "Gestión de Archivos", "Chatbot RAG"))

st.sidebar.title("Configuración del Modelo")
# MODIFICADO: Solo dos opciones
model_choice = st.sidebar.selectbox(
    "Elige tu proveedor de IA:",
    ("Ollama", "OpenAI"),
    index=0
)

if "current_config" not in st.session_state:
    st.session_state.current_config = {}

def test_ollama_connection(config):
    if 'ollama_tested' not in st.session_state or not st.session_state.ollama_tested:
        try:
            client_test = OpenAI(base_url=config['host'] + "/v1", api_key=config['api_key'])
            client_test.models.list()
            st.sidebar.success("Conexión con servidor Ollama exitosa.")
            st.session_state.ollama_tested = True
        except Exception as e:
            st.sidebar.error(f"No se pudo conectar a Ollama en {config['host']}.")
            st.session_state.ollama_tested = False
            st.stop()

if model_choice == "Ollama":
    st.session_state.current_config = CONFIG_OLLAMA
    st.sidebar.subheader("Configuración de Ollama")
    st.sidebar.json(CONFIG_OLLAMA)
    test_ollama_connection(CONFIG_OLLAMA)

elif model_choice == "OpenAI":
    st.sidebar.subheader("Configuración de OpenAI")
    api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
    config_gpt = CONFIG_OPENAI.copy()
    config_gpt["api_key"] = api_key_input
    st.session_state.current_config = config_gpt
    if not api_key_input:
        st.sidebar.warning("Por favor, introduce tu API Key de OpenAI.")
    else:
        st.sidebar.success("API Key cargada.")
    st.session_state.ollama_tested = False

# ==============================================================================
# SECCIÓN 5: PÁGINAS DE LA APLICACIÓN (MODIFICADAS)
# ==============================================================================

if opcion == "Inicio":
    st.title("Bienvenido a tu Aplicación RAG con Streamlit 🚀")
    st.write("Esta interfaz te permite gestionar tus documentos y chatear con ellos usando diferentes modelos de IA.")
    st.subheader("Pasos a seguir:")
    st.markdown(f"""
    1.  **Configura tu Modelo**: En la barra lateral, elige tu proveedor.
        * `Ollama`: Usa tu servidor Ollama para todo (chat y embeddings).
        * `OpenAI`: Usa la API de OpenAI para todo (requiere API Key).
    2.  **Gestiona tus Archivos**: Ve a `Gestión de Archivos` para subir tus documentos y crear un "Vault". Los archivos de texto se guardarán en la carpeta `{VAULT_DIRECTORY}`.
    3.  **Chatea**: Ve al `Chatbot RAG`, selecciona el "Vault" que creaste, ¡y empieza a hacer preguntas!
    """)

elif opcion == "Gestión de Archivos":
    st.title("Gestión de Archivos y Vaults 📂")
    st.write(f"Sube tus archivos (PDF, TXT, JSON) para consolidarlos en un único archivo 'Vault' de texto. Los archivos se guardarán en la carpeta **'{VAULT_DIRECTORY}'**.")
    
    # MODIFICADO: Asegurarse de que el directorio Vaults exista
    os.makedirs(VAULT_DIRECTORY, exist_ok=True)
    
    vault_name = st.text_input("Nombre del Vault (ej: 'documentos_legales', 'proyecto_tesis'):")
    uploaded_files = st.file_uploader("Elige tus archivos...", type=["pdf", "txt", "json"], accept_multiple_files=True)
    
    if st.button("Procesar Archivos y Crear/Actualizar Vault"):
        if not vault_name: st.error("Por favor, asigna un nombre al Vault.")
        elif not uploaded_files: st.error("Por favor, sube al menos un archivo.")
        else:
            # MODIFICADO: Construir la ruta del archivo de salida dentro de la carpeta Vaults
            file_name = f"{vault_name.strip()}.txt"
            output_filename = os.path.join(VAULT_DIRECTORY, file_name)
            
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            file_paths = [os.path.join(temp_dir, f.name) for f in uploaded_files]
            for i, f in enumerate(uploaded_files):
                with open(file_paths[i], "wb") as temp_f: temp_f.write(f.getbuffer())
            
            st.info(f"Iniciando procesamiento... El resultado se guardará en '{output_filename}'")
            try:
                with open(output_filename, "w", encoding="utf-8") as vault_file:
                    with st.spinner("Procesando archivos..."):
                        success_count = sum(1 for path in file_paths if process_file_logic(path, vault_file))
                    st.success(f"¡Proceso completado! {success_count}/{len(file_paths)} archivos procesados y guardados en '{output_filename}'.")
            except IOError as e: st.error(f"Error: No se pudo escribir en el archivo '{output_filename}': {e}")
            finally:
                for path in file_paths: os.remove(path)
                os.rmdir(temp_dir)

elif opcion == "Chatbot RAG":
    st.title(f"Chatbot RAG 🤖 (Modo: {model_choice})")

    if st.sidebar.button("Limpiar Historial de Chat"):
        if "messages" in st.session_state: st.session_state.messages = []
        st.rerun()

    config = st.session_state.current_config
    if not config or (config['type'] == 'openai' and not config.get("api_key")):
        st.warning("Por favor, configura tu modelo y/o introduce tu API Key en la barra lateral.")
        st.stop()
    
    # MODIFICADO: Buscar archivos .txt dentro de la carpeta Vaults
    os.makedirs(VAULT_DIRECTORY, exist_ok=True) # Asegurarse de que exista
    try:
        vault_files = [f for f in os.listdir(VAULT_DIRECTORY) if f.endswith(".txt")]
        if not vault_files:
            st.error(f"No se encontró ningún archivo 'Vault' (.txt) en la carpeta '{VAULT_DIRECTORY}'. Ve a 'Gestión de Archivos' para crear uno.")
            st.stop()
        
        selected_vault_filename = st.selectbox("Selecciona el Vault a consultar:", vault_files)
        # Construir la ruta completa al archivo seleccionado
        selected_vault_path = os.path.join(VAULT_DIRECTORY, selected_vault_filename)

    except Exception as e:
        st.error(f"Error al listar archivos Vault desde la carpeta '{VAULT_DIRECTORY}': {e}")
        st.stop()

    try:
        if "host" in config: # Configuración de Ollama
            client = OpenAI(base_url=f"{config['host']}/v1", api_key=config['api_key'])
        else: # Configuración de OpenAI
            client = OpenAI(api_key=config['api_key'])

        # Usar la ruta completa para abrir el archivo
        vault_content_full = open_file(selected_vault_path)
        if not vault_content_full: st.error("No se pudo leer el archivo Vault."); st.stop()
        
        vault_content_lines = vault_content_full.splitlines()

        st.session_state.rag_collection = initialize_vector_store(
            selected_vault_path, vault_content_lines, client, config
        )
        st.session_state.client = client
    except Exception as e:
        st.error(f"Error al inicializar el backend RAG: {e}")
        st.stop()

    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Haz una consulta sobre tus documentos..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                client = st.session_state.client
                chat_model = config['chat_model']

                rewritten_query = rewrite_query(prompt, st.session_state.messages, client, chat_model)
                st.info(f"Consulta reescrita: *{rewritten_query}*")

                context = get_relevant_context(
                    rewritten_query, st.session_state.rag_collection, client, config
                )
                
                context_str = "\n".join(context)
                with st.expander("Contexto recuperado"):
                    st.code(context_str if context_str else "No se encontró contexto relevante.")

                system_message = "Eres un asistente experto. Responde la consulta del usuario basándote únicamente en el 'Contexto Relevante' proporcionado."
                final_prompt_with_context = f"Contexto Relevante:\n{context_str}\n\nConsulta del Usuario:\n{prompt}"
                
                messages_for_api = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": final_prompt_with_context}
                ]
                
                try:
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=messages_for_api,
                        max_tokens=2000,
                    )
                    response_content = response.choices[0].message.content
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                except Exception as e:
                    st.error(f"Error al contactar la API de Chat: {e}")

