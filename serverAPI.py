import os
from openai import OpenAI
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS  # Asegúrate de tener flask-cors instalado (pip install flask-cors)

# ==============================================================================
# SECCIÓN 1: CONFIGURACIONES GLOBALES
# ==============================================================================
VAULT_DIRECTORY = "Vaults"
# ... (El resto de tus configuraciones CONFIG_OLLAMA y CONFIG_OPENAI_VARIABLE no cambian) ...
CONFIG_OLLAMA = {
    "type": "ollama",
    "host": "http://148.206.82.35:11434",
    "chat_model": "mixtral:latest",
    "embedding_model": "bge-m3:latest",
    "api_key": "ollama"
}
CONFIG_OPENAI_VARIABLE = {
    "type": "openai",
    "chat_model": "gpt-4o",
    "embedding_model": "text-embedding-3-small",
    "api_key": "sk-algo"
}

# ==============================================================================
# SECCIÓN 2: LÓGICA RAG (SIN CAMBIOS)
# ==============================================================================
def initialize_vector_store(_vault_file_path, _client, config):
    if not os.path.exists(_vault_file_path):
        raise FileNotFoundError(f"El archivo de Vault seleccionado no existe: {_vault_file_path}")
    if not config or 'embedding_model' not in config:
        raise ValueError("La configuración del modelo de embedding no está cargada.")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    model_name_safe = config['embedding_model'].replace(':', '_').replace('/', '_')
    collection_name = f"vault_Vaults_{os.path.basename(_vault_file_path).replace('.','_')}_{model_name_safe}"
    try:
        collection = db_client.get_collection(name=collection_name)
        print(f"Colección '{collection_name}' cargada exitosamente.")
        return collection
    except ValueError:
         raise ValueError(f"La base de datos vectorial '{collection_name}' no ha sido creada.")
    except Exception as e:
        raise Exception(f"Ocurrió un error inesperado al cargar la colección: {e}")

def get_relevant_context(query, collection, client, config, top_k=3):
    if not collection:
        raise ValueError("La colección de vectores no está disponible.")
    try:
        response = client.embeddings.create(input=query, model=config['embedding_model'])
        results = collection.query(query_embeddings=[response.data[0].embedding], n_results=top_k)
        return results['documents'][0]
    except Exception as e:
        raise Exception(f"Error al consultar ChromaDB: {e}")

# ==============================================================================
# SECCIÓN 3: LÓGICA DE LA API (Flask) - MODIFICADA
# ==============================================================================

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas y métodos (incluyendo POST)

@app.route("/consulta", methods=['POST']) 
def handle_rag_query():
    """
    Este endpoint maneja las peticiones RAG.
    Espera un JSON con: prompt, model_choice, vault_file, history
    """
    try:
        # --- CAMBIO: Leer JSON del cuerpo de la petición ---
        data = request.json
        prompt = data.get("prompt")
        model_choice = data.get("model_choice", "Ollama")
        vault_file = data.get("vault_file")
        history = data.get("history", [])  #<--historial de las conversaciones
        # -------------------------------------------------

        if not all([prompt, model_choice, vault_file]):
            return jsonify({
                "error": "Parámetros faltantes. Se requieren: 'prompt', 'model_choice', y 'vault_file'."
            }), 400
            
        if model_choice == "Ollama":
            config_to_use = CONFIG_OLLAMA
        elif model_choice == "OpenAI":
            config_to_use = CONFIG_OPENAI_VARIABLE
        else:
            return jsonify({"error": "model_choice no válido. Usar 'Ollama' o 'OpenAI'"}), 400

        # ... (La lógica para inicializar el 'client' no cambia) ...
        if config_to_use.get("type") == "openai":
            client = OpenAI(api_key=config_to_use.get("api_key"))
        elif config_to_use.get("type") == "ollama":
            client = OpenAI(
                base_url=f"{config_to_use.get('host')}/v1",
                api_key=config_to_use.get("api_key")
            )
        else:
            raise ValueError("Tipo de configuración no reconocido.")

        # ... (La lógica de RAG: initialize_vector_store y get_relevant_context no cambia) ...
        selected_vault_path = os.path.join(VAULT_DIRECTORY, vault_file)
        rag_collection = initialize_vector_store(
            selected_vault_path, 
            client, 
            config_to_use
        )
        context = get_relevant_context(
            prompt,
            rag_collection,
            client,
            config_to_use
        )
        context_str = "\n\n---\n\n".join(context)

        # Construir el prompt final para el LLM ---
        system_message = (""""
Eres un asistente educacional experto. Tu tarea es responder las consultas del sector alumando.
Basándote únicamente en el 'Contexto Relevante' proporcionado. 
Si la respuesta no está en el contexto, indica que no tienes información al respecto.
Si tienes poca informacion, puedes completar la información, pero debes de tener en cuenta que debes ir guiando la informacion segun el contexto que tengas.
Toma en cuenta la conversación previa para dar respuestas más naturales y fluidas.            
Respuestas solo en español, si utilizas anglicismos pon entre parentesis su traduccion al español.                          
            """
        )
        
        # Este es el prompt final que incluye el contexto RAG
        final_prompt_with_rag = f"""
Eres un profesor asistente. Tu objetivo es ayudar a un estudiante a entender un concepto complejo.
Utiliza el contexto proporcionado para dar una explicación **detallada y clara** sobre la pregunta del estudiante.

**Instrucciones:**
1.  Utiliza **exclusivamente** la información del "Contexto" para formular tu explicación.
2.  Estructura tu respuesta de forma lógica (p.ej., empieza con la definición, luego los puntos clave, y si es posible, un ejemplo del texto).
3.  No añadas información que no esté presente en los documentos.
4.  Si el contexto es insuficiente para una explicación completa, explícalo usando la información disponible y menciona qué partes no se pueden detallar.
5.  La salida puede ser en formato HTML
**Contexto:**
{context_str}

**Pregunta:**
{prompt}

**Explicación Detallada:**
        """
        
        # Construir la lista de mensajes completa
        messages_for_api = [
            {"role": "system", "content": system_message}
        ]
        
        # Añadir el historial de la conversación (si existe)
        messages_for_api.extend(history)
        
        # Añadir la nueva pregunta del usuario (con el contexto RAG)
        messages_for_api.append({"role": "user", "content": final_prompt_with_rag})
        # --------------------------------------------------------

        response = client.chat.completions.create(
            model=config_to_use['chat_model'],
            messages=messages_for_api,
            stream=False 
        )
        
        full_response = response.choices[0].message.content

        output = {
            "answer": full_response,
            "context": context_str
        }
        return jsonify(output)

    except FileNotFoundError as e:
        return jsonify({"error": f"Archivo no encontrado: {str(e)}"}), 404
    except ValueError as e:
        return jsonify({"error": f"Error de valor o configuración: {str(e)}"}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

# Sin indentación
@app.route('/archivos', methods=['GET'])
def listar_archivos():
    try:
        files = os.listdir("Vaults")
        return jsonify(files)

    except Exception as e:
        print(f"Error detallado: {e}")
        return jsonify({'error': 'Error al leer la carpeta', 'detalle': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1080)