from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path
import json
import logging
import asyncio
from functools import wraps

# Add backend path to Python path
BACKEND_PATH = Path(__file__).parent / 'backend'
sys.path.insert(0, str(BACKEND_PATH))

from backend.rag_engine import PostGenerator, PostGenerationConfig, VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_rag():
    """Initialize RAG system."""
    try:
        logger.info("Initializing PostGenerator...")
        
        # Assicurati che la directory del database esista
        db_path = Path("./chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Assicurati che la directory dei dati esista
        data_path = Path("./data")
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Inizializza il generatore
        generator = await PostGenerator.create()
        
        # Verifica se il database è vuoto
        if generator.book_store.is_empty():
            logger.info("Vector store is empty, initializing with PDF...")
            pdf_path = "RAPPORTO VESICA (Ver. 17.01.2025).pdf"
            if await generator.process_pdf_and_initialize(pdf_path):
                logger.info("Successfully initialized vector store with PDF")
            else:
                logger.error("Failed to initialize vector store with PDF")
        else:
            logger.info("Vector store already initialized")
        
        logger.info("PostGenerator initialized successfully.")
        return generator
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

# Initialize Flask app
app = Flask(__name__,
            static_folder='frontend/static',
            template_folder='frontend/templates')

# Initialize the generator
post_generator = None

# Initialize RAG on startup
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    post_generator = loop.run_until_complete(initialize_rag())
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")
finally:
    loop.close()

def async_route(f):
    """Decorator to handle async routes."""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        return await f(*args, **kwargs)
    return wrapper

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@async_route
async def generate_post():
    """Generate a new social media post."""
    try:
        if post_generator is None:
            return jsonify({'error': 'Generator not initialized'}), 500

        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query mancante'}), 400

        query = data['query']
        
        # Generate post using RAG
        result = await post_generator.generate_post(query)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating post: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_posts', methods=['GET'])
def get_posts():
    """Get all saved posts."""
    if post_generator is None:
        return jsonify([])
    return jsonify(post_generator.successful_posts)

@app.route('/process_pdf', methods=['POST'])
@async_route
async def process_pdf():
    """Process a PDF and initialize book knowledge."""
    try:
        if post_generator is None:
            return jsonify({'error': 'Generator not initialized'}), 500

        data = request.get_json()
        if not data or 'pdf_path' not in data:
            return jsonify({'error': 'Percorso PDF mancante'}), 400

        pdf_path = data['pdf_path']
        
        # Chiama il metodo per processare il PDF e inizializzare la conoscenza
        success = await post_generator.process_pdf_and_initialize(pdf_path)
        if success:
            return jsonify({'message': 'PDF processato e conoscenza inizializzata con successo'})
        else:
            return jsonify({'error': 'Errore nel processamento del PDF'}), 500
            
    except Exception as e:
        logger.error(f"Errore nel processamento del PDF: {e}")
        return jsonify({'error': 'Errore interno del server'}), 500

@app.route('/add_post', methods=['POST'])
@async_route
async def add_post():
    """Add a new post."""
    try:
        if post_generator is None:
            return jsonify({'error': 'Generator not initialized'}), 500

        data = request.get_json()
        if not data or 'post' not in data:
            return jsonify({'error': 'Post mancante'}), 400

        success = await post_generator.add_successful_post(data['post'])
        if success:
            await post_generator.save_posts()
            return jsonify({'message': 'Post aggiunto con successo'})
        else:
            return jsonify({'error': 'Errore nell\'aggiunta del post'}), 500
    except Exception as e:
        logger.error(f"Errore nell'aggiunta del post: {e}")
        return jsonify({'error': 'Errore interno del server'}), 500

@app.route('/delete_post', methods=['POST'])
@async_route
async def delete_post():
    """Delete a post."""
    try:
        if post_generator is None:
            return jsonify({'error': 'Generator not initialized'}), 500

        data = request.get_json()
        if not data or 'index' not in data:
            return jsonify({'error': 'Indice mancante'}), 400

        index = int(data['index'])
        if 0 <= index < len(post_generator.successful_posts):
            del post_generator.successful_posts[index]
            await post_generator.save_posts()
            return jsonify({'message': 'Post eliminato con successo'})
        else:
            return jsonify({'error': 'Indice non valido'}), 400
    except Exception as e:
        logger.error(f"Errore nell'eliminazione del post: {e}")
        return jsonify({'error': 'Errore interno del server'}), 500

@app.route('/edit_post', methods=['POST'])
@async_route
async def edit_post():
    """Edit a post."""
    try:
        if post_generator is None:
            return jsonify({'error': 'Generator not initialized'}), 500

        data = request.get_json()
        if not data or 'index' not in data or 'post' not in data:
            return jsonify({'error': 'Dati mancanti'}), 400

        index = int(data['index'])
        if 0 <= index < len(post_generator.successful_posts):
            post_generator.successful_posts[index] = data['post']
            await post_generator.save_posts()
            return jsonify({'message': 'Post modificato con successo'})
        else:
            return jsonify({'error': 'Indice non valido'}), 400
    except Exception as e:
        logger.error(f"Errore nella modifica del post: {e}")
        return jsonify({'error': 'Errore interno del server'}), 500

@app.route('/reset_stores', methods=['POST'])
def reset_stores():
    """Reset vector stores to handle embedding dimension changes."""
    try:
        # Delete existing collections
        if hasattr(post_generator, 'social_store'):
            post_generator.social_store.client.delete_collection("social_media_data")
        if hasattr(post_generator, 'book_store'):
            post_generator.book_store.client.delete_collection("book_knowledge")
        
        # Reinitialize vector stores
        post_generator.social_store = VectorStore("social_media_data")
        post_generator.book_store = VectorStore("book_knowledge")
        
        # Reinitialize data
        asyncio.run(post_generator._initialize_data())
        
        return jsonify({"status": "success", "message": "Vector stores reset and reinitialized successfully"})
    except Exception as e:
        logger.error(f"Error resetting vector stores: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Start Flask app without debug mode to prevent double initialization
    app.run(host='0.0.0.0', port=5001, debug=False)

    # Forza un nuovo deploy su Vercel