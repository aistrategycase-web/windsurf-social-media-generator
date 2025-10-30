import os
import json
import asyncio
import logging
import hashlib
import time
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import datetime

import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import PyPDF2
import re

from backend.prompts import rag_only_prompt_flexible, rag_llm_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def retry_with_exponential_backoff(func, max_retries=3, initial_delay=1):
    """Execute a function with exponential backoff retry logic."""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if "429" in str(e):  # Rate limit error
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(wait_time)
                continue
            raise  # Re-raise if it's not a rate limit error
    
    # If we get here, we've exhausted our retries
    logger.error(f"Failed after {max_retries} retries: {last_exception}")
    raise last_exception

@dataclass
class PostGenerationConfig:
    """Configuration for post generation."""
    max_length: int = 500
    temperature: float = 0.2
    top_k: int = 40
    top_p: float = 0.95
    use_llm_knowledge: bool = True
    num_results: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50

class DocumentProcessor:
    """Processes documents into chunks for RAG system."""

    def __init__(self, config: PostGenerationConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def clean_text(self, text: str) -> str:
        """Cleans text from unsupported characters and removes links."""
        text = re.sub(r'[^\x00-\x7F\xC0-\xFF]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        return text.strip()

    async def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Processes PDF and splits it into chunks."""
        try:
            paragraphs = []
            async with asyncio.Lock():
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)

                    for page_num in range(total_pages):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        text = self.clean_text(text)

                        # Split page text into paragraphs using double newline as delimiter
                        page_paragraphs = text.split("\n\n")

                        for paragraph in page_paragraphs:
                            if paragraph.strip():
                                paragraphs.append({
                                    'content': paragraph,
                                    'metadata': {
                                        "source": pdf_path,
                                        "page": page_num + 1
                                    }
                                })
            logger.info(f"Successfully processed PDF: {len(paragraphs)} paragraphs created")
            return paragraphs

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []

    def calculate_hash(self, file_path: str) -> str:
      """Calculate the SHA256 hash of a file."""
      hasher = hashlib.sha256()
      with open(file_path, 'rb') as file:
        while True:
          chunk = file.read(4096)
          if not chunk:
            break
          hasher.update(chunk)
      return hasher.hexdigest()


class VectorStore:
    """Manages vector database operations."""

    def __init__(self, collection_name: str):
        """Initialize the vector store with a persistent client."""
        try:
            # Crea la directory per il database se non esiste
            db_path = Path("./chroma_db")
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Configura il client persistente con impostazioni specifiche
            settings = chromadb.config.Settings(
                persist_directory=str(db_path),
                anonymized_telemetry=False,
                is_persistent=True
            )
            
            # Crea il client persistente
            self.client = chromadb.PersistentClient(path=str(db_path), settings=settings)
            
            # Crea o recupera la collezione
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Usa la similarità del coseno
            )
            
            logger.info(f"Initialized VectorStore with collection '{collection_name}' at {db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise

    def is_empty(self) -> bool:
        """Check if the collection has any documents."""
        return self.collection.count() == 0

    async def add_documents(self,
                          documents: List[str],
                          embeddings: List[List[float]],
                          metadata: Optional[List[Dict]] = None,
                          batch_size: int = 100):
        """Add documents to the vector store in batches."""
        try:
            if not documents or not embeddings:
                logger.warning("No documents or embeddings to add")
                return

            # Generate unique IDs for each document
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadata if metadata else None
            )
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise


   # 
    def _get_word_variants(self, word: str) -> List[str]:
        """Genera varianti di una parola per la ricerca."""
        word = word.lower().strip()
        variants = {word}  # Usa un set per evitare duplicati
        
        # Varianti comuni
        variants.add(word.replace('gas', 'has'))  # gaslighting -> haslighting
        variants.add(word.replace('has', 'gas'))  # haslighting -> gaslighting
        
        # Rimuovi suffissi comuni
        base_words = {word}
        for suffix in ['ing', 'ed', 'er', 's']:
            if word.endswith(suffix):
                base_words.add(word[:-len(suffix)])
        
        # Aggiungi varianti per ogni parola base
        for base in base_words:
            variants.add(base)
            variants.add(base + 'ing')
            variants.add(base + 'ed')
            variants.add(base + 'er')
            variants.add(base + 's')
        
        return list(variants)

    def _extract_concepts(self, text: str) -> List[str]:
        """Estrae i concetti chiave da un testo."""
        # Rimuovi punteggiatura e converti in minuscolo
        text = text.lower()
        
        # Lista di concetti correlati comuni
        concept_mappings = {
            'pavlov': ['condizionamento', 'campanello', 'cane', 'riflesso', 'stimolo', 'risposta', 'apprendimento'],
            'condizionamento': ['pavlov', 'comportamento', 'stimolo', 'risposta', 'apprendimento', 'psicologia'],
            'manipolazione': ['gaslighting', 'controllo', 'psicologico', 'abuso', 'vittima', 'potere'],
            'gaslighting': ['manipolazione', 'dubbio', 'realtà', 'percezione', 'vittima', 'abuso'],
            'esperimento': ['test', 'ricerca', 'studio', 'risultati', 'prova', 'analisi'],
            'psicologia': ['comportamento', 'mente', 'cognitivo', 'percezione', 'memoria', 'apprendimento']
        }
        
        # Cerca concetti nel testo
        found_concepts = set()
        words = text.split()
        for word in words:
            # Cerca corrispondenze dirette
            if word in concept_mappings:
                found_concepts.add(word)
                # Aggiungi concetti correlati
                found_concepts.update(concept_mappings[word])
            
            # Cerca corrispondenze parziali
            for concept in concept_mappings:
                if concept in word or word in concept:
                    found_concepts.add(concept)
                    found_concepts.update(concept_mappings[concept])
        
        return list(found_concepts)

    def query(self, query_embedding: np.ndarray, n_results: int = 15, keyword_filter: Optional[str] = None) -> Dict:
        """Query the vector store for similar documents."""
        try:
            total_docs = len(self.collection.get()["ids"])
            logger.info(f"Total documents in collection: {total_docs}")
            
            # Estrai parole chiave dalla query
            keywords = set(keyword_filter.lower().split()) if keyword_filter else set()
            logger.info(f"Keywords from query: {keywords}")
            
            # Prima cerca usando l'embedding
            initial_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results * 3, total_docs),  # Cerca più risultati inizialmente
                include=["documents", "distances", "metadatas"]  # Include i metadati
            )
            
            documents = initial_results["documents"][0] if initial_results["documents"] else []
            scores = initial_results["distances"][0] if "distances" in initial_results else []
            metadatas = initial_results["metadatas"][0] if "metadatas" in initial_results else []
            
            # Se abbiamo documenti e parole chiave
            if documents and keywords:
                filtered_results = []
                filtered_scores = []
                filtered_metadatas = []
                
                for doc, score, metadata in zip(documents, scores, metadatas):
                    doc_lower = doc.lower()
                    
                    # Calcola quante parole chiave sono presenti nel documento
                    keyword_matches = sum(1 for keyword in keywords if keyword in doc_lower)
                    
                    # Calcola un boost basato sulle parole chiave trovate
                    keyword_boost = 1.0 + (keyword_matches * 0.3)  # 30% boost per ogni parola chiave
                    
                    # Applica il boost al punteggio
                    adjusted_score = score / keyword_boost  # Dividi per il boost perché scores più bassi sono migliori
                    
                    # Se c'è almeno una parola chiave o il punteggio è molto buono
                    if keyword_matches > 0 or score < 0.3:  # Soglia più stretta per la similarità
                        filtered_results.append(doc)
                        filtered_scores.append(adjusted_score)
                        filtered_metadatas.append(metadata)
                        
                        # Log per debug
                        preview = doc[:150] + "..." if len(doc) > 150 else doc
                        logger.info(f"Including document (score: {adjusted_score:.4f}, keywords: {keyword_matches}):\n{preview}\n")
                
                # Ordina per punteggio
                sorted_triples = sorted(zip(filtered_results, filtered_scores, filtered_metadatas), key=lambda x: x[1])
                documents = [triple[0] for triple in sorted_triples[:n_results]]
                scores = [triple[1] for triple in sorted_triples[:n_results]]
                metadatas = [triple[2] for triple in sorted_triples[:n_results]]
                
                logger.info(f"Found {len(documents)} relevant documents")
            
            return {
                "documents": documents,
                "scores": scores,
                "metadatas": metadatas
            }
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "documents": [],
                "scores": [],
                "metadatas": []
            }

@lru_cache(maxsize=100)
def cache_gemini_response(prompt: str, temp: float = 0.1) -> str:
    """Cache Gemini API responses to reduce API calls."""
    # Nota: questa è una funzione sincrona che verrà chiamata dalla versione asincrona
    model = genai.GenerativeModel("gemini-pro")
    
    # Impostiamo la safety direttamente sul modello
    model.safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temp,
            max_output_tokens=300
        )
    )
    return response.text.strip()

async def async_cache_gemini_response(prompt: str, temp: float = 0.1) -> str:
    """Versione asincrona della cache per Gemini."""
    loop = asyncio.get_event_loop()
    try:
        def _generate_content():
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=800,
                    top_p=0.95,
                    top_k=40
                ),
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )
            
            if not response.candidates:
                # Se non ci sono candidati, prova a dividere il contenuto e riformularlo in parti
                chunks = [prompt[i:i+1000] for i in range(0, len(prompt), 1000)]
                reformulated_chunks = []
                
                for chunk in chunks:
                    try:
                        chunk_response = model.generate_content(
                            f"Riformula il seguente testo mantenendo il significato originale: {chunk}",
                            safety_settings=[{"category": cat, "threshold": "BLOCK_NONE"} 
                                          for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                                                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                        )
                        if chunk_response.candidates:
                            reformulated_chunks.append(chunk_response.text)
                    except Exception as e:
                        logger.error(f"Errore nella riformulazione del chunk: {e}")
                        reformulated_chunks.append(chunk)  # Usa il testo originale se la riformulazione fallisce
                
                return " ".join(reformulated_chunks)
            
            return response.text.strip()
            
        return await loop.run_in_executor(None, _generate_content)
    except Exception as e:
        logger.error(f"Errore in async_cache_gemini_response: {e}")
        # In caso di errore, restituisci il testo originale dal contesto
        original_text = prompt.split("**CONTESTO:**")[-1].split("**ARGOMENTO:**")[0].strip()
        return f"Non è stato possibile riformulare il testo. Ecco il contenuto originale:\n\n{original_text}"

class PostGenerator:
    """Main class for generating social media posts using RAG."""

    def __init__(self, config: Optional[PostGenerationConfig] = None):
        """Regular initialization of instance variables."""
        self.config = config or PostGenerationConfig()
        self._embedding_model = None # Initialize with None
        self.embedding_model_name = "all-roberta-large-v1"  # Modello più recente e performante

        # Initialize data paths
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.config_path = self.data_dir / "rag_config.json"
        self.style_guide_path = self.data_dir / "style_guide.json"
        self.posts_path = self.data_dir / "successful_posts.json"
        self.rag_config = self._load_rag_config()

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-pro")
        self.doc_processor = DocumentProcessor(self.config)

        # Initialize vector stores without forcing recreation
        self.social_store = VectorStore("social_media_data")
        self.book_store = VectorStore("book_knowledge")

        self.successful_posts = []
        logger.info(f"Loading SentenceTransformer model: {self.embedding_model_name}...")

    async def save_posts(self):
        """Save posts to JSON file."""
        try:
            # Pulisci i post prima di salvarli
            clean_posts = []
            for post in self.successful_posts:
                if post and isinstance(post, (str, int, float)):
                    clean_post = str(post).strip()
                    if clean_post:
                        clean_posts.append(clean_post)

            # Crea un backup prima di salvare
            if self.posts_path.exists():
                backup_path = self.posts_path.with_suffix('.json.bak')
                try:
                    import shutil
                    shutil.copy2(self.posts_path, backup_path)
                except Exception as e:
                    logger.error(f"Failed to create backup: {e}")

            # Salva i post puliti
            with open(self.posts_path, 'w', encoding='utf-8') as f:
                json.dump(clean_posts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved {len(clean_posts)} posts")
            return True
        except Exception as e:
            logger.error(f"Error saving posts: {e}")
            return False

    async def add_successful_post(self, post: str) -> bool:
        """Add a new successful post to storage."""
        try:
            if post and isinstance(post, str):
                clean_post = post.strip()
                if clean_post:
                    # Evita duplicati
                    if clean_post not in self.successful_posts:
                        self.successful_posts.append(clean_post)
                        await self.save_posts()  # Salva subito dopo l'aggiunta
                        return True
            return False
        except Exception as e:
            logger.error(f"Error adding post: {e}")
            return False

    @property
    def embedding_model(self):
        """Lazy initialization of the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading SentenceTransformer model: {self.embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("SentenceTransformer model loaded.")
        return self._embedding_model

    async def process_pdf_and_initialize(self, pdf_path: str) -> bool:
        """Process PDF and initialize book knowledge."""
        logger.info(f"Starting process_pdf_and_initialize for PDF: {pdf_path}")
        try:
            # Reinitialize the collection
            logger.info("Clearing book_knowledge collection...")
            try:
                self.book_store.client.delete_collection(name=self.book_store.collection.name)
                logger.info("book_knowledge collection deleted (if it existed).")
            except Exception as e:
                logger.info(f"book_knowledge collection not found, no action was needed {e}")
            
            # Reinitialize the collection
            self.book_store = VectorStore("book_knowledge")
            logger.info("book_knowledge collection re-initialized.")

            # Process the PDF and store chunks
            success = await self.process_and_store_pdf(pdf_path)
            logger.info(f"process_and_store_pdf returned: {success}")
            
            if success:
                # Update the configuration with new PDF hash
                pdf_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()
                self.rag_config = {
                    'pdf_hash': pdf_hash,
                    'last_processed': str(Path(pdf_path).stat().st_mtime)
                }
                
                # Save config to file
                with open(self.config_path, 'w') as f:
                    json.dump(self.rag_config, f)
                
            return success
        except Exception as e:
            logger.error(f"Error processing PDF and initializing knowledge: {e}")
            return False

    async def process_and_store_pdf(self, pdf_path: str) -> bool:
        """Process a PDF file and store its contents in the vector store."""
        logger.info(f"Starting process_and_store_pdf for PDF: {pdf_path}")
        
        try:
            # Process PDF into chunks
            logger.info(f"Calling doc_processor.process_pdf with path: {pdf_path}")
            documents = await self.doc_processor.process_pdf(pdf_path)
            logger.info(f"doc_processor.process_pdf returned {len(documents)} chunks.")
            
            if not documents:
                logger.error("No documents were extracted from the PDF")
                return False
            
            # Generate embeddings for all documents
            logger.info(f"Generating embeddings for {len(documents)} documents.")
            embeddings = await self.generate_embeddings(documents)
            
            if not embeddings or len(embeddings) != len(documents):
                logger.error(f"Embedding generation failed. Got {len(embeddings) if embeddings else 0} embeddings for {len(documents)} documents")
                return False
            
            # Store documents in batches to avoid memory issues
            batch_size = 100
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch_docs = [doc['content'] for doc in documents[i:i+batch_size]]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadata = [doc['metadata'] for doc in documents[i:i+batch_size]]
                
                # Generate unique IDs for this batch
                batch_ids = [f"doc_{i+j}" for j in range(len(batch_docs))]
                
                try:
                    # Add documents to collection
                    self.book_store.collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    logger.info(f"Successfully added batch {i//batch_size + 1}/{total_batches} to collection")
                except Exception as e:
                    logger.error(f"Error adding batch to collection: {str(e)}")
                    return False
            
            # Verify documents were added
            total_docs = len(self.book_store.collection.get()["ids"])
            logger.info(f"Total documents in collection after processing: {total_docs}")
            
            if total_docs != len(documents):
                logger.error(f"Document count mismatch. Expected {len(documents)}, got {total_docs}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in process_and_store_pdf: {str(e)}")
            return False

    async def generate_embeddings(self, documents: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates embeddings for a list of documents, using batching."""
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            # Convert numpy array to list of lists
            batch_embeddings = batch_embeddings.tolist()
            embeddings.extend(batch_embeddings)
        return embeddings

    @staticmethod
    async def expand_query_with_llm(query: str) -> str:
        """Expands the user query using Gemini Pro to include related concepts."""
        gen_model = genai.GenerativeModel("gemini-pro") 

        prompt = f"""
        Espandi la seguente query di ricerca per includere concetti correlati, sinonimi e parole chiave associate. 
        La query è: {query}
        
        Restituisci un elenco di termini e frasi correlate, uno per riga.
        Esempio per la query "machine learning":
        machine learning
        deep learning
        neural networks
        artificial intelligence
        pattern recognition
        data mining
        supervised learning
        unsupervised learning
        """
        
        async def _generate():
            response = gen_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for factual extraction
                    max_output_tokens=300
                )
            )
            return response.text.strip()
        
        try:
            expanded_query = await retry_with_exponential_backoff(
                _generate
            )
            
            if isinstance(expanded_query, str):
                return expanded_query
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query  # Return original query if expansion fails

    def _load_style_guide(self) -> dict:
        """Load style guide with error handling."""
        try:
            if not self.style_guide_path.exists():
                logger.warning("Style guide file not found, using defaults")
                return {
                    "tone": "Divertente e informale",
                    "hashtags": "#windsurf #windsurfing",
                    "structure": "Domanda iniziale + punti chiave + call-to-action"
                }
            
            return json.loads(self.style_guide_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading style guide: {e}")
            return {
                "tone": "Divertente e informale",
                "hashtags": "#windsurf #windsurfing",
                "structure": "Domanda iniziale + punti chiave + call-to-action"
            }

    def _load_json(self, path: Path) -> List[str]:
        """Load JSON data with error handling and cleaning."""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # File vuoto
                        return []
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            # Pulisci i post
                            return [str(post).strip() for post in data if post and isinstance(post, (str, int, float))]
                    except json.JSONDecodeError:
                        # Se il file è corrotto, prova a recuperare i post uno per uno
                        posts = []
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith(('[', ']', '{', '}')):
                                posts.append(line)
                        return posts
            return []
        except Exception as e:
            logger.error(f"Error loading JSON from {path}: {e}")
            # Fai un backup del file corrotto
            if path.exists():
                backup_path = path.with_suffix('.json.bak')
                try:
                    import shutil
                    shutil.copy2(path, backup_path)
                    logger.info(f"Created backup of corrupted file at {backup_path}")
                except Exception as be:
                    logger.error(f"Failed to create backup: {be}")
            return []

    def _load_rag_config(self) -> Dict:
        """Loads RAG configuration from file or returns default if not found."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return {"pdf_hash": None, "embedding_model": self.embedding_model_name}
        except (FileNotFoundError, json.JSONDecodeError):
            return {"pdf_hash": None, "embedding_model": self.embedding_model_name}

    def _save_rag_config(self, pdf_hash: str):
        """Saves RAG configuration to file."""
        config_data = {
            "pdf_hash": pdf_hash,
            "embedding_model": self.embedding_model_name,
            "last_processed": str(datetime.datetime.now())
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    async def _initialize_data(self) -> None:
        """Initialize data by processing PDF and storing in vector stores."""
        logger.info("Initializing data...")
        pdf_path = "RAPPORTO VESICA (Ver. 17.01.2025).pdf"
        
        try:
            # Calculate PDF hash
            pdf_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()
            
            # Process and store PDF
            success = await self.process_and_store_pdf(pdf_path)
            
            if success:
                # Save configuration with PDF hash
                self._save_rag_config(pdf_hash)
                logger.info("Data initialization completed successfully")
            else:
                logger.error("Failed to initialize data")
        except Exception as e:
            logger.error(f"Error in _initialize_data: {str(e)}")

    @classmethod
    async def create(cls, config: Optional[PostGenerationConfig] = None):
        """Factory method to create and initialize a PostGenerator instance."""
        instance = cls(config)
        
        try:
            # Initialize embedding model
            logger.info(f"Loading SentenceTransformer model: {instance.embedding_model_name}...")
            instance._embedding_model = SentenceTransformer(instance.embedding_model_name)
            logger.info("SentenceTransformer model loaded.")

            # Load successful posts
            instance.successful_posts = instance._load_json(instance.posts_path)
            logger.info(f"Loaded {len(instance.successful_posts)} saved posts")

            # Check if we need to initialize vector stores
            if await instance._is_initialization_needed():
                logger.info("Initializing vector stores...")
                await instance._initialize_data()
            else:
                total_docs = len(instance.book_store.collection.get()['ids'])
                logger.info(f"Vector stores are already initialized with {total_docs} documents")

            return instance
            
        except Exception as e:
            logger.error(f"Error creating PostGenerator: {e}")
            raise

    async def _is_initialization_needed(self) -> bool:
        """Check if we need to initialize the vector stores."""
        try:
            if not self.book_store.collection.get()['ids']:
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking initialization status: {e}")
            return True

    async def generate_post(self, query: str, config: Optional[PostGenerationConfig] = None) -> Dict:
        """Generate a social media post using RAG."""
        config = config or self.config
        try:
            # Load style guide
            style_guide = self._load_style_guide()

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Get relevant passages
            results = self.book_store.query(query_embedding, n_results=15, keyword_filter=query)
            
            if not results:
                return {
                    "error": "Nessun contenuto rilevante trovato per la query",
                    "chunks": [],
                    "sources": []
                }

            # Format context and prepare chunks
            chunks = []
            sources = set()
            
            for doc, metadata in zip(results['documents'], results.get('metadatas', [])):
                source = metadata.get('source', 'Documento sconosciuto')
                page = metadata.get('page', 'N/A')
                chunks.append({
                    'text': doc,
                    'source': source,
                    'page': page
                })
                sources.add(source)

            # Format context for prompt
            context = "\n\n".join(doc['text'] for doc in chunks)
            
            # Generate post
            prompt = rag_only_prompt_flexible.format(
                style_guide_tone=style_guide.get("tone", "professionale e informativo"),
                style_guide_hashtags=style_guide.get("hashtags", "#informazione #consapevolezza"),
                book_context=context,
                query=query
            )

            try:
                post = await async_cache_gemini_response(prompt)
                if post:
                    return {
                        "post": post,
                        "chunks": chunks,
                        "sources": list(sources)
                    }
                else:
                    return {
                        "error": "Il contenuto è stato rifiutato per motivi di sicurezza. Per favore, prova con un argomento diverso o riformula la richiesta in modo più professionale.",
                        "chunks": chunks,
                        "sources": list(sources)
                    }
            except Exception as e:
                if "safety_ratings" in str(e) and "HIGH" in str(e):
                    return {
                        "error": "Il contenuto è stato rifiutato per motivi di sicurezza. Per favore, prova con un argomento diverso o riformula la richiesta in modo più professionale.",
                        "chunks": chunks,
                        "sources": list(sources)
                    }
                raise

        except Exception as e:
            logger.error(f"Error generating post: {e}")
            return {
                "error": f"Errore durante la generazione del post: {str(e)}",
                "chunks": [],
                "sources": []
            }

    def _split_into_batches(self, items: List[str], batch_size: int = 3) -> List[List[str]]:
        """Divide una lista di items in batch della dimensione specificata."""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    async def _process_batches(self, query: str, batches: List[List[str]], scores: List[float] = None) -> str:
        """Processa i batch di documenti e combina i risultati."""
        all_key_points = []
        total_batches = len(batches)
        
        for i, batch in enumerate(batches):
            batch_weight = scores[i] if scores else 1.0
            logger.info(f"Processing batch {i+1}/{total_batches}")
            logger.info(f"Batch {i+1} weight: {batch_weight:.4f}")
            
            # Log batch content preview
            logger.info("Batch content preview:")
            for j, doc in enumerate(batch):
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                logger.info(f"Document {j+1}: {preview}")
            
            try:
                # Estrai prima le informazioni chiave dal batch
                key_info = self._extract_key_info(batch)
                if not key_info:
                    continue
                
                # Prova a generare contenuto sicuro usando solo le informazioni chiave
                safe_prompt = f"""
                Genera un post educativo e costruttivo sui seguenti concetti, mantenendo un tono professionale e rispettoso:

                CONCETTI CHIAVE:
                {key_info}

                LINEE GUIDA:
                - Concentrati sugli aspetti educativi e di consapevolezza
                - Evita riferimenti diretti a comportamenti dannosi
                - Mantieni un tono costruttivo e orientato alla soluzione
                - Offri prospettive utili per la crescita personale
                - Non includere esempi specifici di manipolazione
                
                QUERY: {query}
                """
                
                try:
                    response = await retry_with_exponential_backoff(
                        lambda: async_cache_gemini_response(safe_prompt, temp=0.1)
                    )
                    if isinstance(response, str):
                        all_key_points.append(response)
                        continue
                except Exception as e:
                    logger.error(f"Error generating safe content: {e}")
                
                # Se fallisce, usa direttamente le informazioni chiave
                all_key_points.append(f"Informazione educativa: {key_info}")
                    
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                continue
        
        logger.info(f"Total key points extracted: {len(all_key_points)} batches")
        return all_key_points

    def _extract_key_info(self, batch: List[str]) -> str:
        """Estrae le informazioni chiave da un batch di documenti in modo sicuro."""
        try:
            # Unisci i documenti
            combined_text = " ".join(batch)
            
            # Cerca definizioni e concetti chiave
            key_info = []
            
            # Estrai definizioni in modo sicuro
            sentences = combined_text.split(". ")
            for sentence in sentences:
                # Rimuovi riferimenti diretti a comportamenti dannosi
                clean_sentence = self._sanitize_content(sentence)
                if clean_sentence:
                    key_info.append(clean_sentence)
            
            if key_info:
                # Prendi solo le prime 2-3 frasi più rilevanti e sicure
                return ". ".join(key_info[:3])
            
            return None
                
        except Exception as e:
            logger.error(f"Error extracting key info: {e}")
            return None
            
    def _sanitize_content(self, text: str) -> Optional[str]:
        """Pulisce il contenuto da riferimenti potenzialmente problematici."""
        # Lista di parole chiave educative e costruttive
        educational_keywords = ["consapevolezza", "comprensione", "riconoscere", "imparare", 
                              "crescita", "sviluppo", "benessere", "comunicazione"]
                              
        # Rimuovi riferimenti diretti a comportamenti dannosi
        text = text.lower()
        
        # Verifica se la frase contiene almeno una parola chiave educativa
        if any(keyword in text for keyword in educational_keywords):
            # Riformula in modo costruttivo
            text = text.replace("manipolazione", "dinamica relazionale")
            text = text.replace("vittima", "persona")
            text = text.replace("abuso", "comportamento")
            text = text.replace("violenza", "situazione")
            
            # Capitalizza la prima lettera
            return text[0].upper() + text[1:]
            
        return None

    @staticmethod
    def _clean_post(post: str) -> str:
        """Clean post content from HTML and UI elements."""
        # Remove common UI text
        ui_elements = [
            "Modifica",
            "Elimina",
            "                        ",
            "                            "
        ]
        cleaned = post
        for element in ui_elements:
            cleaned = cleaned.replace(element, "")

        # Remove extra whitespace and newlines
        cleaned = " ".join(cleaned.split())

        return cleaned.strip()