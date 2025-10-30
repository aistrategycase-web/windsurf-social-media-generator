import logging
from backend.document_processor import DocumentProcessor
from backend.rag_engine import PostGenerator
# Configurazione logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Inizializza il processor e il generator
doc_processor = DocumentProcessor()
generator = PostGenerator()

# Processa il PDF
pdf_path = "/Users/alessiocavatassi/windsurf-social-rag/RAPPORTO VESICA (Ver. 17.01.2025).pdf"
chunks = doc_processor.process_pdf(pdf_path)
print(f"Creati {len(chunks)} chunks dal PDF")

# Inizializza la knowledge base con i chunks del libro
if generator.initialize_book_knowledge(chunks):
    print("Knowledge base inizializzata con successo")
    
    # Test di generazione post
    test_query = "Parlami del rapporto Vesica"
    result = generator.generate_post(test_query, use_llm_knowledge=True)
    print("\nPost generato:")
    print(result)
else:
    print("Errore nell'inizializzazione della knowledge base")