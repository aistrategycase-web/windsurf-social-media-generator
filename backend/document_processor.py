import logging  # Assicurati che questa riga sia presente
import PyPDF2
import re

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text):
        """Pulisce il testo da caratteri non supportati"""
        text = re.sub(r'[^\x00-\x7F\xC0-\xFF]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_pdf(self, pdf_path):
        """Processa il PDF e lo divide in chunks"""
        try:
            chunks = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                current_chunk = ""
                current_metadata = {
                    "source": pdf_path,
                    "start_page": 1,
                    "end_page": 1
                }
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text = self.clean_text(text)
                    
                    current_chunk += text + " "
                    
                    if len(current_chunk) >= self.chunk_size:
                        chunks.append({
                            'content': current_chunk[:self.chunk_size],
                            'metadata': {
                                **current_metadata,
                                "end_page": page_num + 1
                            }
                        })
                        current_chunk = current_chunk[self.chunk_size - self.chunk_overlap:]
                        current_metadata = {
                            "source": pdf_path,
                            "start_page": page_num + 1,
                            "end_page": page_num + 1
                        }
            
            if current_chunk:
                chunks.append({
                    'content': current_chunk,
                    'metadata': current_metadata
                })
            
            logging.info(f"PDF processato con successo: {len(chunks)} chunks creati")
            return chunks
            
        except Exception as e:
            logging.error(f"Errore nel processing del PDF: {e}")
            return []