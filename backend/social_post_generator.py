import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from backend.rag_engine import SocialPostGenerator
import sys
import time

# Carica variabili d'ambiente
load_dotenv()

# Configurazione
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("social_posts")

# Carica dati
def load_data():
    with open("data/style_guide.txt", "r") as f:
        style_guide = f.read()
    
    with open("data/successful_posts.json", "r") as f:
        successful_posts = json.load(f)
    
    return [style_guide] + successful_posts

# Inizializza il database vettoriale
documents = load_data()
embeddings = embedding_model.encode(documents).tolist()
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"id_{i}" for i in range(len(documents))]
)

# Funzione di generazione
def generate_post(query: str) -> dict:
    # Recupera contesto
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context_style = results["documents"][0][0]
    context_examples = results["documents"][0][1:]
    
    # Prompt per Gemini
    prompt = f"""
    **LINEA GUIDA**:
    {context_style}

    **ESEMPI DI SUCCESSO**:
    {context_examples}

    **RICHIESTA**:
    {query}

    Genera un post per social media in italiano, seguendo lo stile sopra.
    """
    
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return {"post": response.text}

class SocialPostGenerator:
    def __init__(self):
        self.posts_path = "data/successful_posts.json"
        self.successful_posts = load_data()[1:]

    def generate_post(self, query: str) -> dict:
        return generate_post(query)

    async def add_successful_post(self, post: str) -> bool:
        self.successful_posts.append(post)
        await self.save_posts()
        return True

    async def save_posts(self):
        os.makedirs(os.path.dirname(self.posts_path), exist_ok=True)
        with open(self.posts_path, "w") as f:
            json.dump(self.successful_posts, f, indent=2)

class SocialPostCLI:
    def __init__(self):
        print("Inizializzazione del generatore di post...")
        self.generator = SocialPostGenerator()
        print("\nGeneratore inizializzato con successo!")

    def show_menu(self):
        """Mostra il menu principale"""
        print("\n=== Social Post Generator ===")
        print("1. Genera nuovo post")
        print("2. Aggiungi post di esempio")
        print("3. Visualizza post esistenti")
        print("4. Gestisci post esistenti")
        print("5. Esci")
        return input("\nScegli un'opzione (1-5): ")

    def generate_post(self):
        """Gestisce la generazione di un nuovo post"""
        print("\n=== Genera Nuovo Post ===")
        query = input("Di cosa vuoi parlare nel post? ")
        print("\nGenerazione in corso...")
        post = self.generator.generate_post(query)
        print("\nPost generato:")
        print("-" * 50)
        print(post["post"])
        print("-" * 50)
        
        if input("\nVuoi salvare questo post come esempio? (s/n): ").lower() == 's':
            self.generator.add_successful_post(post["post"])
            print("Post salvato con successo!")

    def add_example_post(self):
        """Gestisce l'aggiunta di un nuovo post di esempio"""
        print("\n=== Aggiungi Post di Esempio ===")
        print("Inserisci il testo del post (premi Ctrl+D su una nuova riga quando hai finito):")
        print("Per terminare l'inserimento, premi Invio, poi Ctrl+D (o Ctrl+Z su Windows)")
        
        # Raccoglie l'input multilinea
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            post = '\n'.join(lines)
        except KeyboardInterrupt:
            print("\nOperazione annullata.")
            return
        
        if not post.strip():
            print("Post vuoto. Operazione annullata.")
            return
            
        if self.generator.add_successful_post(post):
            print("\nPost aggiunto con successo!")
        else:
            print("\nErrore nell'aggiunta del post.")

    def view_posts(self):
        """Visualizza i post esistenti"""
        print("\n=== Post Esistenti ===")
        posts = self.generator.successful_posts
        if not posts:
            print("Nessun post trovato.")
            return
        
        for i, post in enumerate(posts, 1):
            print(f"\n{i}. {post}")

    def manage_posts(self):
        """Gestisce le operazioni sui post esistenti"""
        while True:
            print("\n=== Gestione Post ===")
            print("1. Modifica un post")
            print("2. Elimina un post")
            print("3. Torna al menu principale")
            
            choice = input("\nScegli un'opzione (1-3): ")
            
            if choice == "1":
                self.edit_post()
            elif choice == "2":
                self.delete_post()
            elif choice == "3":
                break
            else:
                print("\nOpzione non valida. Riprova.")

    def edit_post(self):
        """Modifica un post esistente"""
        posts = self.generator.successful_posts
        if not posts:
            print("Nessun post da modificare.")
            return

        self.view_posts()
        try:
            index = int(input("\nInserisci il numero del post da modificare: ")) - 1
            if 0 <= index < len(posts):
                print("\nPost attuale:")
                print(posts[index])
                
                print("\nInserisci il nuovo testo (premi Ctrl+D su una nuova riga quando hai finito):")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    new_post = '\n'.join(lines)
                except KeyboardInterrupt:
                    print("\nModifica annullata.")
                    return

                # Aggiorna il post
                posts[index] = new_post
                
                # Salva i cambiamenti nel file JSON
                with open("data/successful_posts.json", "w") as f:
                    json.dump(posts, f, indent=2)
                
                print("\nPost modificato con successo!")
            else:
                print("Numero di post non valido.")
        except ValueError:
            print("Input non valido. Inserisci un numero.")

    def delete_post(self):
        """Elimina un post esistente"""
        posts = self.generator.successful_posts
        if not posts:
            print("Nessun post da eliminare.")
            return

        self.view_posts()
        try:
            index = int(input("\nInserisci il numero del post da eliminare: ")) - 1
            if 0 <= index < len(posts):
                deleted_post = posts.pop(index)
                
                # Salva i cambiamenti nel file JSON
                with open("data/successful_posts.json", "w") as f:
                    json.dump(posts, f, indent=2)
                
                print(f"\nPost eliminato:\n{deleted_post}")
            else:
                print("Numero di post non valido.")
        except ValueError:
            print("Input non valido. Inserisci un numero.")

    def run(self):
        """Esegue il loop principale dell'applicazione"""
        while True:
            choice = self.show_menu()
            
            if choice == "1":
                self.generate_post()
            elif choice == "2":
                self.add_example_post()
            elif choice == "3":
                self.view_posts()
            elif choice == "4":
                self.manage_posts()
            elif choice == "5":
                print("\nGrazie per aver usato Social Post Generator!")
                sys.exit(0)
            else:
                print("\nOpzione non valida. Riprova.")
            
            time.sleep(1)

if __name__ == "__main__":
    try:
        cli = SocialPostCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nUscita dal programma...")
        sys.exit(0)