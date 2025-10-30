import logging
import asyncio
from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch
import google.generativeai as genai
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
client = genai.Client()

async def get_instagram_updates_prompt():
    """
    Funzione asincrona per cercare aggiornamenti su Instagram tramite Ricerca Google e creare un prompt aggiornato.
    Restituisce una stringa contenente il prompt aggiornato con le informazioni dal web.
    """
    logger.info("Inizio ricerca aggiornamenti Instagram dal web...")

    google_search_tool = Tool(google_search=GoogleSearch()) # Inizializza lo strumento di ricerca Google

    query_ricerca = "ultimi aggiornamenti algoritmo Instagram best practice Instagram 2025" # Query di ricerca mirata

    try:
        response = await asyncio.to_thread(
            client.models.generate_content, # Usa client.models.generate_content invece di genai.GenerativeModel
            model="gemini-pro", # Specifica il model_id qui
            contents=query_ricerca,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"]
            )
        )

        search_result_text = ""
        for part in response.candidates[0].content.parts: # Itera sulle parti del contenuto della risposta
            search_result_text += part.text + "\n\n" # Aggiungi il testo di ogni parte e una riga vuota

        logger.info("Ricerca aggiornamenti Instagram dal web completata.")
        logger.info(f"Risultati ricerca web (prime 500 char): {search_result_text[:500]}...") # Log dei primi 500 caratteri dei risultati

        # Costruisci il prompt aggiornato IN BASE AI RISULTATI DELLA RICERCA WEB (DA IMPLEMENTARE!)
        # *** ATTENZIONE: Al momento, questo è solo un ESEMPIO BASE! ***
        # *** Dovrai *personalizzare* * *molto* * *questo prompt* * *per estrarre * * *davvero* * *informazioni utili dai risultati di ricerca! ***
        updated_prompt = f"""
        **ULTIMI AGGIORNAMENTI E BEST PRACTICE INSTAGRAM (dal web):**
        {search_result_text}

        **PROMPT ORIGINALE (DA MIGLIORARE CON LE INFO DAL WEB):**
        ... (QUI DOVRESTI INSERIRE IL TUO PROMPT RAG-ONLY FLESSIBILE ORIGINALE) ... 
        """ 

        return updated_prompt # Restituisci il prompt aggiornato

    except Exception as e:
        logger.error(f"Errore durante la ricerca di aggiornamenti Instagram dal web: {e}")
        return None # In caso di errore, restituisci None

# Funzione di test (per testare la funzione di ricerca web *separatamente*)
async def main():
    updated_prompt = await get_instagram_updates_prompt()
    if updated_prompt:
        print("\nPROMPT AGGIORNATO:\n")
        print(updated_prompt)
    else:
        print("\nErrore nel recupero del prompt aggiornato.")

if __name__ == "__main__":
    asyncio.run(main())