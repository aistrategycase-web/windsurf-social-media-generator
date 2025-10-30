document.addEventListener('DOMContentLoaded', function() {
    const generateButton = document.getElementById('generateButton');
    const queryInput = document.getElementById('queryInput');
    const generatedPostSection = document.getElementById('generatedPostSection');
    const generatedPostDiv = document.getElementById('generatedPost');
    const savePostButton = document.getElementById('savePostButton');
    const editPostButton = document.getElementById('editPostButton');
    const savedPostsList = document.getElementById('savedPostsList');
    const editPostModal = new bootstrap.Modal(document.getElementById('editPostModal'));
    const editPostTextarea = document.getElementById('editPostTextarea');
    const saveEditedPostButton = document.getElementById('saveEditedPostButton');
    const llmKnowledgeSwitch = document.getElementById('llmKnowledgeSwitch');

    let currentPost = '';
    let editingPostIndex = -1;

    // Carica i post salvati all'avvio
    function loadSavedPosts() {
        fetch('/get_posts')
            .then(response => response.json())
            .then(posts => {
                savedPostsList.innerHTML = '';
                posts.forEach((post, index) => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item d-flex justify-content-between align-items-center';
                    li.innerHTML = `
                        ${post}
                        <div>
                            <button class="btn btn-sm btn-warning me-2 edit-post" data-index="${index}">Modifica</button>
                            <button class="btn btn-sm btn-danger delete-post" data-index="${index}">Elimina</button>
                        </div>
                    `;
                    savedPostsList.appendChild(li);
                });

                // Aggiungi event listener per modificare e eliminare
                document.querySelectorAll('.edit-post').forEach(btn => {
                    btn.addEventListener('click', function() {
                        editingPostIndex = this.getAttribute('data-index');
                        editPostTextarea.value = this.closest('li').textContent.trim();
                        editPostModal.show();
                    });
                });

                document.querySelectorAll('.delete-post').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const index = this.getAttribute('data-index');
                        fetch('/delete_post', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({index: parseInt(index)})
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.message) { 
                                loadSavedPosts();
                            } else {
                                alert('Errore durante l\'eliminazione: ' + (data.error || 'Errore sconosciuto')); 
                            }
                        })
                        .catch(error => {
                            console.error('Errore:', error);
                            alert('Errore durante l\'eliminazione del post');
                        });
                    });
                });
            })
            .catch(error => {
                console.error('Errore nel caricamento dei post salvati:', error);
                alert('Errore nel caricamento dei post salvati.');
            });
    }

    generateButton.addEventListener('click', function() {
        const query = queryInput.value;
        const useLlmKnowledge = llmKnowledgeSwitch.checked;

        if (!query.trim()) {
            alert('Per favore inserisci una query');
            return;
        }

        generateButton.disabled = true;
        generateButton.textContent = 'Generazione in corso...';

        fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                use_llm_knowledge: useLlmKnowledge
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Errore HTTP! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            currentPost = data.post;
            generatedPostDiv.textContent = currentPost;
            generatedPostSection.style.display = 'block';

            // Gestione e visualizzazione dei chunk recuperati
            const retrievedChunksSection = document.getElementById('retrievedChunksSection');
            const retrievedChunksPlaceholder = document.getElementById('retrievedChunksPlaceholder');
            const retrievedChunksList = document.getElementById('retrievedChunksList');

            if (data.chunks && data.chunks.length > 0 && !useLlmKnowledge) {
                retrievedChunksList.innerHTML = '';
                
                // Aggiungi prima le fonti utilizzate
                if (data.sources && data.sources.length > 0) {
                    const sourcesHeader = document.createElement('li');
                    sourcesHeader.className = 'list-group-item bg-light';
                    sourcesHeader.innerHTML = `<strong>📚 Fonti utilizzate:</strong><br>${data.sources.join('<br>')}`;
                    retrievedChunksList.appendChild(sourcesHeader);
                }
                
                // Aggiungi i chunk
                data.chunks.forEach(chunk => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    
                    let sourceInfo = '';
                    if (chunk.source) {
                        sourceInfo = `<small class="text-muted">📄 Fonte: ${chunk.source}`;
                        if (chunk.page) {
                            sourceInfo += ` (Pagina ${chunk.page})`;
                        }
                        sourceInfo += '</small>';
                    }
                    
                    li.innerHTML = `
                        <div class="mb-2">${sourceInfo}</div>
                        <div class="chunk-text">${chunk.text}</div>
                    `;
                    retrievedChunksList.appendChild(li);
                });
                
                retrievedChunksPlaceholder.style.display = 'none';
                retrievedChunksList.style.display = 'block';
                retrievedChunksSection.style.display = 'block';
            } else {
                retrievedChunksPlaceholder.textContent = useLlmKnowledge ? 
                    'Nessun chunk mostrato in modalità LLM Knowledge.' : 
                    'Nessun chunk recuperato.';
                retrievedChunksPlaceholder.style.display = 'block';
                retrievedChunksList.style.display = 'none';
                retrievedChunksSection.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Errore nella generazione del post:', error);
            alert('Errore nella generazione del post: ' + error.message);
        })
        .finally(() => {
            generateButton.disabled = false;
            generateButton.textContent = '🚀 Genera Post! 🚀';
        });
    });

    savePostButton.addEventListener('click', function() {
        fetch('/add_post', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ post: currentPost })
        })
        .then(response => response.json())
        .then(data => {
             if (data.message) { 
                loadSavedPosts();
            } else {
                alert('Errore durante il salvataggio: ' + (data.error || 'Errore sconosciuto')); 
            }
        })
        .catch(error => {
            console.error('Errore nel salvataggio del post:', error);
            alert('Errore nel salvataggio del post.');
        });
    });

    editPostButton.addEventListener('click', function() {
        editPostTextarea.value = currentPost;
        editPostModal.show();
    });

    saveEditedPostButton.addEventListener('click', function() {
        const editedPost = editPostTextarea.value;

        if (editingPostIndex !== -1) {
            // Modifica un post esistente
            fetch('/edit_post', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    index: parseInt(editingPostIndex),
                    post: editedPost
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) { 
                    editPostModal.hide();
                    loadSavedPosts();
                } else {
                    alert('Errore durante la modifica: ' + (data.error || 'Errore sconosciuto')); 
                }
            })
            .catch(error => {
                console.error('Errore durante la modifica del post:', error);
                alert('Errore durante la modifica del post');
            });
        } else {
            // Modifica il post generato
            currentPost = editedPost;
            generatedPostDiv.textContent = currentPost;
            editPostModal.hide();
        }
    });


    // Carica i post salvati all'avvio
    loadSavedPosts();
});