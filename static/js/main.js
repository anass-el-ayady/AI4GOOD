// Variables globales
let sessionId = null;
let dataType = null;
let selectedModel = null;
let selectedDataset = null;
let availableModels = [];

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    loadDatasets();
});

// Charger les jeux de donn\u00e9es disponibles
async function loadDatasets() {
    showLoading();

    try {
        const response = await fetch('/get_datasets');
        const data = await response.json();
        
        if (response.ok) {
            displayDatasets(data.datasets);
            hideLoading();
        } else {
            showError('Erreur lors du chargement des jeux de donn\u00e9es');
            hideLoading();
        }
    } catch (error) {
        showError('Erreur lors du chargement des jeux de donn\u00e9es: ' + error.message);
        hideLoading();
    }
}

// Afficher les cartes de jeux de donn\u00e9es
function displayDatasets(datasets) {
    const container = document.getElementById('dataset-selection');
    
    container.innerHTML = datasets.map(dataset => `
        <div class="col-md-6 col-lg-3">
            <div class="model-card" onclick="selectDataset('${dataset.id}', this)">
                <div class="model-icon">${dataset.icon}</div>
                <h5>${dataset.name}</h5>
                <p class="small">${dataset.description}</p>
                <div class="mt-2">
                    <span class="badge bg-primary">${dataset.size}</span>
                    <span class="badge bg-success">${dataset.classes} classes</span>
                </div>
            </div>
        </div>
    `).join('');
}

// S\u00e9lectionner un jeu de donn\u00e9es
async function selectDataset(datasetId, element) {
    selectedDataset = datasetId;
    
    // Retirer la s\u00e9lection de toutes les cartes
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Ajouter la s\u00e9lection \u00e0 la carte cliqu\u00e9e
    element.classList.add('selected');
    
    // Charger le jeu de donn\u00e9es
    await loadDataset(datasetId);
}

// Charger les \u00e9chantillons du jeu de donn\u00e9es
async function loadDataset(datasetId) {
    showLoading();

    try {
        const response = await fetch('/load_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset_id: datasetId })
        });

        const data = await response.json();
        
        if (response.ok) {
            sessionId = data.session_id;
            dataType = data.data_type;
            
            displayDatasetPreview(data);
            hideLoading();
        } else {
            showError(data.error || 'Erreur lors du chargement du jeu de donn\u00e9es');
            hideLoading();
        }
    } catch (error) {
        showError('Erreur r\u00e9seau: ' + error.message);
        hideLoading();
    }
}

// Afficher l'aper\u00e7u du jeu de donn\u00e9es
function displayDatasetPreview(data) {
    const previewDiv = document.getElementById('data-preview');
    const infoDiv = document.getElementById('dataset-info');
    const contentDiv = document.getElementById('preview-content');
    const learningSection = document.getElementById('learning-section');
    
    // Afficher la section d'apprentissage
    learningSection.classList.remove('d-none');
    
    // Obtenir les d\u00e9tails d\u00e9taill\u00e9s en fonction du jeu de donn\u00e9es
    const datasetDetails = getDatasetDetails(selectedDataset);
    
    // Afficher les informations am\u00e9lior\u00e9es du jeu de donn\u00e9es
    infoDiv.innerHTML = `
        <div class="row">
            <div class="col-lg-8">
                <h5><i class="fas fa-database"></i> ${data.dataset_name}</h5>
                <p class="lead mb-3">${data.info}</p>
                <div class="alert alert-info mb-3">
                    <h6 class="mb-2"><i class="fas fa-graduation-cap"></i> Ce que vous allez apprendre:</h6>
                    <p class="mb-0">${datasetDetails.learning}</p>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6 class="card-title"><i class="fas fa-chart-bar"></i> Statistiques Rapides</h6>
                        ${datasetDetails.stats}
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="card border-primary">
                    <div class="card-body">
                        <h6><i class="fas fa-bullseye"></i> Utilisations R\u00e9elles</h6>
                        <ul class="mb-0 small">
                            ${datasetDetails.uses.map(use => `<li>${use}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card border-success">
                    <div class="card-body">
                        <h6><i class="fas fa-lightbulb"></i> Pourquoi ce Jeu de Donn\u00e9es?</h6>
                        <p class="mb-0 small">${datasetDetails.why}</p>
                    </div>
                </div>
            </div>
        </div>
    `;

    let html = '';

    if (data.data_type === 'image') {
        html = `
            <h6 class="mb-3 mt-4"><i class="fas fa-images"></i> Exemples d'Images du Jeu de Donn\u00e9es:</h6>
            <div class="alert alert-light">
                <small><i class="fas fa-info-circle"></i> Ce sont de vrais exemples du jeu de donn\u00e9es. Remarquez la vari\u00e9t\u00e9 et les diff\u00e9rents styles d'\u00e9criture!</small>
            </div>
            <div class="row g-3">
        `;
        
        data.samples.forEach((sample, idx) => {
            html += `
                <div class="col-6 col-md-3 col-lg-2 text-center">
                    <div class="border rounded p-2 bg-white">
                        <img src="${sample.image}" class="img-fluid" alt="\u00c9chantillon ${idx}">
                        <small class="d-block mt-2"><strong>\u00c9tiquette:</strong> ${sample.label}</small>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    } else {
        // Donn\u00e9es tabulaires
        html = `
            <h6 class="mb-3 mt-4"><i class="fas fa-table"></i> Exemples de Donn\u00e9es du Jeu de Donn\u00e9es:</h6>
            <div class="alert alert-light">
                <small><i class="fas fa-info-circle"></i> Chaque ligne est un exemple, et chaque colonne est une caract\u00e9ristique dont l'IA apprendra!</small>
            </div>
            <div class="table-responsive">
                ${data.sample_data}
            </div>
        `;
        
        // Ajouter les descriptions des caract\u00e9ristiques si disponibles
        if (data.features && data.features.length > 0) {
            html += `
                <div class="mt-3">
                    <h6><i class="fas fa-list"></i> Caract\u00e9ristiques du Jeu de Donn\u00e9es:</h6>
                    <div class="row">
            `;
            data.features.slice(0, 4).forEach(feature => {
                html += `
                    <div class="col-md-6 col-lg-3 mb-2">
                        <span class="badge bg-primary">${feature}</span>
                    </div>
                `;
            });
            html += '</div></div>';
        }
    }

    contentDiv.innerHTML = html;
    previewDiv.classList.remove('d-none');
    
    // Faire d\u00e9filer jusqu'\u00e0 la section d'apprentissage
    learningSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Obtenir des informations d\u00e9taill\u00e9es sur chaque jeu de donn\u00e9es
function getDatasetDetails(datasetId) {
    const details = {
        'mnist': {
            learning: 'Travaillez avec des donn\u00e9es d\'images et comprenez comment les ordinateurs "voient" les chiffres manuscrits. Parfait pour apprendre la vision par ordinateur et les r\u00e9seaux de neurones!',
            stats: `
                <p class="mb-1"><strong>\u00c9chantillons Totaux:</strong> 70 000 images</p>
                <p class="mb-1"><strong>Taille de l'Image:</strong> 28\u00d728 pixels (784 valeurs)</p>
                <p class="mb-1"><strong>Classes:</strong> 10 chiffres (0-9)</p>
                <p class="mb-1"><strong>Couleur:</strong> Niveaux de gris</p>
                <p class="mb-0"><strong>T\u00e2che:</strong> Classification</p>
            `,
            uses: [
                '<img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" width="16" height="16" style="vertical-align: middle;"> Services postaux pour tri automatique du courrier',
                '<img src="https://cdn-icons-png.flaticon.com/512/2830/2830284.png" width="16" height="16" style="vertical-align: middle;"> Banques pour le traitement des ch\u00e8ques',
                '<img src="https://cdn-icons-png.flaticon.com/512/3143/3143609.png" width="16" height="16" style="vertical-align: middle;"> Lecture num\u00e9rique de formulaires et automatisation',
                '<img src="https://cdn-icons-png.flaticon.com/512/1476/1476960.png" width="16" height="16" style="vertical-align: middle;"> Enseigner aux ordinateurs la reconnaissance d\'images de base'
            ],
            why: 'MNIST est le "Hello World" de l\'apprentissage automatique! C\'est parfait pour les d\u00e9butants car c\'est assez simple pour s\'entra\u00eener rapidement mais assez complexe pour apprendre de vrais concepts d\'IA. Chaque chercheur en IA a travaill\u00e9 avec MNIST!'
        },
        'iris': {
            learning: 'Apprenez la classification avec un jeu de donn\u00e9es classique! Comprenez comment l\'IA peut identifier des motifs dans la nature \u00e0 l\'aide de mesures.',
            stats: `
                <p class="mb-1"><strong>\u00c9chantillons Totaux:</strong> 150 fleurs</p>
                <p class="mb-1"><strong>Caract\u00e9ristiques:</strong> 4 mesures</p>
                <p class="mb-1"><strong>Classes:</strong> 3 esp\u00e8ces</p>
                <p class="mb-1"><strong>Esp\u00e8ces:</strong> Setosa, Versicolor, Virginica</p>
                <p class="mb-0"><strong>T\u00e2che:</strong> Classification</p>
            `,
            uses: [
                '<img src="https://cdn-icons-png.flaticon.com/512/628/628324.png" width="16" height="16" style="vertical-align: middle;"> Identification des esp\u00e8ces v\u00e9g\u00e9tales en botanique',
                '<img src="https://cdn-icons-png.flaticon.com/512/3488/3488751.png" width="16" height="16" style="vertical-align: middle;"> Reconnaissance de motifs en recherche biologique',
                '<img src="https://cdn-icons-png.flaticon.com/512/2920/2920277.png" width="16" height="16" style="vertical-align: middle;"> Enseignement des concepts de classification de base',
                '<img src="https://cdn-icons-png.flaticon.com/512/3655/3655589.png" width="16" height="16" style="vertical-align: middle;"> Comprendre comment les caract\u00e9ristiques sont li\u00e9es aux cat\u00e9gories'
            ],
            why: 'Iris est le jeu de donn\u00e9es le plus c\u00e9l\u00e8bre en statistiques! Cr\u00e9\u00e9 en 1936, il est parfait pour l\'apprentissage car il ne comporte que 4 caract\u00e9ristiques simples (longueur de p\u00e9tale, largeur, etc.) mais montre clairement de puissants concepts d\'IA.'
        },
        'wine': {
            learning: 'D\u00e9couvrez comment l\'IA analyse les propri\u00e9t\u00e9s chimiques pour classifier les produits. Id\u00e9al pour comprendre la classification multi-caract\u00e9ristiques!',
            stats: `
                <p class="mb-1"><strong>\u00c9chantillons Totaux:</strong> 178 vins</p>
                <p class="mb-1"><strong>Caract\u00e9ristiques:</strong> 13 propri\u00e9t\u00e9s chimiques</p>
                <p class="mb-1"><strong>Classes:</strong> 3 types de vin</p>
                <p class="mb-1"><strong>Propri\u00e9t\u00e9s:</strong> Alcool, acidit\u00e9, pH, etc.</p>
                <p class="mb-0"><strong>T\u00e2che:</strong> Classification</p>
            `,
            uses: [
                '<img src="https://cdn-icons-png.flaticon.com/512/924/924514.png" width="16" height="16" style="vertical-align: middle;"> \u00c9valuation et notation de la qualit\u00e9 du vin',
                '<img src="https://cdn-icons-png.flaticon.com/512/3063/3063822.png" width="16" height="16" style="vertical-align: middle;"> Contr\u00f4le qualit\u00e9 dans l\'industrie alimentaire',
                '<img src="https://cdn-icons-png.flaticon.com/512/3655/3655650.png" width="16" height="16" style="vertical-align: middle;"> Analyse chimique et classification',
                '<img src="https://cdn-icons-png.flaticon.com/512/2920/2920349.png" width="16" height="16" style="vertical-align: middle;"> Comprendre les relations complexes entre caract\u00e9ristiques'
            ],
            why: 'Le jeu de donn\u00e9es Wine montre comment l\'IA peut trouver des motifs que les humains pourraient manquer! Avec 13 mesures chimiques diff\u00e9rentes, il d\u00e9montre comment l\'IA g\u00e8re plusieurs caract\u00e9ristiques pour faire des pr\u00e9dictions pr\u00e9cises sur les types de vin.'
        },
        'digits': {
            learning: 'Une version plus petite et plus rapide de MNIST! Parfait pour des exp\u00e9riences rapides avec la reconnaissance d\'images et voir les r\u00e9sultats rapidement.',
            stats: `
                <p class="mb-1"><strong>\u00c9chantillons Totaux:</strong> 1 797 images</p>
                <p class="mb-1"><strong>Taille de l'Image:</strong> 8\u00d78 pixels (64 valeurs)</p>
                <p class="mb-1"><strong>Classes:</strong> 10 chiffres (0-9)</p>
                <p class="mb-1"><strong>Couleur:</strong> Niveaux de gris</p>
                <p class="mb-0"><strong>T\u00e2che:</strong> Classification</p>
            `,
            uses: [
                '<img src="https://cdn-icons-png.flaticon.com/512/4230/4230573.png" width="16" height="16" style="vertical-align: middle;"> Prototypage rapide et test de mod\u00e8les d\'IA',
                '<img src="https://cdn-icons-png.flaticon.com/512/2232/2232688.png" width="16" height="16" style="vertical-align: middle;"> D\u00e9monstrations \u00e9ducatives d\'IA d\'image',
                '<img src="https://cdn-icons-png.flaticon.com/512/3649/3649425.png" width="16" height="16" style="vertical-align: middle;"> Applications de reconnaissance de chiffres en temps r\u00e9el',
                '<img src="https://cdn-icons-png.flaticon.com/512/3097/3097393.png" width="16" height="16" style="vertical-align: middle;"> Apprentissage rapide des bases de la vision par ordinateur'
            ],
            why: 'Digits est la version "d\u00e9marrage rapide" de la reconnaissance d\'images! Sa taille plus petite (8\u00d78 au lieu de 28\u00d728) signifie que les mod\u00e8les s\'entra\u00eenent en quelques secondes, ce qui le rend parfait pour l\'exp\u00e9rimentation et l\'apprentissage sans longues attentes.'
        }
    };
    
    return details[datasetId] || details['mnist'];
}

// Afficher la r\u00e9ponse aux questions d'apprentissage
function showAnswer(questionId) {
    const answerDisplay = document.getElementById('answer-display');
    const answerContent = document.getElementById('answer-content');
    
    const answers = {
        q1: {
            title: '<img src="https://cdn-icons-png.flaticon.com/512/2875/2875433.png" width="24" height="24" style="vertical-align: middle;"> Pourquoi Avons-nous Besoin de Jeux de Donn\u00e9es?',
            content: `
                <h5 class="text-primary mb-3">Pensez \u00e0 l'IA comme un \u00e9tudiant apprenant \u00e0 faire du v\u00e9lo!</h5>
                <p class="lead">Tout comme vous devez vous entra\u00eener \u00e0 faire du v\u00e9lo plusieurs fois pour y arriver, l'IA doit voir beaucoup d'exemples (jeux de donn\u00e9es) pour apprendre des motifs et faire des pr\u00e9dictions!</p>
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="alert alert-success">
                            <h6><i class="fas fa-lightbulb"></i> Sans Donn\u00e9es:</h6>
                            <p class="mb-0"><img src="https://cdn-icons-png.flaticon.com/512/753/753345.png" width="20" height="20" style="vertical-align: middle;"> L'IA est comme un \u00e9tudiant sans manuel - elle ne peut rien apprendre!</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="alert alert-info">
                            <h6><i class="fas fa-star"></i> Avec des Donn\u00e9es:</h6>
                            <p class="mb-0"><img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="20" height="20" style="vertical-align: middle;"> L'IA apprend des exemples, tout comme vous apprenez en pratiquant!</p>
                        </div>
                    </div>
                </div>
                <p class="mt-3"><strong>Exemple R\u00e9el:</strong> Pour enseigner \u00e0 l'IA \u00e0 reconna\u00eetre les chats, nous lui montrons des milliers d'images de chats. Plus elle voit d'exemples, meilleure elle devient! <img src="https://cdn-icons-png.flaticon.com/512/2138/2138440.png" width="20" height="20" style="vertical-align: middle;"></p>
            `
        },
        q2: {
            title: '<img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="24" height="24" style="vertical-align: middle;"> Comment Savoir si les Donn\u00e9es sont Bonnes?',
            content: `
                <h5 class="text-success mb-3">Les bonnes donn\u00e9es sont comme une alimentation saine pour l'IA!</h5>
                <p class="lead">Tout comme manger une vari\u00e9t\u00e9 d'aliments sains vous rend fort, l'IA a besoin de donn\u00e9es diverses et propres pour \u00eatre intelligente!</p>
                <div class="card bg-light mt-4 mb-4">
                    <div class="card-body">
                        <h6 class="mb-3"><img src="https://cdn-icons-png.flaticon.com/512/1828/1828640.png" width="20" height="20" style="vertical-align: middle;"> Liste de V\u00e9rification des Bonnes Donn\u00e9es:</h6>
                        <ul class="list-unstyled">
                            <li class="mb-2"><img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="18" height="18" style="vertical-align: middle;"> <strong>Assez d'Exemples:</strong> Plus c'est mieux! (Comme \u00e9tudier plus pour un test)</li>
                            <li class="mb-2"><img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="18" height="18" style="vertical-align: middle;"> <strong>Divers:</strong> Diff\u00e9rents types d'exemples (Pas seulement des chats persans, tous les types de chats!)</li>
                            <li class="mb-2"><img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="18" height="18" style="vertical-align: middle;"> <strong>Pr\u00e9cis:</strong> Correctement \u00e9tiquet\u00e9 (Imaginez si les images de chats \u00e9taient \u00e9tiquet\u00e9es comme des chiens - d\u00e9routant!)</li>
                            <li class="mb-2"><img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="18" height="18" style="vertical-align: middle;"> <strong>\u00c9quilibr\u00e9:</strong> Exemples \u00e9gaux de chaque type (Pas 1000 chats et seulement 10 chiens)</li>
                            <li class="mb-2"><img src="https://cdn-icons-png.flaticon.com/512/5290/5290058.png" width="18" height="18" style="vertical-align: middle;"> <strong>Propre:</strong> Pas d'erreurs bizarres ou de parties manquantes</li>
                        </ul>
                    </div>
                </div>
                <p><strong>Fait Amusant:</strong> Mauvaises donn\u00e9es = Mauvaise IA, m\u00eame avec les meilleurs algorithmes! <img src="https://cdn-icons-png.flaticon.com/512/2875/2875433.png" width="20" height="20" style="vertical-align: middle;"> "Entr\u00e9e d\u00e9chets, sortie d\u00e9chets!"</p>
            `
        },
        q3: {
            title: '<img src="https://cdn-icons-png.flaticon.com/512/2920/2920235.png" width="24" height="24" style="vertical-align: middle;"> Pourquoi Diviser les Donn\u00e9es?',
            content: `
                <h5 class="text-warning mb-3">C'est comme \u00e9tudier pour un examen!</h5>
                <p class="lead">Imaginez \u00e9tudier TOUTES les questions de pratique... puis l'examen a LES M\u00caMES questions! Vous auriez 100%, mais avez-vous vraiment appris? <img src="https://cdn-icons-png.flaticon.com/512/2621/2621040.png" width="20" height="20" style="vertical-align: middle;"></p>
                <div class="row mt-4">
                    <div class="col-md-4">
                        <div class="card text-center h-100" style="border: 3px solid #28a745;">
                            <div class="card-body">
                                <h3><img src="https://cdn-icons-png.flaticon.com/512/2232/2232688.png" width="32" height="32" style="vertical-align: middle;"> 80%</h3>
                                <h6 class="text-success">Donn\u00e9es d'Entra\u00eenement</h6>
                                <p class="small">C'est l\u00e0 que l'IA \u00e9tudie et apprend les motifs!</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center h-100" style="border: 3px solid #ffc107;">
                            <div class="card-body">
                                <h3><img src="https://cdn-icons-png.flaticon.com/512/2541/2541988.png" width="32" height="32" style="vertical-align: middle;"> 10%</h3>
                                <h6 class="text-warning">Donn\u00e9es de Validation</h6>
                                <p class="small">Test de pratique - aide \u00e0 ajuster l'IA pendant l'apprentissage!</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center h-100" style="border: 3px solid #dc3545;">
                            <div class="card-body">
                                <h3><img src="https://cdn-icons-png.flaticon.com/512/3074/3074767.png" width="32" height="32" style="vertical-align: middle;"> 10%</h3>
                                <h6 class="text-danger">Donn\u00e9es de Test</h6>
                                <p class="small">Examen final - NOUVELLES donn\u00e9es que l'IA n'a JAMAIS vues!</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="alert alert-primary mt-4">
                    <strong><img src="https://cdn-icons-png.flaticon.com/512/3176/3176369.png" width="20" height="20" style="vertical-align: middle;"> Point Cl\u00e9:</strong> Nous cachons les donn\u00e9es de test jusqu'\u00e0 la fin pour voir si l'IA a vraiment appris ou juste m\u00e9moris\u00e9! C'est tester la vraie intelligence! <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" width="20" height="20" style="vertical-align: middle;"></div>
            `
        },
        q4: {
            title: '<img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" width="24" height="24" style="vertical-align: middle;"> Comment l\'IA Apprend-elle R\u00e9ellement?',
            content: `
                <h5 class="text-info mb-3">L'IA apprend en jouant \u00e0 un jeu de devinettes... des millions de fois!</h5>
                <p class="lead">Pensez \u00e0 enseigner \u00e0 un b\u00e9b\u00e9 \u00e0 identifier les animaux. Vous montrez des images et dites "chat!" ou "chien!" - le b\u00e9b\u00e9 apprend de vos corrections! <img src="https://cdn-icons-png.flaticon.com/512/2784/2784403.png" width="20" height="20" style="vertical-align: middle;"></p>
                <div class="card mt-4 mb-4" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none;">
                    <div class="card-body text-white">
                        <h6 class="mb-3 text-white"><img src="https://cdn-icons-png.flaticon.com/512/3094/3094837.png" width="20" height="20" style="vertical-align: middle;"> La Boucle d'Apprentissage:</h6>
                        <ol class="mb-0 text-white">
                            <li class="mb-2"><strong>Faire une Supposition:</strong> L'IA regarde les donn\u00e9es et fait une pr\u00e9diction (souvent fausse au d\u00e9but!)</li>
                            <li class="mb-2"><strong>V\u00e9rifier la R\u00e9ponse:</strong> Comparer la supposition \u00e0 la bonne r\u00e9ponse</li>
                            <li class="mb-2"><strong>Apprendre des Erreurs:</strong> S'ajuster pour \u00eatre plus pr\u00e9cis la prochaine fois</li>
                            <li class="mb-2"><strong>R\u00e9p\u00e9ter:</strong> Faire cela des milliers ou des millions de fois!</li>
                        </ol>
                    </div>
                </div>
                <p><strong>Exemple Cool:</strong> Pour reconna\u00eetre un "5" manuscrit, l'IA pourrait se tromper 1000 fois au d\u00e9but. Mais chaque erreur lui apprend \u00e0 quoi ressemble un "5" jusqu'\u00e0 ce qu'elle y arrive presque \u00e0 chaque fois! <img src="https://cdn-icons-png.flaticon.com/512/2875/2875433.png" width="20" height="20" style="vertical-align: middle;"></p>
                <div class="alert alert-success mt-3">
                    <strong><img src="https://cdn-icons-png.flaticon.com/512/2235/2235219.png" width="20" height="20" style="vertical-align: middle;"> Formule Magique:</strong> Beaucoup de Donn\u00e9es + Nombreuses Tentatives + Apprendre des Erreurs = IA Intelligente! <img src="https://cdn-icons-png.flaticon.com/512/3588/3588592.png" width="20" height="20" style="vertical-align: middle;"></div>
            `
        }
    };
    
    const answer = answers[questionId];
    answerContent.innerHTML = `
        <h4>${answer.title}</h4>
        ${answer.content}
    `;
    
    answerDisplay.classList.remove('d-none');
    answerDisplay.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Fermer l'affichage de la r\u00e9ponse
function closeAnswer() {
    document.getElementById('answer-display').classList.add('d-none');
}

// Naviguer vers une \u00e9tape
function goToStep(stepNumber) {
    // Masquer toutes les \u00e9tapes
    document.querySelectorAll('.step-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Afficher l'\u00e9tape s\u00e9lectionn\u00e9e
    document.getElementById(`step-${stepNumber}`).classList.add('active');
    
    // Mettre \u00e0 jour les indicateurs d'\u00e9tape
    document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
        if (index + 1 <= stepNumber) {
            indicator.classList.add('active');
        } else {
            indicator.classList.remove('active');
        }
    });

    // Charger les mod\u00e8les si on va \u00e0 l'\u00e9tape 2
    if (stepNumber === 2 && sessionId) {
        loadModels();
    }

    // Faire d\u00e9filer vers le haut
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Charger les mod\u00e8les disponibles
async function loadModels() {
    showLoading();

    try {
        const response = await fetch('/get_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });

        const data = await response.json();
        
        if (response.ok) {
            availableModels = data.models;
            displayModels(data.models);
            hideLoading();
        } else {
            showError(data.error);
            hideLoading();
        }
    } catch (error) {
        showError('Erreur lors du chargement des mod\u00e8les: ' + error.message);
        hideLoading();
    }
}

// Afficher les cartes de mod\u00e8les
function displayModels(models) {
    const container = document.getElementById('model-selection');
    
    container.innerHTML = models.map(model => `
        <div class="col-md-6 col-lg-3">
            <div class="model-card" onclick="selectModel('${model.id}', this)">
                <div class="model-icon">${model.icon}</div>
                <h5>${model.name}</h5>
                <p>${model.description}</p>
                <span class="badge bg-info">${model.type}</span>
            </div>
        </div>
    `).join('');
}

// S\u00e9lectionner un mod\u00e8le
function selectModel(modelId, element) {
    selectedModel = modelId;
    
    // Retirer la s\u00e9lection de toutes les cartes
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Ajouter la s\u00e9lection \u00e0 la carte cliqu\u00e9e
    element.classList.add('selected');
    
    // Trouver les hyperparam\u00e8tres du mod\u00e8le s\u00e9lectionn\u00e9
    const model = availableModels.find(m => m.id === modelId);
    if (model && model.hyperparameters) {
        displayHyperparameters(model.hyperparameters);
    } else {
        document.getElementById('hyperparameter-config').classList.add('d-none');
    }
    
    // Activer le bouton d'entra\u00eenement
    document.getElementById('train-button').disabled = false;
}

// Afficher les entr\u00e9es d'hyperparam\u00e8tres
function displayHyperparameters(hyperparameters) {
    const container = document.getElementById('hyperparameter-inputs');
    const configDiv = document.getElementById('hyperparameter-config');
    
    let html = '';
    
    for (const [key, config] of Object.entries(hyperparameters)) {
        html += `
            <div class="col-md-6">
                <label class="form-label fw-bold">
                    ${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    <i class="fas fa-info-circle text-info ms-1" 
                       data-bs-toggle="tooltip" 
                       data-bs-placement="top" 
                       title="${config.description}"></i>
                </label>
        `;
        
        if (config.type === 'int' || config.type === 'float') {
            const step = config.type === 'int' ? '1' : '0.001';
            html += `
                <input type="number" 
                       class="form-control" 
                       id="hp-${key}" 
                       value="${config.default}"
                       min="${config.min || 0}"
                       max="${config.max || 9999}"
                       step="${step}">
            `;
        } else if (config.type === 'select') {
            html += `
                <select class="form-select" id="hp-${key}">
            `;
            config.options.forEach(option => {
                const selected = option === config.default ? 'selected' : '';
                html += `<option value="${option}" ${selected}>${option}</option>`;
            });
            html += '</select>';
        }
        
        html += `
                <small class="text-muted">${config.description}</small>
            </div>
        `;
    }
    
    container.innerHTML = html;
    configDiv.classList.remove('d-none');
    
    // Initialiser les infobulles
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Collecter les valeurs des hyperparam\u00e8tres
function collectHyperparameters() {
    const model = availableModels.find(m => m.id === selectedModel);
    if (!model || !model.hyperparameters) {
        return {};
    }
    
    const hyperparameters = {};
    for (const key of Object.keys(model.hyperparameters)) {
        const input = document.getElementById(`hp-${key}`);
        if (input) {
            const value = input.value;
            // Convertir au type appropri\u00e9
            if (model.hyperparameters[key].type === 'int') {
                hyperparameters[key] = parseInt(value);
            } else if (model.hyperparameters[key].type === 'float') {
                hyperparameters[key] = parseFloat(value);
            } else {
                hyperparameters[key] = value;
            }
        }
    }
    
    return hyperparameters;
}

// Entra\u00eener le mod\u00e8le
async function trainModel() {
    if (!selectedModel) {
        showError('Veuillez d\'abord s\u00e9lectionner un mod\u00e8le');
        return;
    }

    // Collecter les hyperparam\u00e8tres
    const hyperparameters = collectHyperparameters();

    // Aller \u00e0 l'\u00e9tape d'entra\u00eenement
    goToStep(3);
    
    // Effacer les \u00e9tapes d'entra\u00eenement pr\u00e9c\u00e9dentes
    const stepsContent = document.getElementById('training-steps-content');
    stepsContent.innerHTML = '<div class="text-muted">D\u00e9marrage de l\'entra\u00eenement...</div>';

    try {
        const response = await fetch('/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                model_id: selectedModel,
                hyperparameters: hyperparameters
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            // Afficher les \u00e9tapes d'entra\u00eenement
            if (data.training_steps && data.training_steps.length > 0) {
                displayTrainingSteps(data.training_steps);
            }
            
            // Afficher les r\u00e9sultats apr\u00e8s un bref d\u00e9lai
            setTimeout(() => {
                displayResults(data);
                goToStep(4);
            }, 1000);
        } else {
            showError(data.error);
            goToStep(2);
        }
    } catch (error) {
        showError('Erreur lors de l\'entra\u00eenement du mod\u00e8le: ' + error.message);
        goToStep(2);
    }
}

// Afficher les \u00e9tapes d'entra\u00eenement
function displayTrainingSteps(steps) {
    const container = document.getElementById('training-steps-content');
    
    let html = '';
    steps.forEach((step, idx) => {
        html += `
            <div class="list-group-item">
                <div class="d-flex align-items-start">
                    <div class="me-3">
                        <i class="fas fa-check-circle text-success"></i>
                    </div>
                    <div>
                        <h6 class="mb-1">\u00c9tape ${idx + 1}</h6>
                        <p class="mb-0 text-muted">${step}</p>
                    </div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Afficher les r\u00e9sultats d'entra\u00eenement
function displayResults(data) {
    const resultsDiv = document.getElementById('results-content');
    
    let html = `
        <div class="alert alert-success">
            <h5><i class="fas fa-check-circle"></i> ${data.message}</h5>
        </div>
    `;

    // Afficher les m\u00e9triques
    if (data.metrics && Object.keys(data.metrics).length > 0) {
        html += '<div class="row mb-4">';
        
        for (const [key, value] of Object.entries(data.metrics)) {
            html += `
                <div class="col-md-6 col-lg-3">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h6 class="text-muted mb-2">${key}</h6>
                            <h4 class="mb-0">${typeof value === 'number' ? value.toFixed(4) : value}</h4>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
    }

    // Afficher le graphique
    if (data.plot) {
        html += `
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title"><i class="fas fa-chart-bar"></i> Visualisation</h5>
                    <img src="${data.plot}" class="img-fluid rounded" alt="R\u00e9sultats du mod\u00e8le">
                </div>
            </div>
        `;
    }

    resultsDiv.innerHTML = html;

    // Afficher l'option de visualisation des couches si disponible
    if (data.can_visualize_layers) {
        document.getElementById('layer-viz-section').classList.remove('d-none');
    } else {
        document.getElementById('layer-viz-section').classList.add('d-none');
    }
}

// Visualiser les couches cach\u00e9es
async function visualizeLayers() {
    showLoading();

    try {
        const response = await fetch('/visualize_layers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            const vizDiv = document.getElementById('layer-viz-content');
            vizDiv.innerHTML = `
                <div class="alert alert-info">
                    <h5><i class="fas fa-lightbulb"></i> Comment le R\u00e9seau de Neurones "Voit"</h5>
                    <p class="mb-0">${data.message}</p>
                </div>
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Activations des Couches</h5>
                        <img src="${data.plot}" class="img-fluid rounded" alt="Activations des couches">
                        <div class="alert alert-light mt-3">
                            <p class="mb-0">
                                <strong>Ce que vous voyez:</strong> Chaque image montre ce que diff\u00e9rents "neurones" 
                                dans les couches CNN d\u00e9tectent - bords, formes, couleurs et motifs. Les premi\u00e8res 
                                couches (haut) d\u00e9tectent des caract\u00e9ristiques simples comme les bords, tandis que les couches plus profondes (bas) 
                                d\u00e9tectent des motifs plus complexes!
                            </p>
                        </div>
                    </div>
                </div>
            `;
            vizDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            hideLoading();
        } else {
            showError(data.error);
            hideLoading();
        }
    } catch (error) {
        showError('Erreur lors de la visualisation des couches: ' + error.message);
        hideLoading();
    }
}

// Recommencer
function startOver() {
    sessionId = null;
    dataType = null;
    selectedModel = null;
    selectedDataset = null;
    availableModels = [];
    
    // Effacer tout le contenu
    document.getElementById('learning-section').classList.add('d-none');
    document.getElementById('answer-display').classList.add('d-none');
    document.getElementById('data-preview').classList.add('d-none');
    document.getElementById('dataset-info').innerHTML = '';
    document.getElementById('preview-content').innerHTML = '';
    document.getElementById('model-selection').innerHTML = '';
    document.getElementById('hyperparameter-config').classList.add('d-none');
    document.getElementById('hyperparameter-inputs').innerHTML = '';
    document.getElementById('training-steps-content').innerHTML = '';
    document.getElementById('results-content').innerHTML = '';
    document.getElementById('layer-viz-content').innerHTML = '';
    document.getElementById('layer-viz-section').classList.add('d-none');
    document.getElementById('train-button').disabled = true;
    
    // Retirer les s\u00e9lections
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Recharger les jeux de donn\u00e9es
    loadDatasets();
    
    // Retourner \u00e0 l'\u00e9tape 1
    goToStep(1);
}

// Afficher la superposition de chargement
function showLoading() {
    document.getElementById('loading-overlay').classList.remove('d-none');
}

// Masquer la superposition de chargement
function hideLoading() {
    document.getElementById('loading-overlay').classList.add('d-none');
}

// Afficher le message d'erreur
function showError(message) {
    alert('Erreur: ' + message);
}
