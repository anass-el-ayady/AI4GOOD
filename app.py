import os
import io
import base64
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_openml
from PIL import Image
import time

# Set TensorFlow environment variables before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Lazy load TensorFlow (only when needed)
tf = None
keras = None
layers = None

def load_tensorflow():
    """Lazy load TensorFlow only when neural network is needed"""
    global tf, keras, layers
    if tf is None:
        print("Loading TensorFlow...")
        import tensorflow as tensorflow_module
        from tensorflow import keras as keras_module
        from tensorflow.keras import layers as layers_module
        tf = tensorflow_module
        keras = keras_module
        layers = layers_module
        print("TensorFlow loaded successfully!")
    return tf, keras, layers

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['MODELS_FOLDER'] = 'pretrained_models'

# Create models directory if it doesn't exist
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Global storage for session data
data_store = {}
pretrained_models = {}

# Dataset definitions
DATASETS = {
    'mnist': {
        'name': 'Chiffres Manuscrits MNIST',
        'description': 'Jeu de données célèbre de 70 000 chiffres manuscrits (0-9)',
        'type': 'image_classification',
        'icon': '✍️',
        'size': 'Images 28×28 en niveaux de gris',
        'classes': 10,
        'task': 'Reconnaître les chiffres manuscrits'
    },
    'iris': {
        'name': 'Fleurs d\'Iris',
        'description': 'Jeu de données classique avec 3 espèces de fleurs d\'iris',
        'type': 'classification',
        'icon': '🌸',
        'size': '150 échantillons, 4 caractéristiques',
        'classes': 3,
        'task': 'Classifier les espèces d\'iris'
    },
    'wine': {
        'name': 'Qualité du Vin',
        'description': 'Classification du vin basée sur les propriétés chimiques',
        'type': 'classification',
        'icon': '🍷',
        'size': '178 échantillons, 13 caractéristiques',
        'classes': 3,
        'task': 'Classifier le type de vin'
    },
    'digits': {
        'name': 'Chiffres (8×8)',
        'description': 'Petit jeu de données de chiffres manuscrits pour apprentissage rapide',
        'type': 'image_classification',
        'icon': '🔢',
        'size': '1 797 échantillons d\'images 8×8',
        'classes': 10,
        'task': 'Reconnaître les chiffres manuscrits'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_datasets', methods=['GET'])
def get_datasets():
    """Return available datasets"""
    # Convert dictionary to list with IDs
    datasets_list = []
    for dataset_id, dataset_info in DATASETS.items():
        dataset_data = dataset_info.copy()
        dataset_data['id'] = dataset_id
        datasets_list.append(dataset_data)
    
    return jsonify({'datasets': datasets_list})

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    """Load a selected dataset and return samples"""
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        
        if dataset_id not in DATASETS:
            return jsonify({'error': 'Invalid dataset'}), 400
        
        # Generate unique session ID
        session_id = str(np.random.randint(1000000, 9999999))
        
        # Load the dataset
        if dataset_id == 'mnist':
            # Load MNIST
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)
            
            # Sample for display (first 20)
            sample_images = []
            for i in range(20):
                img = X[i].reshape(28, 28)
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                plt.tight_layout(pad=0)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                sample_images.append({
                    'image': f'data:image/png;base64,{img_str}',
                    'label': int(y[i])
                })
            
            data_store[session_id] = {
                'dataset_id': dataset_id,
                'type': 'image_classification',
                'X': X,
                'y': y,
                'shape': X.shape
            }
            
            return jsonify({
                'session_id': session_id,
                'dataset_info': DATASETS[dataset_id],
                'samples': sample_images,
                'total_samples': len(X)
            })
        
        elif dataset_id == 'iris':
            iris = load_iris()
            X, y = iris.data, iris.target
            df = pd.DataFrame(X, columns=iris.feature_names)
            df['species'] = [iris.target_names[i] for i in y]
            
            data_store[session_id] = {
                'dataset_id': dataset_id,
                'type': 'classification',
                'X': X,
                'y': y,
                'feature_names': iris.feature_names,
                'target_names': iris.target_names
            }
            
            return jsonify({
                'session_id': session_id,
                'dataset_name': DATASETS[dataset_id]['name'],
                'info': DATASETS[dataset_id]['description'],
                'data_type': 'tabular',
                'sample_data': df.head(20).to_html(classes='table table-striped', index=False),
                'features': list(iris.feature_names),
                'classes': list(iris.target_names),
                'total_samples': len(X)
            })
        
        elif dataset_id == 'wine':
            wine = load_wine()
            X, y = wine.data, wine.target
            df = pd.DataFrame(X, columns=wine.feature_names)
            df['wine_class'] = [wine.target_names[i] for i in y]
            
            data_store[session_id] = {
                'dataset_id': dataset_id,
                'type': 'classification',
                'X': X,
                'y': y,
                'feature_names': wine.feature_names,
                'target_names': wine.target_names
            }
            
            return jsonify({
                'session_id': session_id,
                'dataset_name': DATASETS[dataset_id]['name'],
                'info': DATASETS[dataset_id]['description'],
                'data_type': 'tabular',
                'sample_data': df.head(20).to_html(classes='table table-striped', index=False),
                'features': list(wine.feature_names),
                'classes': list(wine.target_names),
                'total_samples': len(X)
            })
        
        elif dataset_id == 'digits':
            digits = load_digits()
            X, y = digits.data, digits.target
            
            # Sample images
            sample_images = []
            for i in range(20):
                img = digits.images[i]
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                plt.tight_layout(pad=0)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                sample_images.append({
                    'image': f'data:image/png;base64,{img_str}',
                    'label': int(y[i])
                })
            
            data_store[session_id] = {
                'dataset_id': dataset_id,
                'type': 'image_classification',
                'X': X,
                'y': y,
                'images': digits.images
            }
            
            return jsonify({
                'session_id': session_id,
                'dataset_name': DATASETS[dataset_id]['name'],
                'info': DATASETS[dataset_id]['description'],
                'data_type': 'image',
                'samples': sample_images,
                'total_samples': len(X)
            })
        
        return jsonify({'error': 'Dataset loading not implemented'}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error loading dataset: {str(e)}'}), 500

@app.route('/get_models', methods=['POST'])
def get_models():
    """Return available models based on data type"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404
        
        data_info = data_store[session_id]
        
        if data_info['type'] == 'classification':
            models = [
                {
                    'id': 'logistic_regression',
                    'name': 'Logistic Regression',
                    'description': 'Simple but powerful linear classifier',
                    'type': 'classification',
                    'icon': '📊',
                    'hyperparameters': {
                        'C': {
                            'name': 'Regularization Strength',
                            'description': 'Controls model complexity. Higher = simpler model',
                            'default': 1.0
                        },
                        'max_iter': {
                            'name': 'Max Iterations',
                            'description': 'Maximum number of training iterations',
                            'default': 100
                        }
                    }
                },
                {
                    'id': 'decision_tree',
                    'name': 'Decision Tree',
                    'description': 'Makes decisions using a tree of if-then rules',
                    'type': 'classification',
                    'icon': '🌳',
                    'hyperparameters': {
                        'max_depth': {
                            'name': 'Maximum Depth',
                            'description': 'How deep the tree can grow. Deeper = more complex',
                            'default': 5
                        },
                        'min_samples_split': {
                            'name': 'Min Samples to Split',
                            'description': 'Minimum samples needed to split a node',
                            'default': 2
                        }
                    }
                },
                {
                    'id': 'random_forest',
                    'name': 'Random Forest',
                    'description': 'Ensemble of decision trees for better accuracy',
                    'type': 'classification',
                    'icon': '🌲',
                    'hyperparameters': {
                        'n_estimators': {
                            'name': 'Number of Trees',
                            'description': 'How many trees to grow. More = better but slower',
                            'default': 10
                        },
                        'max_depth': {
                            'name': 'Tree Depth',
                            'description': 'Maximum depth of each tree',
                            'default': 5
                        }
                    }
                }
            ]
        else:  # image_classification
            models = [
                {
                    'id': 'cnn',
                    'name': 'Convolutional Neural Network',
                    'description': 'Deep learning model specifically designed for images',
                    'type': 'classification',
                    'icon': '🧠',
                    'hyperparameters': {
                        'epochs': {
                            'name': 'Training Epochs',
                            'description': 'How many times to see the entire dataset. More = better learning but slower',
                            'default': 5
                        },
                        'batch_size': {
                            'name': 'Batch Size',
                            'description': 'Number of images to process at once. Larger = faster but more memory',
                            'default': 32
                        },
                        'learning_rate': {
                            'name': 'Learning Rate',
                            'description': 'How fast the model learns. Too high = unstable, too low = slow',
                            'default': 0.001
                        }
                    }
                }
            ]
        
        return jsonify({'models': models})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the selected model with visualization"""
    try:
        data = request.json
        session_id = data.get('session_id')
        model_id = data.get('model_id')
        hyperparams = data.get('hyperparameters', {})
        
        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404
        
        data_info = data_store[session_id]
        X = data_info['X']
        y = data_info['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        training_steps = []
        
        if data_info['type'] == 'classification':
            # Traditional ML models
            if model_id == 'logistic_regression':
                training_steps.append('Initializing Logistic Regression model...')
                
                model = LogisticRegression(
                    C=hyperparams.get('C', 1.0),
                    max_iter=hyperparams.get('max_iter', 100),
                    random_state=42
                )
                
                training_steps.append({
                    'step': 2,
                    'message': 'Training model on data...',
                    'detail': f'Learning from {len(X_train)} training samples'
                })
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                training_steps.append({
                    'step': 3,
                    'message': 'Model training complete!',
                    'detail': f'Achieved {accuracy:.2%} accuracy'
                })
                
                # Visualization
                cm = confusion_matrix(y_test, y_pred)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Predicted Label')
                axes[0].set_ylabel('True Label')
                
                # Feature importance (coefficients)
                if hasattr(model, 'coef_'):
                    feature_names = data_info.get('feature_names', [f'Feature {i}' for i in range(X.shape[1])])
                    importance = np.abs(model.coef_[0])
                    indices = np.argsort(importance)[::-1][:10]
                    
                    axes[1].barh(range(len(indices)), importance[indices], color='skyblue')
                    axes[1].set_yticks(range(len(indices)))
                    axes[1].set_yticklabels([feature_names[i] for i in indices])
                    axes[1].set_xlabel('Importance (Absolute Coefficient)')
                    axes[1].set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
                    axes[1].invert_yaxis()
                
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                return jsonify({
                    'success': True,
                    'metrics': {
                        'Accuracy': f'{accuracy:.4f}',
                        'Training Samples': len(X_train),
                        'Test Samples': len(X_test)
                    },
                    'plot': f'data:image/png;base64,{plot_data}',
                    'training_steps': training_steps,
                    'message': f'Logistic Regression trained! Accuracy: {accuracy:.2%}'
                })
            
            elif model_id == 'decision_tree':
                training_steps.append({
                    'step': 1,
                    'message': 'Building Decision Tree...',
                    'detail': 'Creating tree structure with if-then rules'
                })
                
                model = DecisionTreeClassifier(
                    max_depth=hyperparams.get('max_depth', 5),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    random_state=42
                )
                
                training_steps.append({
                    'step': 2,
                    'message': 'Growing the tree...',
                    'detail': 'Splitting nodes to find best decision rules'
                })
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                training_steps.append({
                    'step': 3,
                    'message': 'Tree construction complete!',
                    'detail': f'Tree has {model.tree_.node_count} nodes and depth {model.tree_.max_depth}'
                })
                
                # Visualization
                cm = confusion_matrix(y_test, y_pred)
                feature_names = data_info.get('feature_names', [f'Feature {i}' for i in range(X.shape[1])])
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
                axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Predicted Label')
                axes[0].set_ylabel('True Label')
                
                # Feature importance
                importance = model.feature_importances_
                indices = np.argsort(importance)[::-1][:10]
                
                axes[1].barh(range(len(indices)), importance[indices], color='lightgreen')
                axes[1].set_yticks(range(len(indices)))
                axes[1].set_yticklabels([feature_names[i] for i in indices])
                axes[1].set_xlabel('Importance')
                axes[1].set_title('Feature Importance', fontsize=14, fontweight='bold')
                axes[1].invert_yaxis()
                
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                return jsonify({
                    'success': True,
                    'metrics': {
                        'Accuracy': f'{accuracy:.4f}',
                        'Tree Depth': int(model.tree_.max_depth),
                        'Number of Nodes': int(model.tree_.node_count)
                    },
                    'plot': f'data:image/png;base64,{plot_data}',
                    'training_steps': training_steps,
                    'message': f'Decision Tree trained! Accuracy: {accuracy:.2%}'
                })
            
            elif model_id == 'random_forest':
                training_steps.append({
                    'step': 1,
                    'message': 'Creating Random Forest...',
                    'detail': f'Building {hyperparams.get("n_estimators", 10)} decision trees'
                })
                
                model = RandomForestClassifier(
                    n_estimators=hyperparams.get('n_estimators', 10),
                    max_depth=hyperparams.get('max_depth', 5),
                    random_state=42
                )
                
                training_steps.append({
                    'step': 2,
                    'message': 'Training ensemble of trees...',
                    'detail': 'Each tree learns from a random subset of data'
                })
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                training_steps.append({
                    'step': 3,
                    'message': 'Forest is ready!',
                    'detail': f'All {model.n_estimators} trees trained successfully'
                })
                
                # Visualization
                cm = confusion_matrix(y_test, y_pred)
                feature_names = data_info.get('feature_names', [f'Feature {i}' for i in range(X.shape[1])])
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0])
                axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Predicted Label')
                axes[0].set_ylabel('True Label')
                
                # Feature importance
                importance = model.feature_importances_
                indices = np.argsort(importance)[::-1][:10]
                
                axes[1].barh(range(len(indices)), importance[indices], color='plum')
                axes[1].set_yticks(range(len(indices)))
                axes[1].set_yticklabels([feature_names[i] for i in indices])
                axes[1].set_xlabel('Importance')
                axes[1].set_title('Feature Importance (Averaged)', fontsize=14, fontweight='bold')
                axes[1].invert_yaxis()
                
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                return jsonify({
                    'success': True,
                    'metrics': {
                        'Accuracy': f'{accuracy:.4f}',
                        'Number of Trees': int(model.n_estimators),
                        'Average Tree Depth': f'{np.mean([tree.tree_.max_depth for tree in model.estimators_]):.1f}'
                    },
                    'plot': f'data:image/png;base64,{plot_data}',
                    'training_steps': training_steps,
                    'message': f'Random Forest trained! Accuracy: {accuracy:.2%}'
                })
        
        elif data_info['type'] == 'image_classification':
            # CNN for images
            if model_id == 'cnn':
                # Load TensorFlow
                tf, keras, layers = load_tensorflow()
                
                training_steps.append({
                    'step': 1,
                    'message': 'Preparing image data...',
                    'detail': f'Reshaping {len(X_train)} images for neural network'
                })
                
                # Reshape data based on dataset
                if data_info['dataset_id'] == 'mnist':
                    X_train_reshaped = X_train.reshape(-1, 28, 28, 1) / 255.0
                    X_test_reshaped = X_test.reshape(-1, 28, 28, 1) / 255.0
                    input_shape = (28, 28, 1)
                elif data_info['dataset_id'] == 'digits':
                    X_train_reshaped = X_train.reshape(-1, 8, 8, 1) / 16.0
                    X_test_reshaped = X_test.reshape(-1, 8, 8, 1) / 16.0
                    input_shape = (8, 8, 1)
                else:
                    return jsonify({'error': 'Unsupported image dataset'}), 400
                
                training_steps.append({
                    'step': 2,
                    'message': 'Building Convolutional Neural Network...',
                    'detail': 'Creating layers: Conv2D → MaxPooling → Conv2D → Dense'
                })
                
                # Build CNN - architecture depends on image size
                if data_info['dataset_id'] == 'mnist':
                    # MNIST: 28x28 - can handle more pooling layers
                    model = keras.Sequential([
                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
                        layers.MaxPooling2D((2, 2), name='pool1'),
                        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
                        layers.MaxPooling2D((2, 2), name='pool2'),
                        layers.Flatten(name='flatten'),
                        layers.Dense(64, activation='relu', name='dense1'),
                        layers.Dropout(0.5, name='dropout'),
                        layers.Dense(10, activation='softmax', name='output')
                    ])
                else:
                    # Digits: 8x8 - simpler architecture for small images
                    model = keras.Sequential([
                        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
                        layers.Conv2D(32, (3, 3), activation='relu', name='conv2'),
                        layers.Flatten(name='flatten'),
                        layers.Dense(64, activation='relu', name='dense1'),
                        layers.Dropout(0.3, name='dropout'),
                        layers.Dense(10, activation='softmax', name='output')
                    ])
                
                lr = hyperparams.get('learning_rate', 0.001)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                training_steps.append({
                    'step': 3,
                    'message': 'Training neural network...',
                    'detail': f'Running {hyperparams.get("epochs", 5)} epochs with batch size {hyperparams.get("batch_size", 32)}'
                })
                
                # Train (using small epochs for demo)
                history = model.fit(
                    X_train_reshaped, y_train,
                    validation_split=0.2,
                    epochs=hyperparams.get('epochs', 5),
                    batch_size=hyperparams.get('batch_size', 32),
                    verbose=0
                )
                
                y_pred = np.argmax(model.predict(X_test_reshaped, verbose=0), axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                
                training_steps.append({
                    'step': 4,
                    'message': 'Training complete!',
                    'detail': f'Final accuracy: {accuracy:.2%}'
                })
                
                # Store model for layer visualization
                data_store[session_id]['trained_model'] = model
                data_store[session_id]['X_test'] = X_test_reshaped
                
                # Visualization
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Training history
                ax1 = fig.add_subplot(gs[0, :2])
                ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
                ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                ax1.set_xlabel('Epoch', fontsize=11)
                ax1.set_ylabel('Accuracy', fontsize=11)
                ax1.set_title('Training Progress', fontsize=13, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Confusion matrix
                ax2 = fig.add_subplot(gs[1:, :2])
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar_kws={'label': 'Count'})
                ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
                ax2.set_xlabel('Predicted Digit')
                ax2.set_ylabel('True Digit')
                
                # Sample predictions
                ax3 = fig.add_subplot(gs[0, 2])
                sample_idx = np.random.randint(0, len(X_test))
                if data_info['dataset_id'] == 'mnist':
                    sample_img = X_test_reshaped[sample_idx].reshape(28, 28)
                else:
                    sample_img = X_test_reshaped[sample_idx].reshape(8, 8)
                ax3.imshow(sample_img, cmap='gray')
                ax3.set_title(f'Predicted: {y_pred[sample_idx]}\nActual: {y_test[sample_idx]}', fontsize=10)
                ax3.axis('off')
                
                # Accuracy by class
                ax4 = fig.add_subplot(gs[1:, 2])
                class_acc = []
                for i in range(10):
                    mask = y_test == i
                    if mask.sum() > 0:
                        class_acc.append(accuracy_score(y_test[mask], y_pred[mask]))
                    else:
                        class_acc.append(0)
                ax4.barh(range(10), class_acc, color='coral')
                ax4.set_yticks(range(10))
                ax4.set_yticklabels([f'Digit {i}' for i in range(10)])
                ax4.set_xlabel('Accuracy')
                ax4.set_title('Per-Digit Accuracy', fontsize=13, fontweight='bold')
                ax4.set_xlim([0, 1])
                ax4.invert_yaxis()
                
                plt.suptitle('CNN Training Results', fontsize=16, fontweight='bold', y=0.995)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                return jsonify({
                    'success': True,
                    'metrics': {
                        'Test Accuracy': f'{accuracy:.4f}',
                        'Training Accuracy': f'{history.history["accuracy"][-1]:.4f}',
                        'Validation Accuracy': f'{history.history["val_accuracy"][-1]:.4f}',
                        'Total Parameters': int(model.count_params())
                    },
                    'plot': f'data:image/png;base64,{plot_data}',
                    'training_steps': training_steps,
                    'message': f'CNN trained successfully! Test Accuracy: {accuracy:.2%}',
                    'can_visualize_layers': True
                })
        
        return jsonify({'error': 'Model not supported'}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error training model: {str(e)}'}), 500

@app.route('/visualize_layers', methods=['POST'])
def visualize_layers():
    """Visualize hidden layer activations for trained CNN models"""
    try:
        # Load TensorFlow when needed
        tf, keras, layers = load_tensorflow()
        
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404
        
        data_info = data_store[session_id]
        
        if 'trained_model' not in data_info:
            return jsonify({'error': 'Please train a CNN model first'}), 400
        
        model = data_info['trained_model']
        X_test = data_info.get('X_test')
        
        if X_test is None:
            return jsonify({'error': 'No test data available'}), 400
        
        # Get a random sample
        sample_idx = np.random.randint(0, len(X_test))
        sample_image = X_test[sample_idx:sample_idx+1]
        
        # Build the model by running a prediction first
        model.predict(sample_image, verbose=0)
        
        # Get layer outputs - find all conv layers
        layer_outputs = []
        layer_info = []
        for layer in model.layers:
            if 'conv' in layer.name or 'pool' in layer.name:
                layer_outputs.append(layer.output)
                layer_info.append(layer.name)
        
        if not layer_outputs:
            return jsonify({'error': 'No convolutional layers found'}), 400
        
        # Create activation model
        activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(sample_image, verbose=0)
        
        # Visualize - dynamic based on number of layers
        num_layers = len(activations)
        fig = plt.figure(figsize=(16, 4 * num_layers))
        
        # Original image
        ax = plt.subplot(num_layers + 1, 9, 1)
        if data_info['dataset_id'] == 'mnist':
            ax.imshow(sample_image[0].reshape(28, 28), cmap='gray')
        else:
            ax.imshow(sample_image[0].reshape(8, 8), cmap='gray')
        ax.set_title('Input Image', fontweight='bold', fontsize=11)
        ax.axis('off')
        
        # Visualize each layer
        colors = ['viridis', 'plasma', 'inferno', 'magma']
        for layer_idx, (activation, layer_name) in enumerate(zip(activations, layer_info)):
            color = colors[layer_idx % len(colors)]
            
            # Show up to 8 feature maps per layer
            num_features = min(8, activation.shape[-1])
            for feature_idx in range(num_features):
                position = (layer_idx + 1) * 9 + feature_idx + 2
                ax = plt.subplot(num_layers + 1, 9, position)
                
                # Handle 2D feature maps
                if len(activation.shape) == 4:
                    ax.imshow(activation[0, :, :, feature_idx], cmap=color)
                ax.set_title(f'{layer_name}\n#{feature_idx+1}', fontsize=8)
                ax.axis('off')
        
        plt.suptitle('🧠 How the CNN "Sees" the Image Through Different Layers', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'plot': f'data:image/png;base64,{plot_data}',
            'message': 'Visualized how the neural network "sees" the image! Each layer detects different features: edges, shapes, and patterns.'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
