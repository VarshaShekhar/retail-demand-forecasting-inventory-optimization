"""
FastAPI Application with Full ML Pipeline
Integrates: Feature Engineering, Vectorization, PCA, Random Forest, Deep Learning, Transformers
Based on CapstoneFinal1 notebook
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

# ============ Preprocessing ============
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============ Vectorization ============
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize
    import gensim.downloader as api
    GENSIM_AVAILABLE = True
except:
    GENSIM_AVAILABLE = False

# ============ Deep Learning ============
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# ============ Transformers (Text Embeddings) ============
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
    from xgboost import XGBClassifier, XGBRegressor
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

# ============ Initialize FastAPI ============
app = FastAPI(
    title="Advanced Retail Predictor API",
    description="Full ML Pipeline: Feature Engineering + Vectorization + PCA + ML/DL/Transformers",
    version="3.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Global Models & Preprocessors ============
models = {
    "mlp_regressor": None,
    "random_forest": None,
    "keras_model": None,
    "transformer_model": None,
    "distilbert_xgb_classifier": None,  # For demand binning
}

preprocessors = {
    "feature_scaler": None,
    "target_scaler": None,
    "pca": None,
    "tfidf_vectorizers": {},  # Per text column
    "word2vec_models": {},    # Per text column
    "label_encoders": {},     # For categorical
    "distilbert_tokenizer": None,
    "distilbert_model": None,
}

# Track metrics from the most recent training run for better prediction selection
model_performance = {}

feature_config = {
    "numeric_cols": [],
    "categorical_cols": [],
    "text_cols": [],
    "vectorization_method": "none",  # 'none', 'tfidf', 'word2vec', 'glove'
    "use_pca": False,
    "pca_variance": 0.97
}

# ============ Request/Response Models ============

class PriceFeatures(BaseModel):
    current_price: float
    base_price: float
    competitor_price: float
    unit_cost: float

class SalesFeatures(BaseModel):
    in_store_sales_units: float
    online_sales_units: float
    website_visits: float
    app_traffic_index: float
    no_of_customer_purchases: float
    footfall_index: float

class PromotionFeatures(BaseModel):
    discount_percentage: float
    promotion_flag: int
    marketing_spend: float
    loyalty_program_usage_count: float

class StoreFeatures(BaseModel):
    no_of_checkout_counters: float

class DateFeatures(BaseModel):
    day_of_week: int
    is_weekend: int
    month: Optional[int] = None
    day: Optional[int] = None

class SocialFeatures(BaseModel):
    social_media_sentiment: float

class CategoryFeatures(BaseModel):
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand_tier: Optional[str] = None
    promotion_type: Optional[str] = None
    store_type: Optional[str] = None

class PredictionInput(BaseModel):
    prices: PriceFeatures
    sales: SalesFeatures
    promotion: PromotionFeatures
    store: StoreFeatures
    date: DateFeatures
    social: SocialFeatures
    categories: Optional[CategoryFeatures] = None

class PredictionOutput(BaseModel):
    daily_units_sold: float
    features_processed: int
    model_used: str

class AdvancedTrainingRequest(BaseModel):
    """
    Advanced training with full pipeline:
    - Feature engineering
    - Vectorization (none, tfidf, word2vec, glove)
    - PCA dimensionality reduction
    - Model ensemble (mlp, rf, keras, transformer)
    """
    features: List[List[float]]
    feature_names: List[str]
    labels: List[float]
    
    # Data handling
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None
    text_cols: Optional[List[str]] = None
    
    # Vectorization
    vectorization_method: str = "none"  # 'none', 'tfidf', 'word2vec', 'glove'
    
    # Dimensionality reduction
    use_pca: bool = False
    pca_variance: float = 0.97
    
    # Models to train
    models_to_train: List[str] = ["mlp_regressor"]  # Can add 'random_forest', 'keras', 'transformer', 'distilbert_xgb'
    
    # Hyperparameters
    epochs: int = 20
    batch_size: int = 32
    test_size: float = 0.2

# ============ Feature Engineering Functions ============

def engineer_features(X: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
    """Apply feature engineering transformations"""
    X_eng = X.copy()
    
    # Handle categorical encoding
    for col in feature_config.get("categorical_cols", []):
        if col in X_eng.columns:
            if col not in preprocessors["label_encoders"]:
                preprocessors["label_encoders"][col] = LabelEncoder()
                X_eng[col] = preprocessors["label_encoders"][col].fit_transform(X_eng[col].astype(str))
            else:
                X_eng[col] = preprocessors["label_encoders"][col].transform(X_eng[col].astype(str))
    
    return X_eng

# ============ Vectorization Functions ============

def apply_tfidf_vectorization(X: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
    """Apply TF-IDF vectorization to text columns"""
    X_vec = X.copy()
    
    for col in feature_config.get("text_cols", []):
        if col not in X_vec.columns:
            continue
            
        if col not in preprocessors["tfidf_vectorizers"]:
            preprocessors["tfidf_vectorizers"][col] = TfidfVectorizer(max_features=50)
            tfidf_matrix = preprocessors["tfidf_vectorizers"][col].fit_transform(X_vec[col].astype(str))
        else:
            tfidf_matrix = preprocessors["tfidf_vectorizers"][col].transform(X_vec[col].astype(str))
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        )
        X_vec = pd.concat([X_vec.drop(columns=[col]), tfidf_df], axis=1)
    
    return X_vec

def apply_word2vec_vectorization(X: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
    """Apply Word2Vec vectorization to text columns"""
    if not GENSIM_AVAILABLE:
        return X
    
    X_vec = X.copy()
    
    for col in feature_config.get("text_cols", []):
        if col not in X_vec.columns:
            continue
        
        sentences = X_vec[col].astype(str).apply(word_tokenize).tolist()
        
        if col not in preprocessors["word2vec_models"]:
            preprocessors["word2vec_models"][col] = Word2Vec(
                sentences, vector_size=50, window=5, min_count=1, workers=4, seed=42
            )
        
        w2v_model = preprocessors["word2vec_models"][col]
        embeddings = []
        for sent in sentences:
            vecs = [w2v_model.wv[word] for word in sent if word in w2v_model.wv]
            embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(50))
        
        w2v_df = pd.DataFrame(embeddings, columns=[f"{col}_w2v_{i}" for i in range(50)])
        X_vec = pd.concat([X_vec.drop(columns=[col]), w2v_df], axis=1)
    
    return X_vec

def apply_glove_vectorization(X: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
    """Apply GloVe vectorization to text columns"""
    if not GENSIM_AVAILABLE:
        return X
    
    try:
        glove_model = api.load("glove-wiki-gigaword-100")
    except:
        return X
    
    X_vec = X.copy()
    
    for col in feature_config.get("text_cols", []):
        if col not in X_vec.columns:
            continue
        
        embeddings = []
        for text in X_vec[col].astype(str):
            tokens = word_tokenize(text)
            vecs = [glove_model[word] for word in tokens if word in glove_model]
            embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(100))
        
        glove_df = pd.DataFrame(embeddings, columns=[f"{col}_glove_{i}" for i in range(100)])
        X_vec = pd.concat([X_vec.drop(columns=[col]), glove_df], axis=1)
    
    return X_vec

# ============ Model Building Functions ============

def build_diamond_model(input_dim: int, dropout_rate: float = 0.2):
    """Build Diamond-shaped Keras model with bounded output"""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential(name="UnifiedDiamondDL_Regression")
    
    model.add(Dense(input_dim * 2, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(input_dim * 3, activation='gelu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(input_dim * 4, activation='swish'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(input_dim * 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(input_dim, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def get_distilbert_embeddings(text_list, batch_size=16):
    """Generate DistilBERT embeddings for text data"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    if preprocessors["distilbert_tokenizer"] is None:
        preprocessors["distilbert_tokenizer"] = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        preprocessors["distilbert_model"] = DistilBertModel.from_pretrained("distilbert-base-uncased")
        preprocessors["distilbert_model"].eval()
    
    tokenizer = preprocessors["distilbert_tokenizer"]
    model = preprocessors["distilbert_model"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # CLS token embedding
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def row_to_text(row):
    """Convert DataFrame row to text representation"""
    return " ".join(row.values.astype(str))

def train_distilbert_xgb_classifier(X, y, n_bins=5):
    """Train DistilBERT + XGBoost classifier for demand binning"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    # Create demand bins
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    # Create text representation
    text_data = X.apply(row_to_text, axis=1).tolist()
    
    # Generate embeddings
    bert_embeddings = get_distilbert_embeddings(text_data)
    
    # Add numeric features if available
    numeric_X = X.select_dtypes(include=[np.number])
    if not numeric_X.empty:
        numeric_X = numeric_X.fillna(0).values
        X_final = np.hstack([bert_embeddings, numeric_X])
    else:
        X_final = bert_embeddings
    
    # Train XGBoost classifier
    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(np.unique(y_binned)),
        random_state=42,
        eval_metric="mlogloss"
    )
    
    xgb_clf.fit(X_final, y_binned)
    return xgb_clf

# ============ Endpoints ============

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "✅ Advanced API is running",
        "api_version": "3.0.0",
        "models_loaded": {k: v is not None for k, v in models.items()},
        "task": "Full ML Pipeline: Feature Engineering + Vectorization + Models"
    }

@app.get("/pipeline-info")
def pipeline_info():
    """Get current pipeline configuration"""
    return {
        "feature_config": feature_config,
        "vectorization_available": {
            "tfidf": True,
            "word2vec": GENSIM_AVAILABLE,
            "glove": GENSIM_AVAILABLE
        },
        "models_available": {
            "mlp_regressor": True,
            "random_forest": True,
            "keras": TF_AVAILABLE,
            "transformer": TRANSFORMERS_AVAILABLE,
            "distilbert_xgb": TRANSFORMERS_AVAILABLE,
        }
    }

@app.post("/train-advanced")
def train_advanced(request: AdvancedTrainingRequest):
    """
    Advanced training with full ML pipeline
    
    Steps:
    1. Data cleaning & feature engineering
    2. Vectorization (optional: TF-IDF, Word2Vec, GloVe)
    3. PCA (optional dimensionality reduction)
    4. Train multiple models: MLP, Random Forest, Keras, Transformer
    """
    global feature_config, models, preprocessors, model_performance
    
    try:
        # Convert to DataFrame
        X = pd.DataFrame(request.features, columns=request.feature_names)
        y = np.array(request.labels)
        
        # Update feature config
        feature_config["numeric_cols"] = request.numeric_cols or []
        feature_config["categorical_cols"] = request.categorical_cols or []
        feature_config["text_cols"] = request.text_cols or []
        feature_config["vectorization_method"] = request.vectorization_method
        feature_config["use_pca"] = request.use_pca
        
        # ========== Step 1: Feature Engineering ==========
        X_eng = engineer_features(X, feature_config)
        
        # ========== Step 2: Vectorization ==========
        X_vec = X_eng.copy()
        
        if request.vectorization_method == "tfidf":
            X_vec = apply_tfidf_vectorization(X_vec, feature_config)
        elif request.vectorization_method == "word2vec":
            X_vec = apply_word2vec_vectorization(X_vec, feature_config)
        elif request.vectorization_method == "glove":
            X_vec = apply_glove_vectorization(X_vec, feature_config)
        
        # Keep only numeric columns for scaling
        X_numeric = X_vec.select_dtypes(include=['number']).copy()
        
        # ========== Step 3: Feature Scaling ==========
        if preprocessors["feature_scaler"] is None:
            preprocessors["feature_scaler"] = StandardScaler()
            X_scaled = preprocessors["feature_scaler"].fit_transform(X_numeric)
        else:
            X_scaled = preprocessors["feature_scaler"].transform(X_numeric)
        
        # Target scaling
        if preprocessors["target_scaler"] is None:
            preprocessors["target_scaler"] = StandardScaler()
            y_scaled = preprocessors["target_scaler"].fit_transform(y.reshape(-1, 1)).flatten()
        else:
            y_scaled = preprocessors["target_scaler"].transform(y.reshape(-1, 1)).flatten()
        
        # ========== Step 4: PCA (Optional) ==========
        if request.use_pca:
            if preprocessors["pca"] is None:
                preprocessors["pca"] = PCA(n_components=request.pca_variance)
                X_scaled = preprocessors["pca"].fit_transform(X_scaled)
            else:
                X_scaled = preprocessors["pca"].transform(X_scaled)
        
        # ========== Step 5: Train-Test-Validation Split ==========
        # First split: 5% test, 95% train+validation
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.05, random_state=42
        )
        # Second split: from remaining 95%, take 5.26% as validation (5% of original) and 89.74% as train (90% of original)
        # Calculation: val_size = 0.05/0.95 ≈ 0.0526
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.05/0.95, random_state=42
        )
        
        # ========== Step 6: Train Models ==========
        results = {}
        
        # MLP Regressor
        if "mlp_regressor" in request.models_to_train:
            models["mlp_regressor"] = MLPRegressor(
                hidden_layer_sizes=(X_train.shape[1]*2, X_train.shape[1]*3, X_train.shape[1]*2),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
            models["mlp_regressor"].fit(X_train, y_train)
            y_pred = models["mlp_regressor"].predict(X_test)
            results["mlp_regressor"] = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
        
        # Random Forest
        if "random_forest" in request.models_to_train:
            models["random_forest"] = RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )
            models["random_forest"].fit(X_train, y_train)
            y_pred = models["random_forest"].predict(X_test)
            results["random_forest"] = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
        
        # Keras Deep Learning
        if "keras" in request.models_to_train and TF_AVAILABLE:
            models["keras_model"] = build_diamond_model(X_train.shape[1])
            models["keras_model"].fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=request.epochs,
                batch_size=request.batch_size,
                verbose=0
            )
            y_pred = models["keras_model"].predict(X_test, verbose=0).flatten()
            results["keras"] = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
        
        # Transformer Model
        if "transformer" in request.models_to_train and TRANSFORMERS_AVAILABLE:
            try:
                # For now, use a simple transformer-based approach
                # This could be enhanced with proper transformer architecture
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                # Load a pre-trained transformer model
                model_name = "distilbert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                transformer_model = AutoModel.from_pretrained(model_name)
                
                # Create a simple regression head on top of transformer
                class TransformerRegressor(nn.Module):
                    def __init__(self, transformer_model, input_dim, hidden_dim=256):
                        super().__init__()
                        self.transformer = transformer_model
                        self.regressor = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(hidden_dim, 1)
                        )
                    
                    def forward(self, x):
                        # For numeric features, we'll use a simple approach
                        # In a real implementation, you'd integrate text features properly
                        return self.regressor(x)
                
                # Since we don't have text features in this simple case,
                # we'll create a basic neural network that could be extended
                models["transformer_model"] = nn.Sequential(
                    nn.Linear(X_train.shape[1], 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1)
                )
                
                # Simple training loop (this should be improved)
                optimizer = torch.optim.Adam(models["transformer_model"].parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
                
                # Train for a few epochs
                for epoch in range(min(request.epochs, 10)):  # Limit epochs for transformer
                    models["transformer_model"].train()
                    optimizer.zero_grad()
                    outputs = models["transformer_model"](X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                models["transformer_model"].eval()
                with torch.no_grad():
                    y_pred_tensor = models["transformer_model"](X_test_tensor)
                    y_pred = y_pred_tensor.numpy().flatten()
                
                results["transformer"] = {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred))
                }
                
            except Exception as e:
                results["transformer"] = {"error": f"Failed to train transformer model: {str(e)}"}
        
        # DistilBERT + XGBoost Classifier
        if "distilbert_xgb" in request.models_to_train and TRANSFORMERS_AVAILABLE:
            try:
                # Convert back to DataFrame for text processing
                X_df = pd.DataFrame(request.features, columns=request.feature_names)
                models["distilbert_xgb_classifier"] = train_distilbert_xgb_classifier(X_df, y)
                
                if models["distilbert_xgb_classifier"] is not None:
                    # Simple evaluation (would need proper test split)
                    results["distilbert_xgb"] = {"status": "trained"}
                else:
                    results["distilbert_xgb"] = {"error": "Failed to train DistilBERT + XGBoost"}
                    
            except Exception as e:
                results["distilbert_xgb"] = {"error": f"Failed to train DistilBERT + XGBoost: {str(e)}"}
        
        if "mlp_regressor" in request.models_to_train:
            model_performance["mlp_regressor"] = results.get("mlp_regressor")
        if "random_forest" in request.models_to_train:
            model_performance["random_forest"] = results.get("random_forest")
        if "keras" in request.models_to_train and TF_AVAILABLE:
            model_performance["keras_model"] = results.get("keras")
        if "transformer" in request.models_to_train and TRANSFORMERS_AVAILABLE:
            model_performance["transformer_model"] = results.get("transformer")
        if "distilbert_xgb" in request.models_to_train and TRANSFORMERS_AVAILABLE:
            model_performance["distilbert_xgb_classifier"] = results.get("distilbert_xgb")

        return {
            "status": "success",
            "message": "Advanced training completed",
            "pipeline": {
                "feature_engineering": "applied",
                "vectorization": request.vectorization_method,
                "pca": request.use_pca,
                "models_trained": request.models_to_train
            },
            "results": results,
            "data_shape": {
                "original": (len(X), len(request.feature_names)),
                "after_vectorization": X_numeric.shape,
                "train_test_validation_split": {
                    "train": X_train.shape,
                    "validation": X_val.shape,
                    "test": X_test.shape,
                    "percentages": "90% train, 5% validation, 5% test"
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Simple prediction using best available model"""
    global models, preprocessors, model_performance
    
    try:
        if all(v is None for v in models.values()):
            raise HTTPException(status_code=503, detail="No models trained. Use POST /train-advanced first")
        
        # Extract numeric features (18 features)
        features = [
            input_data.prices.current_price,
            input_data.prices.base_price,
            input_data.prices.competitor_price,
            input_data.prices.unit_cost,
            input_data.sales.in_store_sales_units,
            input_data.sales.online_sales_units,
            input_data.sales.website_visits,
            input_data.sales.app_traffic_index,
            input_data.sales.no_of_customer_purchases,
            input_data.sales.footfall_index,
            input_data.promotion.discount_percentage,
            input_data.promotion.promotion_flag,
            input_data.promotion.marketing_spend,
            input_data.promotion.loyalty_program_usage_count,
            input_data.store.no_of_checkout_counters,
            input_data.date.day_of_week,
            input_data.date.is_weekend,
            input_data.social.social_media_sentiment
        ]
        
        X = np.array(features).reshape(1, -1)
        
        # Process text features if provided
        text_features = []
        if input_data.categories:
            categories_dict = input_data.categories.dict(exclude_none=True)
            if categories_dict and preprocessors.get("tfidf_vectorizers"):
                text_cols = list(preprocessors["tfidf_vectorizers"].keys())
                for col in text_cols:
                    text_value = categories_dict.get(col, f"unknown_{col}")
                    if preprocessors["tfidf_vectorizers"][col]:
                        try:
                            text_vec = preprocessors["tfidf_vectorizers"][col].transform([str(text_value)]).toarray()
                            text_features.extend(text_vec[0].tolist())
                        except:
                            pass
                
                if text_features:
                    X = np.concatenate([X, np.array(text_features).reshape(1, -1)], axis=1)
        
        # Scale features
        if preprocessors["feature_scaler"]:
            X_scaled = preprocessors["feature_scaler"].transform(X)
        else:
            X_scaled = X
        
        # Apply PCA if fitted
        if preprocessors["pca"]:
            X_scaled = preprocessors["pca"].transform(X_scaled)
        
        # Choose the best available model based on the latest training performance metrics
        default_order = ["keras_model", "transformer_model", "mlp_regressor", "random_forest", "distilbert_xgb_classifier"]
        best_model_key = None
        if model_performance:
            best_r2 = float("-inf")
            for key in ["random_forest", "mlp_regressor", "keras_model", "transformer_model"]:
                perf = model_performance.get(key)
                if isinstance(perf, dict) and perf.get("r2") is not None:
                    if perf["r2"] > best_r2:
                        best_r2 = perf["r2"]
                        best_model_key = key
        model_order = default_order
        if best_model_key in default_order:
            model_order = [best_model_key] + [m for m in default_order if m != best_model_key]

        model_used = None
        prediction_scaled = None
        for model_key in model_order:
            if models.get(model_key) is None:
                continue

            try:
                if model_key == "keras_model":
                    prediction_scaled = float(models["keras_model"].predict(X_scaled, verbose=0)[0][0])
                    model_used = "keras"
                elif model_key == "transformer_model":
                    import torch
                    models["transformer_model"].eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_scaled)
                        prediction_tensor = models["transformer_model"](X_tensor)
                        prediction_scaled = float(prediction_tensor.item())
                    model_used = "transformer"
                elif model_key == "mlp_regressor":
                    prediction_scaled = float(models["mlp_regressor"].predict(X_scaled)[0])
                    model_used = "mlp_regressor"
                elif model_key == "random_forest":
                    prediction_scaled = float(models["random_forest"].predict(X_scaled)[0])
                    model_used = "random_forest"
            except Exception:
                continue

            if prediction_scaled is None or not np.isfinite(prediction_scaled):
                prediction_scaled = None
                continue

            if model_key == "keras_model" and prediction_scaled <= 0 and any(models.get(k) is not None for k in ["transformer_model", "mlp_regressor", "random_forest"]):
                prediction_scaled = None
                model_used = None
                continue

            break

        # Unscale prediction
        if preprocessors["target_scaler"] and prediction_scaled is not None:
            prediction = float(preprocessors["target_scaler"].inverse_transform([[prediction_scaled]])[0][0])
        else:
            prediction = prediction_scaled
        
        # Ensure prediction is reasonable (check for extreme values from poorly trained keras model)
        if prediction is None or not np.isfinite(prediction):
            prediction = 0
        elif prediction < 0:
            prediction = 0
        elif prediction > 100000:  # Sanity cap for retail units (daily)
            # If keras model produces extreme values, try fallback models
            if model_used == "keras" and any(models.get(k) is not None for k in ["mlp_regressor", "random_forest", "transformer_model"]):
                prediction = 0
                model_used = "keras_invalid"
        
        return {
            "daily_units_sold": round(max(0, prediction), 2),
            "features_processed": len(features) + len(text_features),
            "numeric_features": len(features),
            "text_features": len(text_features),
            "model_used": model_used or "none"
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
def batch_predict(predictions: List[PredictionInput]):
    """Batch prediction"""
    results = []
    for input_data in predictions:
        try:
            result = predict(input_data)
            results.append(result)
        except HTTPException as e:
            results.append({"error": str(e.detail)})
    
    return {
        "predictions": results,
        "count": len(results),
        "status": "completed"
    }

@app.get("/feature-importance")
def get_feature_importance():
    """Get feature importance from Random Forest model"""
    if models["random_forest"] is None:
        raise HTTPException(status_code=503, detail="Random Forest model not trained")
    
    feature_names = [f"feature_{i}" for i in range(len(models["random_forest"].feature_importances_))]
    importance = models["random_forest"].feature_importances_.tolist()
    
    return {
        "model": "RandomForest",
        "feature_importance": dict(zip(feature_names, importance)),
        "top_5": dict(sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5])
    }

class ClassificationInput(BaseModel):
    features: List[float]
    feature_names: List[str]

class ClassificationOutput(BaseModel):
    demand_bin: int
    confidence: float
    model_used: str

@app.post("/predict-demand-bin", response_model=ClassificationOutput)
def predict_demand_bin(input_data: ClassificationInput):
    """Predict demand bin using DistilBERT + XGBoost classifier"""
    if models["distilbert_xgb_classifier"] is None:
        raise HTTPException(status_code=503, detail="DistilBERT + XGBoost classifier not trained")
    
    try:
        # Convert to DataFrame
        X = pd.DataFrame([input_data.features], columns=input_data.feature_names)
        
        # Create text representation
        text_data = X.apply(row_to_text, axis=1).tolist()
        
        # Generate embeddings
        bert_embeddings = get_distilbert_embeddings(text_data)
        
        # Add numeric features
        numeric_X = X.select_dtypes(include=[np.number])
        if not numeric_X.empty:
            numeric_X = numeric_X.fillna(0).values
            X_final = np.hstack([bert_embeddings, numeric_X])
        else:
            X_final = bert_embeddings
        
        # Predict
        prediction = models["distilbert_xgb_classifier"].predict(X_final)[0]
        probabilities = models["distilbert_xgb_classifier"].predict_proba(X_final)[0]
        confidence = float(max(probabilities))
        
        return {
            "demand_bin": int(prediction),
            "confidence": confidence,
            "model_used": "distilbert_xgb"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/clear-models")
def clear_models():
    """Clear all trained models and reset preprocessors"""
    global models, preprocessors, feature_config, model_performance
    
    # Reset models
    models = {
        "mlp_regressor": None,
        "random_forest": None,
        "keras_model": None,
        "transformer_model": None,
        "distilbert_xgb_classifier": None,
    }
    
    # Reset preprocessors
    preprocessors = {
        "feature_scaler": None,
        "target_scaler": None,
        "pca": None,
        "tfidf_vectorizers": {},
        "word2vec_models": {},
        "label_encoders": {},
        "distilbert_tokenizer": None,
        "distilbert_model": None,
    }
    
    # Reset model performance metrics
    model_performance.clear()
    
    # Reset feature config
    feature_config = {
        "numeric_cols": [],
        "categorical_cols": [],
        "text_cols": [],
        "vectorization_method": "none",
        "use_pca": False,
        "pca_variance": 0.97
    }
    
    return {
        "status": "success",
        "message": "All models and preprocessors have been cleared",
        "models_reset": list(models.keys()),
        "preprocessors_reset": list(preprocessors.keys())
    }

@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    """Global exception handler"""
    return {
        "status": "error",
        "detail": str(exc),
        "type": type(exc).__name__
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
