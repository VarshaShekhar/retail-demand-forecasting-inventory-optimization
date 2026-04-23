# Retail Demand Forecasting FastAPI - Advanced

A comprehensive FastAPI application for retail demand prediction with advanced machine learning models, feature engineering, vectorization, and deep learning architectures.

## 🚀 Features

- **Advanced Feature Engineering**
  - PCA (Principal Component Analysis) for dimensionality reduction
  - TF-IDF vectorization for text features
  - Word2Vec embeddings
  - Sentence Transformers (Hugging Face) for semantic embeddings

- **Multiple ML Models**
  - Hybrid: Advanced hybrid model
  - XGBoost & Random Forest compatibility

- **REST API Endpoints**
  - Single & batch predictions
  - Feature extraction and transformation
  - Text embeddings
  - Model training
  - Data upload and processing
  - Statistics and analysis

- **Comprehensive Data Pipeline**
  - Data preprocessing and cleaning
  - Feature engineering
  - Model training and evaluation
  - Cross-validation support
  - Hyperparameter tuning

## 📋 Prerequisites

- Python 3.11.9
- pip (Python package manager)
- ~2GB disk space
- 4GB RAM minimum

## 📦 Installation

### Option 1: Automated Setup (Recommended)

#### Windows
Double-click the `start_api.bat` file or run:
```bash
start_api.bat
```

#### Linux/Mac
```bash
chmod +x start_api.sh
./start_api.sh
```

This will automatically:
- Create a virtual environment
- Install all dependencies from `requirements.txt`
- Create required directories (`models/`, `data/`, `logs/`)
- Start the API server

### Option 2: Manual Setup

1. **Clone/Download the repository and navigate to the project directory:**
```bash
cd my_api
```

2. **Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create required directories:**
```bash
mkdir models data logs
```

## 🚀 Running the API

### Using the Automated Script
```bash
# Windows
python run_server.py

# Or specify custom host/port
python run_server.py 127.0.0.1 8000 true
```

### Manual Start
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Start the server
uvicorn app_advanced:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://127.0.0.1:8000`

Interactive API documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON:** `http://127.0.0.1:8000/openapi.json`

## 📡 API Endpoints

### Health & Information
- **GET `/`** - Health check and API status
- **GET `/pipeline-info`** - Detailed pipeline configuration and capabilities
- **GET `/model-info`** - Current model status and configuration
- **GET `/feature-importance`** - Feature importance scores (if Random Forest trained)

### Predictions
- **POST `/predict`** - Make a single prediction with named features
- **POST `/batch-predict`** - Make multiple predictions at once

### Training
- **POST `/train-advanced`** - Advanced training with full ML pipeline (vectorization, PCA, multiple models)
- **POST `/clear-models`** - Clear all trained models and reset preprocessors (use when changing feature sets)

## 💡 Usage Examples

### 1. Health Check
```bash
curl http://127.0.0.1:8000/
```

**Response:**
```json
{
  "status": "✅ Advanced API is running",
  "api_version": "3.0.0",
  "models_loaded": {
    "mlp_regressor": false,
    "random_forest": false,
    "keras_model": false,
    "transformer_model": false
  },
  "task": "Full ML Pipeline: Feature Engineering + Vectorization + Models"
}
```

### 2. Get Pipeline Information
```bash
curl http://127.0.0.1:8000/pipeline-info
```

### 3. Make a Single Prediction

**Endpoint:** `POST /predict`

**Sample Input:**
```json
{
  "prices": {
    "current_price": 50.0,
    "base_price": 45.0,
    "competitor_price": 48.0,
    "unit_cost": 20.0
  },
  "sales": {
    "in_store_sales_units": 100,
    "online_sales_units": 50,
    "website_visits": 1000,
    "app_traffic_index": 500,
    "no_of_customer_purchases": 150,
    "footfall_index": 300
  },
  "promotion": {
    "discount_percentage": 10.0,
    "promotion_flag": 1,
    "marketing_spend": 5000.0,
    "loyalty_program_usage_count": 200
  },
  "store": {
    "no_of_checkout_counters": 8
  },
  "date": {
    "day_of_week": 3,
    "is_weekend": 0
  },
  "social": {
    "social_media_sentiment": 0.8
  },
  "categories": {
    "product_name": "Premium Coffee Maker",
    "brand_name": "BrandX",
    "category": "Electronics",
    "sub_category": "Kitchen Appliances",
    "brand_tier": "premium",
    "promotion_type": "seasonal_sale",
    "store_type": "flagship"
  }
}
```

**Curl Command:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prices": {
      "current_price": 50.0,
      "base_price": 45.0,
      "competitor_price": 48.0,
      "unit_cost": 20.0
    },
    "sales": {
      "in_store_sales_units": 100,
      "online_sales_units": 50,
      "website_visits": 1000,
      "app_traffic_index": 500,
      "no_of_customer_purchases": 150,
      "footfall_index": 300
    },
    "promotion": {
      "discount_percentage": 10.0,
      "promotion_flag": 1,
      "marketing_spend": 5000.0,
      "loyalty_program_usage_count": 200
    },
    "store": {
      "no_of_checkout_counters": 8
    },
    "date": {
      "day_of_week": 3,
      "is_weekend": 0
    },
    "social": {
      "social_media_sentiment": 0.8
    },
    "categories": {
      "product_name": "Premium Coffee Maker",
      "brand_name": "BrandX",
      "category": "Electronics",
      "sub_category": "Kitchen Appliances",
      "brand_tier": "premium",
      "promotion_type": "seasonal_sale",
      "store_type": "flagship"
    }
  }'
```

**Sample Response (with text features):**
```json
{
  "daily_units_sold": 245.67,
  "numeric_features": 18,
  "text_features": 350,
  "features_processed": 368,
  "model_used": "random_forest"
}
```

**Note:** Text features are optional. If not provided, the API will only use the 18 numeric features. When text features are included and if the model was trained with text vectorization (TF-IDF), they will be automatically processed and included in the prediction.

### 4. Batch Predictions

**Endpoint:** `POST /batch-predict`

**Sample Input (Array of prediction inputs):**
```json
[
  {
    "prices": {
      "current_price": 50.0,
      "base_price": 45.0,
      "competitor_price": 48.0,
      "unit_cost": 20.0
    },
    "sales": {
      "in_store_sales_units": 100,
      "online_sales_units": 50,
      "website_visits": 1000,
      "app_traffic_index": 500,
      "no_of_customer_purchases": 150,
      "footfall_index": 300
    },
    "promotion": {
      "discount_percentage": 10.0,
      "promotion_flag": 1,
      "marketing_spend": 5000.0,
      "loyalty_program_usage_count": 200
    },
    "store": {
      "no_of_checkout_counters": 8
    },
    "date": {
      "day_of_week": 3,
      "is_weekend": 0
    },
    "social": {
      "social_media_sentiment": 0.8
    }
  },
  {
    "prices": {
      "current_price": 55.0,
      "base_price": 50.0,
      "competitor_price": 52.0,
      "unit_cost": 22.0
    },
    "sales": {
      "in_store_sales_units": 120,
      "online_sales_units": 60,
      "website_visits": 1200,
      "app_traffic_index": 600,
      "no_of_customer_purchases": 180,
      "footfall_index": 350
    },
    "promotion": {
      "discount_percentage": 5.0,
      "promotion_flag": 0,
      "marketing_spend": 3000.0,
      "loyalty_program_usage_count": 150
    },
    "store": {
      "no_of_checkout_counters": 6
    },
    "date": {
      "day_of_week": 5,
      "is_weekend": 1
    },
    "social": {
      "social_media_sentiment": 0.6
    }
  }
]
```

**Curl Command:**
```bash
curl -X POST "http://127.0.0.1:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "prices": {"current_price": 50.0, "base_price": 45.0, "competitor_price": 48.0, "unit_cost": 20.0},
      "sales": {"in_store_sales_units": 100, "online_sales_units": 50, "website_visits": 1000, "app_traffic_index": 500, "no_of_customer_purchases": 150, "footfall_index": 300},
      "promotion": {"discount_percentage": 10.0, "promotion_flag": 1, "marketing_spend": 5000.0, "loyalty_program_usage_count": 200},
      "store": {"no_of_checkout_counters": 8},
      "date": {"day_of_week": 3, "is_weekend": 0},
      "social": {"social_media_sentiment": 0.8}
    },
    {
      "prices": {"current_price": 55.0, "base_price": 50.0, "competitor_price": 52.0, "unit_cost": 22.0},
      "sales": {"in_store_sales_units": 120, "online_sales_units": 60, "website_visits": 1200, "app_traffic_index": 600, "no_of_customer_purchases": 180, "footfall_index": 350},
      "promotion": {"discount_percentage": 5.0, "promotion_flag": 0, "marketing_spend": 3000.0, "loyalty_program_usage_count": 150},
      "store": {"no_of_checkout_counters": 6},
      "date": {"day_of_week": 5, "is_weekend": 1},
      "social": {"social_media_sentiment": 0.6}
    }
  ]'
```

**Sample Response:**
```json
{
  "predictions": [
    {"daily_units_sold": 245.67, "features_processed": 18},
    {"daily_units_sold": 312.45, "features_processed": 18}
  ],
  "count": 2,
  "status": "completed"
}
```

### 5. Train Advanced Model

**Endpoint:** `POST /train-advanced`

**Important Note:** The feature names used during training must match exactly when making predictions. If you switch datasets or feature sets, use the `/clear-models` endpoint first to reset all models.

**Sample Input with Valid Retail Data (20 samples - Extended Dataset):**
```json
{
  "features": [
    [45.99, 42.50, 47.25, 18.50, 85, 42, 850, 425, 120, 250, 8.5, 1, 4200.0, 180, 6, 2, 0, 0.75],
    [52.49, 48.75, 53.99, 21.25, 110, 55, 1100, 550, 165, 320, 12.0, 1, 5800.0, 240, 8, 4, 0, 0.82],
    [38.75, 36.25, 40.99, 15.75, 65, 32, 650, 325, 95, 185, 5.5, 0, 3100.0, 130, 5, 6, 1, 0.68],
    [61.99, 57.50, 64.49, 25.75, 145, 72, 1450, 725, 210, 410, 15.0, 1, 8200.0, 340, 10, 1, 0, 0.88],
    [49.25, 45.99, 51.75, 19.99, 95, 47, 950, 475, 140, 275, 10.0, 1, 5100.0, 210, 7, 5, 1, 0.79],
    [42.50, 39.99, 44.75, 17.25, 78, 39, 780, 390, 115, 225, 7.0, 0, 3800.0, 160, 6, 3, 0, 0.71],
    [55.75, 51.99, 57.25, 22.99, 125, 62, 1250, 625, 185, 360, 13.5, 1, 6700.0, 280, 9, 7, 1, 0.85],
    [47.99, 44.75, 49.49, 20.25, 92, 46, 920, 460, 135, 265, 9.0, 1, 4700.0, 195, 7, 2, 0, 0.77],
    [58.25, 53.99, 60.75, 24.50, 135, 67, 1350, 675, 200, 390, 14.0, 1, 7500.0, 310, 9, 4, 0, 0.86],
    [41.75, 38.99, 43.25, 16.75, 72, 36, 720, 360, 105, 205, 6.5, 0, 3500.0, 145, 5, 6, 1, 0.69],
    [63.49, 59.25, 66.99, 26.75, 155, 77, 1550, 775, 230, 445, 16.5, 1, 8900.0, 370, 11, 1, 0, 0.91],
    [44.25, 41.49, 45.75, 18.25, 82, 41, 820, 410, 125, 240, 8.0, 1, 4000.0, 170, 6, 3, 0, 0.73],
    [50.99, 47.25, 52.49, 21.50, 105, 52, 1050, 525, 155, 300, 11.0, 1, 5400.0, 225, 8, 5, 1, 0.80],
    [39.99, 37.25, 41.49, 16.25, 68, 34, 680, 340, 100, 195, 6.0, 0, 3300.0, 140, 5, 7, 1, 0.70],
    [56.75, 52.99, 58.25, 23.75, 130, 65, 1300, 650, 190, 370, 13.0, 1, 7100.0, 295, 9, 2, 0, 0.84],
    [479.2, 392.94, 487.36, 19, 1008, 409, 7003, 288, 159.4, 557, 9.9, 1, 397.08, 86, 6, 0, 1, 0.34],
    [52.75, 49.25, 54.50, 21.75, 115, 58, 1150, 575, 170, 330, 11.5, 1, 5950.0, 250, 8, 3, 0, 0.80],
    [46.50, 43.75, 48.99, 19.25, 98, 49, 980, 490, 145, 285, 9.5, 1, 4850.0, 205, 7, 4, 1, 0.76],
    [60.25, 56.50, 62.75, 24.75, 140, 70, 1400, 700, 205, 400, 14.5, 1, 8000.0, 330, 10, 1, 0, 0.87],
    [43.75, 41.00, 45.25, 17.75, 80, 40, 800, 400, 120, 235, 7.5, 0, 3900.0, 165, 6, 2, 0, 0.72],
    [485.0, 395.0, 490.0, 20, 1015, 415, 6950, 280, 160, 550, 10.0, 1, 400.0, 85, 6, 1, 1, 0.35],
    [475.0, 390.0, 485.0, 18, 1000, 400, 7050, 295, 158, 565, 9.5, 1, 395.0, 88, 6, 2, 0, 0.32],
    [481.0, 394.0, 488.0, 19.5, 1010, 412, 7000, 290, 161, 560, 9.8, 1, 398.0, 87, 6, 0, 1, 0.36],
    [478.0, 391.0, 486.0, 18.5, 1005, 408, 6980, 285, 159.5, 555, 9.7, 1, 396.0, 86.5, 6, 1, 0, 0.33],
    [73.51, 65.42, 67.65, 15, 12, 172, 1680, 300, 72.8, 103, 2.7, 1, 41.17, 96, 33, 3, 0, 0.42],
    [74.0, 66.0, 68.0, 15.5, 13, 175, 1700, 310, 73.0, 105, 2.8, 1, 42.0, 97, 33, 2, 0, 0.41],
    [73.0, 65.0, 67.0, 14.5, 11, 170, 1650, 290, 72.0, 100, 2.6, 1, 40.0, 95, 33, 4, 1, 0.43],
    [73.75, 65.75, 67.75, 15.25, 12.5, 173, 1680, 305, 72.5, 104, 2.7, 1, 41.5, 96.5, 33, 3, 0, 0.42]
  ],
  "feature_names": [
    "current_price",
    "base_price", 
    "competitor_price",
    "unit_cost",
    "in_store_sales_units",
    "online_sales_units",
    "website_visits",
    "app_traffic_index",
    "no_of_customer_purchases",
    "footfall_index",
    "discount_percentage",
    "promotion_flag",
    "marketing_spend",
    "loyalty_program_usage_count",
    "no_of_checkout_counters",
    "day_of_week",
    "is_weekend",
    "social_media_sentiment"
  ],
  "labels": [185.5, 245.8, 142.3, 312.7, 198.4, 168.9, 278.6, 192.1, 295.3, 155.7, 335.2, 178.4, 215.9, 148.6, 265.8, 70, 250.5, 201.2, 305.6, 172.3, 68, 72, 69, 71, 100, 102, 98, 101],
  "numeric_cols": [
    "current_price",
    "base_price",
    "competitor_price", 
    "unit_cost",
    "in_store_sales_units",
    "online_sales_units",
    "website_visits",
    "app_traffic_index",
    "no_of_customer_purchases",
    "footfall_index",
    "discount_percentage",
    "marketing_spend",
    "loyalty_program_usage_count",
    "no_of_checkout_counters",
    "day_of_week",
    "is_weekend",
    "social_media_sentiment"
  ],
  "categorical_cols": [],
  "text_cols": [],
  "extract_date_features": false,
  "vectorization_method": "none",
  "use_pca": false,
  "pca_variance": 0.97,
  "models_to_train": ["mlp_regressor", "random_forest", "keras", "transformer"],
  "epochs": 20,
  "batch_size": 32,
  "test_size": 0.2
}
```

**Sample Response:**
```json
{
  "status": "success",
  "message": "Advanced training completed",
  "pipeline": {
    "feature_engineering": "applied",
    "vectorization": "none",
    "pca": false,
    "models_trained": ["mlp_regressor", "random_forest", "keras", "transformer"]
  },
  "results": {
    "mlp_regressor": {
      "rmse": 0.123,
      "mae": 0.089,
      "r2": 0.945
    },
    "random_forest": {
      "rmse": 0.098,
      "mae": 0.076,
      "r2": 0.967
    },
    "keras": {
      "rmse": 0.085,
      "mae": 0.065,
      "r2": 0.978
    },
    "transformer": {
      "rmse": 0.092,
      "mae": 0.071,
      "r2": 0.972
    }
  },
  "data_shape": {
    "original": [28, 18],
    "after_vectorization": [28, 18],
    "train_test_validation_split": {
      "train": [25, 18],
      "validation": [1, 18],
      "test": [1, 18],
      "percentages": "90% train, 5% validation, 5% test"
    }
  }
}
```

**Quick Test Curl Command (with 6 samples including Bose Audio 1008):**
```bash
curl -X POST "http://127.0.0.1:8000/train-advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [45.99, 42.50, 47.25, 18.50, 85, 42, 850, 425, 120, 250, 8.5, 1, 4200.0, 180, 6, 2, 0, 0.75],
      [52.49, 48.75, 53.99, 21.25, 110, 55, 1100, 550, 165, 320, 12.0, 1, 5800.0, 240, 8, 4, 0, 0.82],
      [38.75, 36.25, 40.99, 15.75, 65, 32, 650, 325, 95, 185, 5.5, 0, 3100.0, 130, 5, 6, 1, 0.68],
      [479.2, 392.94, 487.36, 19, 1008, 409, 7003, 288, 159.4, 557, 9.9, 1, 397.08, 86, 6, 0, 1, 0.34],
      [61.99, 57.50, 64.49, 25.75, 145, 72, 1450, 725, 210, 410, 15.0, 1, 8200.0, 340, 10, 1, 0, 0.88],
      [49.25, 45.99, 51.75, 19.99, 95, 47, 950, 475, 140, 275, 10.0, 1, 5100.0, 210, 7, 5, 1, 0.79]
    ],
    "feature_names": [
      "current_price", "base_price", "competitor_price", "unit_cost",
      "in_store_sales_units", "online_sales_units", "website_visits", "app_traffic_index",
      "no_of_customer_purchases", "footfall_index", "discount_percentage", "promotion_flag",
      "marketing_spend", "loyalty_program_usage_count", "no_of_checkout_counters",
      "day_of_week", "is_weekend", "social_media_sentiment"
    ],
    "labels": [185.5, 245.8, 142.3, 70, 312.7, 198.4],
    "numeric_cols": [
      "current_price", "base_price", "competitor_price", "unit_cost",
      "in_store_sales_units", "online_sales_units", "website_visits", "app_traffic_index",
      "no_of_customer_purchases", "footfall_index", "discount_percentage",
      "marketing_spend", "loyalty_program_usage_count", "no_of_checkout_counters",
      "day_of_week", "is_weekend", "social_media_sentiment"
    ],
    "categorical_cols": [],
    "text_cols": [],
    "extract_date_features": false,
    "vectorization_method": "none",
    "use_pca": false,
    "pca_variance": 0.97,
    "models_to_train": ["mlp_regressor", "random_forest"],
    "epochs": 10,
    "batch_size": 16,
    "test_size": 0.2
  }'
```

### 6. Clear Models

**Endpoint:** `POST /clear-models`

If you encounter feature name mismatch errors when switching between different datasets or feature sets, use this endpoint to reset all models and preprocessors.

**Curl Command:**
```bash
curl -X POST "http://127.0.0.1:8000/clear-models"
```

**Response:**
```json
{
  "status": "success",
  "message": "All models and preprocessors have been cleared",
  "models_reset": ["mlp_regressor", "random_forest", "keras_model", "transformer_model"],
  "preprocessors_reset": ["feature_scaler", "target_scaler", "pca", "tfidf_vectorizers", "word2vec_models", "label_encoders"]
}
```

**When to use:** Before training with different feature names or when switching datasets.

## 🧪 Testing the API

### Option 1: Interactive Testing
1. Open `http://127.0.0.1:8000/docs` in your browser
2. Click on any endpoint
3. Click "Try it out"
4. Fill in the parameters or use the examples above
5. Click "Execute"

### Option 2: Test Scripts
Run the provided test scripts in a new terminal:

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run API prediction test
python test_api_prediction.py

# Run named features test
python test_named_features.py
```

### Option 3: Using Python Requests
```python
import requests

# Single prediction
url = "http://127.0.0.1:8000/predict"
data = {
  "prices": {"current_price": 50.0, "base_price": 45.0, ...},
  "sales": {"in_store_sales_units": 100, ...},
  # ... complete data
}

response = requests.post(url, json=data)
print(response.json())
```

## 📊 Project Structure

```
my_api/
├── app_advanced.py         # Main FastAPI application with feature engineering
├── config.py               # Configuration settings
├── run_server.py          # Server startup script
├── requirements.txt        # Python dependencies
├── requirements-core.txt   # Core dependencies only
├── test_api_prediction.py  # API testing script
├── test_named_features.py  # Feature testing script
├── start_api.bat           # Windows startup script
├── start_api.sh            # Linux/Mac startup script
├── QUICKSTART.md           # Quick start guide
├── DEPLOYMENT.md           # Deployment guide
├── models/                 # Saved model weights and scalers
├── data/                   # Data storage
└── logs/                   # Application logs
```

## 🏗️ Architecture

### Feature Processing Pipeline
1. **Input Features** → Structured data with 18 features
2. **Feature Engineering** → Categorical encoding, date features
3. **Vectorization** → Optional TF-IDF, Word2Vec, or GloVe for text features
4. **Scaling** → StandardScaler for numeric features
5. **PCA** → Optional dimensionality reduction
6. **Model Prediction** → Ensemble of ML/DL models

### Model Options
- **MLP Regressor** (Neural Network with custom architecture)
- **Random Forest** (Ensemble method with feature importance)
- **Keras Deep Learning** (Diamond-shaped neural network)
- **Transformer Model** (PyTorch-based neural network with transformer components)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues
- **Model not loaded**: Train a model first using `/train-advanced`
- **Feature name mismatch**: Use `POST /clear-models` to reset when switching datasets
- **Import errors**: Ensure all dependencies are installed
- **Port conflicts**: Change the port in `run_server.py`

### Getting Help
- Check the API documentation at `/docs`
- Review the logs in the `logs/` directory
- Run the test scripts to verify functionality

## 📈 Model Performance

The notebook analysis showed:
- **Best Embedding**: Transformer (Sentence-BERT)
- **Best Combination**: TF-IDF + PCA
- **Top Features**: In-store sales, pricing variables, online activity
- **R² Score**: 0.95+ on validation set

## 🔐 Security

- CORS enabled for cross-origin requests
- Input validation using Pydantic models
- Global exception handling
- Error logging

## 🛠️ Configuration

Edit `config.py` to customize:
- Model architectures
- Training parameters
- Feature engineering settings
- Device (CPU/GPU)
- Logging levels

## 📚 API Response Examples

### Prediction Response
```json
{
  "prediction": 545.23,
  "model_used": "fttransformer",
  "confidence": 0.95,
  "feature_dimension": 304
}
```

### Health Check Response
```json
{
  "status": "✅ API is running",
  "device": "cuda",
  "models_available": ["PCA", "TF-IDF"]
}
```

## 🚨 Troubleshooting

### CUDA out of memory
- Set `device="cpu"` in config
- Reduce batch size
- Use a lighter model

### Model not found
- Ensure models are trained first
- Check model files in `models/` directory

### Feature dimension mismatch
- Verify feature count matches training data
- Use `/extract-features` endpoint to verify

## 📝 Logging

Logs are saved to `logs/api.log`. Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Model Training Workflow

1. **Prepare Data**: Upload or provide numeric and text features
2. **Feature Engineering**: Apply PCA and embeddings
3. **Train Models**: Use `/train-model` endpoint
4. **Validate**: Check performance metrics
5. **Deploy**: Use trained models for predictions

## 🎯 Next Steps

- [ ] Add database integration (PostgreSQL/MongoDB)
- [ ] Implement model versioning
- [ ] Add real-time monitoring dashboards
- [ ] Enable A/B testing for models
- [ ] Add advanced hyperparameter tuning
- [ ] Implement model explainability (SHAP)

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created for Retail Demand Forecasting - Advanced ML Project

## 📧 Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Note**: Ensure all required data files (CSV with retail data) are properly formatted and in the `data/` directory before training models.
