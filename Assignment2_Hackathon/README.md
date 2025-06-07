# Land Cover Classification using NDVI Time-Series Data

This project implements a land cover classification system using NDVI (Normalized Difference Vegetation Index) time-series data. The solution uses a Multinomial Logistic Regression model with advanced feature engineering to handle temporal data characteristics and missing values.

## Project Overview

The system classifies land cover types based on NDVI time-series data, which captures vegetation patterns over time. The solution includes comprehensive data preprocessing, feature engineering, and model training pipelines.

## Requirements

- Python 3.8+
- Virtual Environment (recommended)
- Required packages listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Files

Place these files in the project root:
- `hacktrain.csv`: Training dataset with NDVI time-series and land cover classes
- `hacktest.csv`: Test dataset for predictions

## Usage

You can run the solution in two ways:

### 1. Using Jupyter Notebook

```bash
jupyter notebook
```
Open `land_cover_classification.ipynb` and run all cells.

### 2. Using Python Script

```bash
python land_cover_classification.py
```

## Features and Implementation Details

### Data Preprocessing
- Missing value imputation using median strategy
- Feature scaling using StandardScaler
- Automated handling of NaN values

### Feature Engineering
1. **Temporal Features**:
   - Rolling statistics (mean, std) with 3-point window
   - First-order differences between consecutive measurements
   - Seasonal decomposition (4 seasons)

2. **Global Statistics**:
   - Maximum and minimum NDVI values
   - NDVI range
   - Mean and standard deviation
   - Seasonal averages and variations

3. **Quality Assurance**:
   - Automated NaN detection and handling
   - Feature validation steps
   - Robust scaling implementation

### Model Architecture
- Algorithm: Multinomial Logistic Regression
- Solver: 'lbfgs' with increased max_iterations
- Multi-class classification approach
- L2 regularization

## Output

The system generates:
- `submission.csv`: Predictions for test data
- Training accuracy metrics
- Classification report with per-class metrics

## Project Structure
```
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── land_cover_classification.py    # Main Python script
├── land_cover_classification.ipynb # Jupyter notebook version
├── hacktrain.csv          # Training data
├── hacktest.csv           # Test data
└── submission.csv         # Generated predictions
```

## Performance Metrics

The solution provides:
- Overall accuracy score
- Per-class precision, recall, and F1-scores
- Detailed classification report

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed
2. Verify data files are in the correct location
3. Check Python version compatibility
4. Ensure virtual environment is activated

## Notes

- The solution automatically handles missing values and outliers
- Feature engineering is performed consistently across training and test data
- All intermediate steps include validation checks 