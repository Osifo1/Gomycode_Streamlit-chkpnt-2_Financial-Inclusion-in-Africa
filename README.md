# Financial Inclusion in Africa

This repository contains a Jupyter Notebook focused on analyzing and predicting financial inclusion in Africa. The analysis explores key factors that influence individuals' access to financial services and uses machine learning techniques to build a predictive model.

## Project Overview

Financial inclusion is a critical aspect of economic development, ensuring individuals and businesses have access to useful and affordable financial products and services. This project utilizes data analysis, feature engineering, and machine learning to predict the likelihood of individuals owning a bank account.

## Dataset

The dataset used in this project includes demographic, economic, and regional information of individuals across various African countries. It contains features such as:

- **Country**
- **Age**
- **Gender**
- **Marital Status**
- **Education Level**
- **Employment Status**
- **Cellphone Ownership**
- **Target Variable**: `bank_account` (Yes/No)

## Steps in the Analysis

1. **Data Loading and Inspection**:
   - The dataset is loaded using pandas.
   - Initial exploration includes checking for duplicates and missing values.

2. **Data Profiling**:
   - A profiling report is generated using `ydata_profiling` to understand the dataset's structure and distribution.

3. **Data Cleaning**:
   - Handling missing values.
   - Encoding categorical variables (e.g., one-hot encoding).
   - Transforming the target variable (`Yes` to 1, `No` to 0).

4. **Feature Engineering**:
   - Creating new features based on the dataset's characteristics.

5. **Model Building**:
   - Applying machine learning algorithms to predict financial inclusion.

6. **Model Evaluation**:
   - Assessing model performance using metrics such as accuracy, precision, recall, and F1-score.

## Requirements

The project uses the following Python libraries:

- pandas
- ydata_profiling
- scikit-learn
- matplotlib
- seaborn

Install the required libraries using the command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/financial-inclusion-africa.git
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook Streamlit_checkpoint_2_Financial_Inclusion_in_Africa.ipynb
   ```

3. Run the cells sequentially to reproduce the analysis and model.

## Results

The predictive model identifies the key determinants of financial inclusion and provides actionable insights for policymakers and stakeholders to improve access to financial services.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Dataset source: Financial Inclusion open source
- Special thanks to all contributors and reviewers.

