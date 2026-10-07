# Bank Customer Churn Prediction

A machine learning project that predicts which bank customers are likely to leave, and explains why, so a retention team can contact the right people first. It includes a full analysis, three compared models, a tuned decision threshold, SHAP explanations, and a Streamlit web app.

<!--
LIVE DEMO: after you deploy on Streamlit Community Cloud, add this line here:
**[Try the live app](https://your-app-link.streamlit.app)**

SCREENSHOTS: upload 2-3 screenshots of the app to the repo (for example in a folder called images),
then add lines like this:
![Single customer prediction](images/single_customer.png)
-->

## Business problem

Losing a customer costs a bank future revenue, and winning a new customer is far more expensive than keeping an existing one. If the bank can identify customers who are about to leave, it can act early with an offer, a call, or a fix to whatever is causing the problem.

Only about 1 in 5 customers leaves, so overall accuracy is a misleading measure (a model that always predicts "stays" would be about 80% accurate and useless). This project judges models on **recall**, **precision** and **ROC-AUC** instead.

## Results

Final model: **XGBoost**, with a decision threshold of **0.55** chosen on training data only (using cross-validated predictions) and scored once on a held-out test set of 2,000 customers.

| Metric (test set, threshold 0.55) | Result |
|---|---|
| Recall (share of churners caught) | **0.71** |
| Precision (share of flags that were correct) | **0.55** |
| F1 score | 0.62 |
| Customers flagged | 26.4% |
| Churners caught / missed | 288 / 119 |
| Cross-validated ROC-AUC (5-fold) | **0.864** |

**In business terms:** contacting the 26% of customers the model flags reaches about 71% of the customers who actually leave, roughly **2.7 times better than contacting a random group of the same size**.

### Model comparison

Tested on the held-out set at the default 0.5 cutoff, with 5-fold cross-validated ROC-AUC alongside:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (test) | ROC-AUC (5-fold CV) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.389 | 0.705 | 0.501 | 0.777 | 0.766 |
| Random Forest | 0.859 | 0.768 | 0.440 | 0.559 | 0.857 | 0.854 |
| **XGBoost** | 0.804 | 0.512 | 0.747 | 0.607 | 0.867 | 0.864 |

Random Forest has the highest accuracy but catches only 44% of churners, which shows why accuracy alone is the wrong measure here. XGBoost scored higher than Random Forest in all five cross-validation folds.

## Key insights

- **Age:** churn peaks at ages 51 to 60 (56%) and is much lower for customers under 40 (8% to 12%).
- **Number of products:** customers with 2 products are the most loyal (8% churn). Customers with 3 or 4 products churn at 83% and 100%, though these groups are small (326 customers combined).
- **Activity:** inactive members churn at 27% versus 14% for active members.
- **Country:** customers in Germany churn at about 32%, roughly double France and Spain.
- **Linear correlation misses patterns:** the correlation between number of products and churn is only -0.05, yet it is one of the strongest signals. The relationship is a curve, which is why tree-based models outperform logistic regression.

## Approach

1. **Exploratory analysis** of churn by country, age, products, activity, gender and balance.
2. **Feature engineering:** created a `ZeroBalance` flag; dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`).
3. **Preprocessing in a scikit-learn Pipeline:** scaling for numeric features and one-hot encoding for categories, so there is no data leakage and the app transforms new data exactly as in training.
4. **Split:** 80/20 stratified (8,000 train, 2,000 test). The test set stays untouched until the final evaluation.
5. **Class imbalance:** handled with class weights and `scale_pos_weight`.
6. **Model comparison:** Logistic Regression, Random Forest and XGBoost, evaluated with 5-fold stratified cross-validation.
7. **Threshold tuning:** compared three strategies (best F1, a recall target of 80%, and a cost-based rule) and chose **0.55**, which flags about 1 in 4 customers, to fit a realistic retention-team capacity.
8. **Explainability:** SHAP values show the global drivers of churn and the reasons behind each individual prediction.

For the full write-up with charts, see **Bank_Churn_Prediction_Report.pdf**.

## The app

A Streamlit app lets non-technical users use the model:

- **Single customer:** enter customer details and get a churn probability, a risk label, the top factors pushing towards leaving or staying, and a contribution chart.
- **Batch upload:** upload a CSV and get a probability and risk flag for every customer, sorted from highest to lowest risk, with a download button.
- **Threshold slider:** change the cutoff live and see how many customers get flagged.

A ready-made file, `sample_customers.csv`, is included for trying the batch upload.

## Project structure

```
.
├── app.py                          # Streamlit app
├── train_model.py                  # Trains the model and saves churn_model.joblib
├── churn_model.joblib              # Saved pipeline (preprocessing + XGBoost)
├── requirements.txt                # Python dependencies
├── sample_customers.csv            # 50 example customers for the batch upload
├── Bank_Churn_Prediction_Report.pdf  # Full project report
└── (analysis notebook, .ipynb)     # Step-by-step EDA and modelling
```

## How to run it

1. **Clone the repository and install the dependencies**

   ```
   git clone https://github.com/Suman-git-code/bank-churn-prediction.git
   cd bank-churn-prediction
   pip install -r requirements.txt
   ```

2. **Start the app** (the trained model is already included)

   ```
   streamlit run app.py
   ```

3. **Optional: retrain the model.** Download the *Churn Modelling* bank customer dataset from Kaggle (search for "Churn Modelling"), save it as `Churn_Modelling.csv` in the project folder, then run:

   ```
   python train_model.py
   ```

The saved model only loads reliably with the same scikit-learn and XGBoost versions it was trained with. If you see a version warning, retrain with `train_model.py`.

## Limitations

- The dataset is clean and public, and may be partly synthetic. Real bank data would be messier, and the approach would need adapting.
- It is a single snapshot with no time dimension, so it cannot show how churn changes over time.
- Class weighting inflates the predicted probabilities, so scores should be used to rank customers and apply a threshold, not read as literal chances of leaving.
- The cost-based threshold used an assumed 10:1 cost ratio. In a real project, the client's own costs would be used.
- SHAP shows what the model relies on, not what causes churn.
- The model uses gender and country as inputs. A real deployment would need a fairness and regulatory review.

## Next steps

- Run a controlled experiment to test whether retention offers actually reduce churn.
- Tune hyperparameters and calibrate the probabilities.
- Monitor for data drift and retrain on a regular schedule.
- Investigate why customers with 3 to 4 products and customers in Germany leave so often.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Streamlit, joblib.

<!--
AUTHOR: add a short section here with your name and links, for example:

## Author
Your Name | Machine Learning Engineer
LinkedIn: your-link | Upwork: your-link
-->
