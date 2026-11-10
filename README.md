# Smart Support Ticket Triage and Copilot

This project is a technical prototype for an AI-powered customer-support platform. It is designed to help support teams classify tickets, route them to the correct team, suggest grounded answers, and detect emerging incidents before they become large operational problems.

The project follows the requirements described in the teacher's documents:

- Classification, routing, and prioritization
- Retrieval-Augmented Generation (RAG) for suggested responses
- Anomaly and incident detection
- Human-in-the-loop review for uncertain predictions
- Monitoring, evaluation, and reproducible data versioning

## Current Scope

The first implementation milestone covers the data audit, a safe classification baseline, and an initial anomaly analysis. It is intentionally conservative: it measures the data before training, prevents obvious leakage, and keeps the two datasets separate because they describe different things.

Implemented now:

1. Bitext data audit for missing values, duplicates, label leakage, and class distribution.
2. TF-IDF plus Logistic Regression classification baseline.
3. Duplicate-aware train/test splitting.
4. Label-token removal from the text used by the classifier.
5. Ticket volume spike detection.
6. Operational outlier detection using Isolation Forest.
7. JSON and CSV reports for reproducibility.
8. DVC pointer files for the raw datasets.

Not implemented yet:

- DistilBERT or CamemBERT multi-task fine-tuning
- Urgency and sentiment models
- RAG retrieval and response generation
- Embedding-based semantic clustering
- Human review web interface
- FastAPI service
- Grafana dashboard, EvidentlyAI monitoring, and Airflow orchestration

These are later project phases. The current baseline gives us measured evidence and a reliable foundation before adding more complex models.

## Datasets

### 1. Bitext customer-support dataset

File:

`data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

Observed schema:

- `flags`: source or generation flag
- `instruction`: customer request text used as the model input
- `category`: broad support category used as the baseline target
- `intent`: more specific intent label
- `response`: proposed support answer, reserved for later RAG work

Observed size: 26,872 rows.

This dataset is used for text classification and will later be useful as a source of historical answers for RAG. The current classifier uses only `instruction` as input. It does not use `response`, because using the answer while predicting the ticket category would create an unrealistic shortcut.

Important observation: the dataset contains 11 categories, while the teacher's project document mentions 8 classes. The implementation reports the real dataset structure instead of silently changing or hiding it. This difference should be explained during the technical evaluation.

### 2. Customer-support ticket records

File:

`data/raw/customer_support_ticket_records.csv`

Observed schema:

- `ticket_id`
- `created_at`
- `customer_id`
- `phone_number`
- `operator`
- `issue_type`
- `priority`
- `status`
- `channel`
- `assigned_to`
- `resolution_time_hours`
- `customer_satisfaction`

Observed size: 300,000 rows.

This dataset does not contain a free-text ticket field, so it is not merged with the Bitext dataset and is not used for text classification or RAG. It is used for operational analysis: daily ticket volume, issue-type spikes, resolution-time behavior, and satisfaction outliers.

Identifiers such as `ticket_id`, `customer_id`, `phone_number`, and `assigned_to` are excluded from anomaly features. They identify records or people; they do not represent generalizable ticket behavior.

## File-by-File Explanation

### `src/data/show_dataset_head.py`

Purpose: provide a small, shareable preview of the dataset.

Logic:

1. Resolve the repository root from the script location.
2. Load the Bitext CSV with pandas.
3. Print the first five rows.
4. Optionally save those five rows as a CSV file.

Run it with:

```powershell
.\venv\Scripts\python.exe src\data\show_dataset_head.py --output reports\dataset_head_5.csv
```

### `src/data/audit_leakage.py`

Purpose: inspect the Bitext dataset before training.

Logic:

1. Load the dataset and verify the required columns.
2. Normalize instruction text by trimming whitespace and converting it to lowercase.
3. Count exact duplicate instruction rows and duplicate instruction groups.
4. Check whether duplicate instruction groups have conflicting category labels.
5. Count instructions containing their category token or intent token.
6. Report missing values and the category distribution.
7. Confirm that `response` is not used as classifier input.

The audit found:

- 3,785 rows in exact duplicate instruction groups
- 1,187 duplicate instruction groups
- 0 conflicting duplicate groups
- 12,868 instructions containing a category token
- 455 instructions containing an intent token

These findings explain why a normal random row split would be risky: identical or highly related examples could appear in both training and testing data.

Run it with:

```powershell
.\venv\Scripts\python.exe src\data\audit_leakage.py --output reports\bitext_leakage_audit.json
```

### `src/models/train_baseline.py`

Purpose: create a reproducible and leakage-aware classification baseline.

Model:

- TF-IDF vectorization with at most 5,000 features
- Logistic Regression with a fixed random state
- Macro-F1 as the main metric
- Classification report and confusion matrix
- MLflow experiment logging

Logic:

1. Load the Bitext dataset.
2. Optionally sample rows with a fixed random state.
3. Normalize instruction text to create duplicate groups.
4. Use `StratifiedGroupKFold` so the same normalized instruction cannot appear in both train and test sets.
5. Remove the category token and the intent phrase from each instruction before fitting the vectorizer.
6. Fit TF-IDF only on training text.
7. Transform test text with the already-fitted vectorizer.
8. Train Logistic Regression on the training labels.
9. Evaluate on the untouched test partition.
10. Save metrics, a confusion matrix, and the model run in MLflow.

The response column is never used as a model feature. The split also checks for duplicate group overlap and stops if overlap is detected.

Run it with:

```powershell
.\venv\Scripts\python.exe src\models\train_baseline.py
```

Optional arguments:

```powershell
.\venv\Scripts\python.exe src\models\train_baseline.py --sample-size 5000 --random-state 42
```

Current result on the full dataset:

- Training rows: 21,497
- Test rows: 5,375
- Duplicate groups crossing the split: 0
- Macro-F1: approximately 0.9870

The score is a baseline result, not proof that the final system is production-ready. Near-duplicate paraphrases may still make the task easier than real-world support data, so later evaluation should include a more realistic temporal or source-based split if that information becomes available.

### `src/anomaly/analyze_tickets.py`

Purpose: provide the first operational anomaly-detection baseline using the 300,000-row ticket-record dataset.

It contains two independent signals.

#### Volume spike detection

1. Parse `created_at` as a timestamp.
2. Count tickets per day and `issue_type`.
3. For each issue type, calculate a trailing seven-day baseline using only previous days.
4. Calculate a spike score from the current count and the previous baseline.
5. Report cases whose score is at least 3 standard deviations above the baseline.

Using previous days only prevents the current observation from influencing its own baseline.

#### Operational outlier detection

1. Select only `resolution_time_hours` and `customer_satisfaction`.
2. Impute missing numeric values with the median.
3. Standardize the two features.
4. Apply Isolation Forest with a fixed random state and 1% contamination.
5. Save the most unusual operational records with non-identifying context.

This is an initial baseline. It does not yet detect semantic incidents from ticket text because the operational dataset has no ticket-text column. The later version can use sentence-transformer embeddings from a text-bearing dataset and HDBSCAN or k-means.

Run it with:

```powershell
.\venv\Scripts\python.exe src\anomaly\analyze_tickets.py
```

Current result:

- Rows processed: 300,000
- Valid timestamps: 300,000
- Volume spikes: 2
- Operational outliers: 2,928

## Reports

The scripts generate the following files under `reports/`:

- `dataset_head_5.csv`: first five Bitext rows for sharing or inspection
- `bitext_leakage_audit.json`: data quality and leakage audit
- `classification_baseline_metrics.json`: macro-F1 and per-class metrics
- `classification_confusion_matrix.csv`: classifier confusion matrix
- `ticket_anomaly_summary.json`: anomaly-analysis summary
- `ticket_volume_spikes.csv`: detected issue-type volume spikes
- `ticket_operational_outliers.csv`: detected operational outliers

## Reproducibility and Data Safety

The raw CSV files are ignored by Git and represented by DVC pointer files:

- `data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv.dvc`
- `data/raw/customer_support_ticket_records.csv.dvc`

The current `.dvc/config` has no remote storage configured. Therefore, a future collaborator needs access to the original data files or a configured DVC remote before `dvc pull` can restore them.

The local virtual environment is used for execution:

```powershell
.\venv\Scripts\python.exe
```

Main Python packages used by the current implementation:

- pandas
- scikit-learn
- MLflow

The project does not commit the virtual environment, raw data, MLflow run directory, or local MLflow database.

## Leakage, Overfitting, and Underfitting Policy

Every future model change should follow these rules:

1. Split data before fitting preprocessing components.
2. Fit vectorizers, scalers, imputers, and encoders only on training data.
3. Keep test data untouched until final evaluation.
4. Group exact duplicates so they cannot cross the split.
5. Do not use response text, identifiers, or post-resolution information as prediction inputs unless the real workflow has that information at prediction time.
6. Compare training and validation metrics to identify overfitting.
7. Use learning curves, model capacity, and feature checks to diagnose underfitting.
8. Report macro-F1, per-class results, and confusion matrices rather than accuracy alone.
9. Keep a fixed random seed for reproducibility.
10. Document every synthetic label, preprocessing rule, and data limitation.

## Planned Next Steps

1. Add a stronger data split strategy, such as a temporal split when timestamps are available for the text tickets.
2. Build the category baseline comparison and calibration/confidence threshold.
3. Define and document how urgency and sentiment labels will be obtained, because neither is present in the Bitext CSV.
4. Fine-tune a lightweight transformer for multi-task classification if reliable labels are available.
5. Build the RAG index from historical instructions and responses, with retrieved sources shown to the agent.
6. Add grounding, unanchored-response, and adoption metrics.
7. Add embedding-based clustering for semantic incident detection.
8. Connect the results to a human-review interface and monitoring dashboard.
9. Add Airflow orchestration and periodic EvidentlyAI drift reports.
10. Expose the workflow through FastAPI and test the complete pipeline.
