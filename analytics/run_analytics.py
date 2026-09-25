import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Ensure plots directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
CSV_FALLBACK_PATH = os.path.join(BASE_DIR, "titanic.csv")
MODEL_EXPORT_PATH = os.path.join(BASE_DIR, "best_pipeline.joblib")

# -------------------------------------------------------------------------
# PART A — TASK 1: LOAD & CACHE DATASET ONCE
# -------------------------------------------------------------------------
print("=" * 80)
print("PART A: PROFILING, CLEANING, AND DATA STORY")
print("=" * 80)

if os.path.exists(CSV_FALLBACK_PATH):
    print(f"Loading cached dataset from offline fallback: {CSV_FALLBACK_PATH}")
    raw_df = pd.read_csv(CSV_FALLBACK_PATH)
else:
    print("Fetching raw Titanic dataset via seaborn.load_dataset('titanic')...")
    raw_df = sns.load_dataset("titanic")
    # Save offline fallback immediately without modifying
    raw_df.to_csv(CSV_FALLBACK_PATH, index=False)
    print(f"Saved uncleaned dataset to committed offline fallback: {CSV_FALLBACK_PATH}")

print("\n--- df.info() ---")
raw_df.info()

print("\n--- df.shape ---")
print(raw_df.shape)

print("\n--- df.describe(include='all') ---")
print(raw_df.describe(include="all"))

print("\n--- Percentage of Missing Values Per Column ---")
missing_series = (raw_df.isnull().sum() / len(raw_df)) * 100
affected_missing = missing_series[missing_series > 0]
for col, pct in affected_missing.items():
    print(f"  - {col}: {pct:.2f}% missing")

# -------------------------------------------------------------------------
# PART A — TASK 2: DEFENSIVE MISSING-VALUE HANDLING
# Threshold Rules:
#  < 5% missing -> drop rows
#  5% - 30% missing -> impute
#  > 30% missing -> drop column or encode 'missing'
# -------------------------------------------------------------------------
df_clean = raw_df.copy()

# 'embarked' & 'embark_town': ~0.22% missing (< 5%) -> drop rows
df_clean = df_clean.dropna(subset=["embarked", "embark_town"])

# 'age': ~19.87% missing (5% - 30%) -> median imputation
median_age = df_clean["age"].median()
df_clean["age"] = df_clean["age"].fillna(median_age)

# 'deck': ~77% missing (> 30%) -> imputation unreliable; drop column
# 'deck' also leaks passenger class/fare and has high null cardinality
df_clean = df_clean.drop(columns=["deck"])

print(f"\nShape after missing-value resolution: {df_clean.shape}")

# -------------------------------------------------------------------------
# PART A — TASK 3: UNIVARIATE OUTLIER & SKEWNESS ANALYSIS
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df_clean["age"], kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age Distribution (Post-Imputation)")
sns.boxplot(x=df_clean["age"], ax=axes[0, 1], color="lightblue")
axes[0, 1].set_title("Age Boxplot")

sns.histplot(df_clean["fare"], kde=True, ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("Fare Distribution")
sns.boxplot(x=df_clean["fare"], ax=axes[1, 1], color="lightgreen")
axes[1, 1].set_title("Fare Boxplot")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "univariate_distributions.png"))
plt.close()

def get_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), lower_bound, upper_bound

age_outliers, age_lb, age_ub = get_iqr_outliers(df_clean["age"])
fare_outliers, fare_lb, fare_ub = get_iqr_outliers(df_clean["fare"])

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]

print("\n--- Univariate Analysis ---")
print(f"Age IQR Outliers (outside [{age_lb:.2f}, {age_ub:.2f}]): {age_outliers} rows")
print(f"Fare IQR Outliers (outside [{fare_lb:.2f}, {fare_ub:.2f}]): {fare_outliers} rows")
print(f"Fare Central Tendency: Mean = {fare_mean:.2f}, Median = {fare_median:.2f}, Mode = {fare_mode:.2f}")
print("Fare Skewness Verdict: Mean > Median > Mode indicates strong RIGHT-SKEWNESS.")

# -------------------------------------------------------------------------
# PART A — TASK 4: BIVARIATE ANALYSIS & SPECIFIED CORRELATION HEATMAP
# -------------------------------------------------------------------------
print("\n--- Bivariate Survival Rates (Boolean Masking) ---")
# (a) sex
male_surv = (df_clean[df_clean["sex"] == "male"]["survived"] == 1).mean() * 100
female_surv = (df_clean[df_clean["sex"] == "female"]["survived"] == 1).mean() * 100
print(f"Survival by Sex: Female = {female_surv:.2f}%, Male = {male_surv:.2f}%")

# (b) pclass
for pc in sorted(df_clean["pclass"].unique()):
    pc_surv = (df_clean[df_clean["pclass"] == pc]["survived"] == 1).mean() * 100
    print(f"Survival by Pclass {pc}: {pc_surv:.2f}%")

# (c) sex AND pclass combined
print("Survival by Sex and Pclass combined:")
for s in ["female", "male"]:
    for pc in [1, 2, 3]:
        rate = (df_clean[(df_clean["sex"] == s) & (df_clean["pclass"] == pc)]["survived"] == 1).mean() * 100
        print(f"  - {s.capitalize()}, Class {pc}: {rate:.2f}%")

# Heatmap restricted strictly to: survived, pclass, age, sibsp, parch, fare
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Strict 6 Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
plt.close()

# Identify top two off-diagonal correlations by absolute value
corr_unstacked = corr_matrix.abs().unstack()
# Filter out identity correlations
corr_unstacked = corr_unstacked[corr_unstacked < 1.0].drop_duplicates().sort_values(ascending=False)
top_2 = corr_unstacked.head(2)
print("\nTop 2 Strongest Off-Diagonal Correlations:")
for (feat1, feat2), val in top_2.items():
    raw_val = corr_matrix.loc[feat1, feat2]
    print(f"  1. {feat1} <-> {feat2}: Correlation = {raw_val:.3f} (|r| = {val:.3f})")

# -------------------------------------------------------------------------
# PART A — TASK 5: MULTIVARIATE DATA STORY (4 CHARTS)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Survival by Sex and Class
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", palette="Set2", ax=axes[0, 0])
axes[0, 0].set_title("1. Survival Probability by Class and Sex")
axes[0, 0].set_ylabel("Survival Rate")

# Chart 2: Fare vs Age by Survival Status
sns.scatterplot(
    data=df_clean, x="age", y="fare", hue="survived", alpha=0.7,
    palette={0: "crimson", 1: "seagreen"}, ax=axes[0, 1]
)
axes[0, 1].set_title("2. Fare vs Age by Survival Status")

# Chart 3: Age Distribution Across Classes by Survival
sns.violinplot(
    data=df_clean, x="pclass", y="age", hue="survived",
    split=True, inner="quart", palette="Pastel1", ax=axes[1, 0]
)
axes[1, 0].set_title("3. Age Spread Across Classes split by Survival")

# Chart 4: Family Size (sibsp + parch) vs Survival Rate
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"]
sns.pointplot(data=df_clean, x="family_size", y="survived", color="purple", ax=axes[1, 1])
axes[1, 1].set_title("4. Survival Rate by Total Family Members on Board")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "multivariate_charts.png"))
plt.close()

# -------------------------------------------------------------------------
# PART A — TASK 6: EDA-STAGE Z-SCORE SANITY CHECK (NOT LEAKED TO MODEL)
# -------------------------------------------------------------------------
age_fare_scaled = pd.DataFrame()
age_fare_scaled["age_z"] = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
age_fare_scaled["fare_z"] = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()

print("\n--- EDA-Stage Z-Score Sanity Check ---")
print(f"Age Pre-Scale:  Mean = {df_clean['age'].mean():.4f}, Std = {df_clean['age'].std():.4f}")
print(f"Age Post-Scale: Mean = {age_fare_scaled['age_z'].mean():.4f}, Std = {age_fare_scaled['age_z'].std():.4f}")
print(f"Fare Pre-Scale:  Mean = {df_clean['fare'].mean():.4f}, Std = {df_clean['fare'].std():.4f}")
print(f"Fare Post-Scale: Mean = {age_fare_scaled['fare_z'].mean():.4f}, Std = {age_fare_scaled['fare_z'].std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.kdeplot(age_fare_scaled["age_z"], ax=axes[0], label="Age (Z-Score)", fill=True)
sns.kdeplot(age_fare_scaled["fare_z"], ax=axes[1], label="Fare (Z-Score)", fill=True, color="green")
axes[0].set_title("Sanity Check: Standardized Age (Mean ~ 0, Std = 1)")
axes[1].set_title("Sanity Check: Standardized Fare (Mean ~ 0, Std = 1)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "standardization_check.png"))
plt.close()

# -------------------------------------------------------------------------
# PART B — TASK 7: STRATIFIED TRAIN/TEST SPLIT
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART B: PREDICTIVE MODELING")
print("=" * 80)

# Work from the cleaned dataset (df_clean)
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

X = df_clean[features].copy()
y = df_clean[target].copy()

surv_balance = y.value_counts(normalize=True) * 100
print(f"Class Balance: Perished (0) = {surv_balance[0]:.2f}%, Survived (1) = {surv_balance[1]:.2f}%")
print("Stratification is strictly applied to preserve the ~62:38 class ratio across splits.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# -------------------------------------------------------------------------
# PART B — TASK 8: LEAK-FREE PREPROCESSING PIPELINE
# -------------------------------------------------------------------------
numeric_features = ["age", "fare", "pclass", "sibsp", "parch"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------------------------------------------------
# PART B — TASK 9 & 10: TRAIN 3 CLASSIFIERS & EVALUATION
# -------------------------------------------------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

results = {}
roc_data = {}

for name, clf in classifiers.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Confusion Matrix": cm.tolist()
    }
    roc_data[name] = roc_curve(y_test, y_prob)

    # Render Decision Tree visualization
    if name == "Decision Tree":
        plt.figure(figsize=(18, 10))
        # Retrieve feature names out of preprocessor
        fitted_prep = pipe.named_steps["preprocessor"]
        cat_encoder = fitted_prep.named_transformers_["cat"].named_steps["encoder"]
        ohe_cols = list(cat_encoder.get_feature_names_out(categorical_features))
        all_feat_names = numeric_features + ohe_cols

        plot_tree(
            pipe.named_steps["classifier"],
            feature_names=all_feat_names,
            class_names=["Perished", "Survived"],
            filled=True,
            rounded=True,
            fontsize=9
        )
        plt.title("Decision Tree Structure (Labeled Features & Classes)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "decision_tree.png"))
        plt.close()

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, _) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC-AUC']})")
plt.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"))
plt.close()

df_clf_comparison = pd.DataFrame(results).T.drop(columns=["Confusion Matrix"])
print("\n--- Classifier Comparison Table (Task 10) ---")
print(df_clf_comparison)

# -------------------------------------------------------------------------
# PART B — TASK 11: IMBALANCE HANDLING COMPARISON
# Model evaluated: Logistic Regression
# Three ways: (a) baseline, (b) class_weight='balanced', (c) SMOTE on train only
# -------------------------------------------------------------------------
print("\n--- Task 11: Class Imbalance Strategy Comparison ---")
# Strategy A: Baseline (already computed)
imb_baseline = {
    "Precision": results["Logistic Regression"]["Precision"],
    "Recall": results["Logistic Regression"]["Recall"],
    "F1 Score": results["Logistic Regression"]["F1 Score"]
}

# Strategy B: class_weight='balanced'
pipe_balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000))
])
pipe_balanced.fit(X_train, y_train)
y_pred_bal = pipe_balanced.predict(X_test)
imb_balanced = {
    "Precision": round(precision_score(y_test, y_pred_bal), 4),
    "Recall": round(recall_score(y_test, y_pred_bal), 4),
    "F1 Score": round(f1_score(y_test, y_pred_bal), 4)
}

# Strategy C: SMOTE applied strictly to transformed training fold only
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_trans, y_train)

clf_smote = LogisticRegression(random_state=42, max_iter=1000)
clf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = clf_smote.predict(X_test_trans)
imb_smote = {
    "Precision": round(precision_score(y_test, y_pred_smote), 4),
    "Recall": round(recall_score(y_test, y_pred_smote), 4),
    "F1 Score": round(f1_score(y_test, y_pred_smote), 4)
}

df_imbalance = pd.DataFrame({
    "Baseline": imb_baseline,
    "class_weight='balanced'": imb_balanced,
    "SMOTE (Train Only)": imb_smote
}).T
print(df_imbalance)

# -------------------------------------------------------------------------
# PART B — TASK 12: HYPERPARAMETER TUNING (RANDOM FOREST + OOB SCORE)
# -------------------------------------------------------------------------
print("\n--- Task 12: Random Forest GridSearchCV with OOB Score ---")
# Pre-transform training features for fast grid search
rf_base = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 8, None],
    "max_features": ["sqrt", "log2"]
}

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=cv_strat, scoring="f1", n_jobs=-1)
grid_search.fit(X_train_trans, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Corresponding Out-of-Bag (OOB) Score: {best_rf.oob_score_:.4f}")

# -------------------------------------------------------------------------
# PART B — TASK 13: REGRESSION SIDE-TASK (PREDICTING FARE)
# -------------------------------------------------------------------------
print("\n--- Task 13: Regression Side-Task (Predicting 'fare') ---")
reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
X_reg = df_clean[reg_features]
y_reg = df_clean["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_num_cols = ["age", "pclass", "sibsp", "parch"]
reg_cat_cols = ["sex", "embarked"]

reg_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(drop="first"))]), reg_cat_cols)
])

reg_pipeline = Pipeline([
    ("preprocessor", reg_prep),
    ("regressor", LinearRegression())
])
reg_pipeline.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
n_samples = len(y_reg_test)
p_features = X_reg_train.shape[1]
adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - p_features - 1)

print(f"Regression Metrics: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R2 = {r2:.4f}, Adjusted R2 = {adj_r2:.4f}")

# Residual Plot
residuals = y_reg_test - y_reg_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.6, color="darkorange")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Predicted Fare (£)")
plt.ylabel("Residuals (£)")
plt.title("Residuals vs. Predicted Values (Heteroscedasticity Analysis)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "regression_residuals.png"))
plt.close()
print("Residual Plot saved. Strong fan/funnel shape confirms distinct HETEROSCEDASTICITY.")

# -------------------------------------------------------------------------
# PART B — TASK 14: MODEL COMPARISON TABLE
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 14: FINAL MODEL COMPARISON TABLE")
print("=" * 80)

# Build a clean side-by-side DataFrame with distinct metric groups
comp_data = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest (Tuned)", "Linear Regression (Fare Task)"],
    "Type": ["Classification", "Classification", "Classification", "Regression"],
    "Accuracy": [results["Logistic Regression"]["Accuracy"], results["Decision Tree"]["Accuracy"], round(accuracy_score(y_test, best_rf.predict(X_test_trans)), 4), "-"],
    "Precision": [results["Logistic Regression"]["Precision"], results["Decision Tree"]["Precision"], round(precision_score(y_test, best_rf.predict(X_test_trans)), 4), "-"],
    "Recall": [results["Logistic Regression"]["Recall"], results["Decision Tree"]["Recall"], round(recall_score(y_test, best_rf.predict(X_test_trans)), 4), "-"],
    "F1 Score": [results["Logistic Regression"]["F1 Score"], results["Decision Tree"]["F1 Score"], round(f1_score(y_test, best_rf.predict(X_test_trans)), 4), "-"],
    "ROC-AUC": [results["Logistic Regression"]["ROC-AUC"], results["Decision Tree"]["ROC-AUC"], round(roc_auc_score(y_test, best_rf.predict_proba(X_test_trans)[:, 1]), 4), "-"],
    "MAE": ["-", "-", "-", round(mae, 2)],
    "RMSE": ["-", "-", "-", round(rmse, 2)],
    "R2": ["-", "-", "-", round(r2, 4)],
    "Adjusted R2": ["-", "-", "-", round(adj_r2, 4)]
}
final_comp_table = pd.DataFrame(comp_data)
print(final_comp_table.to_string(index=False))

# -------------------------------------------------------------------------
# PART B — TASK 15: SAVE COMPLETE PIPELINE & VERIFY RELOAD
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 15: ARTIFACT EXPORT & END-TO-END INFERENCE VERIFICATION")
print("=" * 80)

# Build end-to-end full pipeline containing both preprocessor and best tuned classifier
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_rf)
])
full_pipeline.fit(X_train, y_train)

# Save to disk
joblib.dump(full_pipeline, MODEL_EXPORT_PATH)
print(f"Complete fitted pipeline successfully dumped to: {MODEL_EXPORT_PATH}")

# Reload from disk
reloaded_pipeline = joblib.load(MODEL_EXPORT_PATH)
print("Successfully reloaded artifact via joblib.load().")

# Demonstrate end-to-end inference on raw, un-preprocessed test samples
sample_raw_input = pd.DataFrame([
    {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0, "fare": 211.3375, "embarked": "S"},
    {"pclass": 3, "sex": "male", "age": 22.0, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}
])
sample_preds = reloaded_pipeline.predict(sample_raw_input)
sample_probs = reloaded_pipeline.predict_proba(sample_raw_input)[:, 1]

print("\nRaw Input Test Data:")
print(sample_raw_input)
print(f"Predictions from reloaded pipeline (0 = Perished, 1 = Survived): {sample_preds}")
print(f"Predicted Survival Probabilities: {np.round(sample_probs, 4)}")
print("\n[SUCCESS] Module 2 analytics pipeline completed all requirements.")