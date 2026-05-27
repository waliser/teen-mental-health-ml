import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Social Media Impact on Teen Mental Health

    Binary classification task predicting depression (`depression_label`: 0 or 1) from social media usage patterns and health indicators in 1,200 teen records.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from collections import Counter
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    from imblearn.combine import SMOTETomek
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        Counter,
        DecisionTreeClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SMOTETomek,
        StandardScaler,
        XGBClassifier,
        classification_report,
        confusion_matrix,
        cross_val_score,
        mo,
        optuna,
        pd,
        plot_tree,
        plt,
        px,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load & Inspect

    Load the dataset and check shape, types, missing values, and class distribution.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
    print(df.shape)
    print(df.dtypes)
    df.head()
    return (df,)


@app.cell
def _(df):
    print(df.isna().sum())
    print(df['depression_label'].value_counts())
    print(df['gender'].unique())
    print(df['platform_usage'].unique())
    print(df['social_interaction_level'].unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Cleaning & Encoding

    - `social_interaction_level`: ordinal map (low=0, medium=1, high=2) — order matters
    - `gender`: binary label encode (male=0, female=1)
    - `platform_usage`: one-hot encode — no natural order between Instagram/TikTok/Both

    Note: `drop_first=True` drops `Both` as the reference category to avoid multicollinearity.
    """)
    return


@app.cell
def _(df, pd):
    sdf = df.copy()
    sdf['social_interaction_level'] = sdf['social_interaction_level'].map({'low': 0, 'medium': 1, 'high': 2})
    sdf['gender'] = sdf['gender'].map({'male': 0, 'female': 1})
    sdf = pd.get_dummies(sdf, columns=['platform_usage'], drop_first=True)
    print(sdf.columns.tolist())
    sdf.head()
    return (sdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Exploratory Data Analysis

    Check distributions, summary stats, and correlations with the target variable.
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(px, sdf):
    corr = sdf.corr().round(2)
    fig = px.imshow(corr, text_auto=True, title='Correlation Matrix')
    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Top features correlated with `depression_label`:
    - `sleep_hours`: -0.19 (more sleep = less depression)
    - `daily_social_media_hours`: +0.18
    - `stress_level`: +0.17
    - `anxiety_level`: +0.17

    `academic_performance` shows near-zero correlation — not a useful predictor in this dataset.
    """)
    return


@app.cell
def _(px, sdf):
    topfeatures = ['daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level']
    figs = px.box(
        sdf[topfeatures + ['depression_label']].melt(id_vars='depression_label'),
        x='variable', y='value', color='depression_label',
        title='Top Features by Depression Label'
    )
    figs.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train/Test Split

    80/20 split with `stratify=y` to preserve the 97/3 class ratio in both sets.

    **Class imbalance warning:** only 31 depressed teens out of 1,200 (2.6%). Accuracy alone is misleading — a model predicting 0 always gets 97.4% accuracy while being completely useless. Focus on F1 score for class 1.
    """)
    return


@app.cell
def _(sdf, train_test_split):
    x = sdf.drop(columns='depression_label')
    y = sdf['depression_label']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=67, stratify=y)

    print(xtrain.shape, xtest.shape)
    print(ytrain.value_counts())
    print(ytest.value_counts())
    return x, xtest, xtrain, y, ytest, ytrain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Baseline Models

    Train Logistic Regression, Decision Tree, and Random Forest with default params and `class_weight='balanced'` to handle imbalance. Scale only for Logistic Regression (distance-sensitive). Evaluate each before tuning.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    classification_report,
    xtest,
    xtrain,
    ytest,
    ytrain,
):
    scaler = StandardScaler()
    xtrains = scaler.fit_transform(xtrain)
    xtests = scaler.transform(xtest)

    logreg = LogisticRegression(class_weight='balanced', random_state=67)
    logreg.fit(xtrains, ytrain)
    ypredlr = logreg.predict(xtests)
    print('Logistic Regression')
    print(classification_report(ytest, ypredlr))
    return (scaler,)


@app.cell
def _(
    DecisionTreeClassifier,
    classification_report,
    xtest,
    xtrain,
    ytest,
    ytrain,
):
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=67)
    dt.fit(xtrain, ytrain)
    ypreddt = dt.predict(xtest)
    print('Decision Tree')
    print(classification_report(ytest, ypreddt))
    print(f'Depth: {dt.get_depth()}, Leaves: {dt.get_n_leaves()}')
    return (dt,)


@app.cell
def _(dt, plot_tree, plt, x):
    figg, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt, feature_names=x.columns.tolist(), class_names=['0', '1'], filled=True, ax=ax)
    plt.title('Decision Tree Structure')
    plt.show()
    return


@app.cell
def _(
    RandomForestClassifier,
    classification_report,
    xtest,
    xtrain,
    ytest,
    ytrain,
):
    rf = RandomForestClassifier(class_weight='balanced', random_state=67)
    rf.fit(xtrain, ytrain)
    ypredrf = rf.predict(xtest)
    print('Random Forest')
    print(classification_report(ytest, ypredrf))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Handling Class Imbalance with SMOTETomek

    SMOTETomek combines two techniques:
    - **SMOTE**: generates synthetic minority class samples by interpolating between existing ones
    - **Tomek Links**: removes majority class samples that are too close to the minority boundary

    Applied **only on training data** — never on test. Applying SMOTE to test data would cause data leakage.
    """)
    return


@app.cell
def _(SMOTETomek, xtrain, ytrain):
    st = SMOTETomek(random_state=67)
    xtrainstk, ytrainstk = st.fit_resample(xtrain, ytrain)
    print(ytrainstk.value_counts())
    return xtrainstk, ytrainstk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Retrain All Models on Balanced Data

    Retrain with SMOTETomek data. Add XGBoost — a sequential boosting ensemble that iteratively corrects previous tree errors. Remove `class_weight='balanced'` since SMOTETomek already handles balance.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier,
    classification_report,
    scaler,
    xtest,
    xtrainstk,
    ytest,
    ytrainstk,
):

    xtrainstks = scaler.transform(xtrainstk)
    xteststks = scaler.transform(xtest)
    logregsm = LogisticRegression(random_state=67)
    logregsm.fit(xtrainstks, ytrainstk)
    ypredlrsm = logregsm.predict(xteststks)

    dtsm = DecisionTreeClassifier(random_state=67)
    dtsm.fit(xtrainstk, ytrainstk)
    ypreddtsm = dtsm.predict(xtest)

    rfsm = RandomForestClassifier(random_state=67)
    rfsm.fit(xtrainstk, ytrainstk)
    ypredrfsm = rfsm.predict(xtest)

    xgb = XGBClassifier(random_state=67, eval_metric='logloss')
    xgb.fit(xtrainstk, ytrainstk)
    ypredxgb = xgb.predict(xtest)

    for name, pred in [('LogReg', ypredlrsm), ('DT', ypreddtsm), ('RF', ypredrfsm), ('XGB', ypredxgb)]:
        print(f'\n{name}')
        print(classification_report(ytest, pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why RF Scores Lower Than DT and XGB

    Random Forest averaged predictions across many decision trees, which makes it
    more conservative — it needs stronger, consistent signals before predicting the
    minority class. With only 6 depressed teens in the test set, that conservatism
    backfires: RF misses one extra case that DT and XGB catch, dropping recall from
    0.83 to 0.67 and F1 from 0.91 to 0.80.

    This is not a general weakness of RF. Across a larger or more balanced test set,
    RF typically matches or outperforms a single DT by reducing variance. The result
    here reflects the instability of evaluating on 6 samples — one missed prediction
    is a 17% recall drop.

    ## Why F1 = 0.91 and Not 1.0

    The one missed case is **index 691** — a depressed teen whose feature profile
    partially contradicts the pattern the model learned:

    | Feature | Her value | Depressed avg | Match |
    |---|---|---|---|
    | `stress_level` | 9 | 8.48 | ✓ |
    | `anxiety_level` | 7 | 8.61 | ✗ slightly low |
    | `addiction_level` | 5 | 5.32 | ✗ near non-depressed avg |
    | `sleep_hours` | 5.7 | low | ✓ |
    | `daily_social_media_hours` | 8.0 | high | ✓ |

    She has high stress and poor sleep, but her anxiety and addiction scores sit
    closer to the non-depressed average. The model's learned splits rely heavily on
    anxiety and addiction as separating features — she falls on the wrong side of
    those boundaries despite being genuinely depressed.

    This was confirmed across 200 random seeds in Section 11: index 691 is the most
    persistently misclassified sample regardless of which split is used. No amount
    of tuning resolves a statistically ambiguous case — the features simply don't
    tell a clean story for her.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Hyperparameter Tuning with Optuna

    Decision Tree and XGBoost tied at F1=0.91 pre-tuning. Both are tuned separately using Optuna — smarter than GridSearchCV because it samples intelligently from the search space rather than exhaustively trying every combination.

    Both studies optimize F1 score on class 1 using 5-fold cross-validation on SMOTETomek training data.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, cross_val_score, optuna, xtrainstk, ytrainstk):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 70)
        }
        model = DecisionTreeClassifier(**params, random_state=67)
        score = cross_val_score(model, xtrainstk, ytrainstk, cv=5, scoring='f1').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, catch=(ValueError,))

    print('Best params:', study.best_params)
    print('Best CV F1:', study.best_value)
    return (study,)


@app.cell
def _(XGBClassifier, cross_val_score, xtrainstk, ytrainstk):
    def objectivexgb(trial):
        paramxgb = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }

        modelxgb = XGBClassifier(**paramxgb, random_state=67, eval_metric='logloss')
        scorexgb = cross_val_score(modelxgb, xtrainstk, ytrainstk, cv=5, scoring='f1').mean()
        return scorexgb

    return (objectivexgb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    XGBoost has more hyperparameters than DT, so the search space is wider:
    - `n_estimators`: number of boosting rounds (50–500)
    - `max_depth`: tree depth per round (1–20)
    - `learning_rate`: shrinkage applied after each round — log scale since small values matter more
    - `subsample`: fraction of training samples used per round — reduces overfitting
    - `colsample_bytree`: fraction of features sampled per tree — similar to RF's feature subsampling

    Note: CV F1=1.0 here is expected. Optuna never sees index 691 in a validation fold during most trials — the hard case only surfaces on the held-out test set in Section 9.
    """)
    return


@app.cell
def _(objectivexgb, optuna):
    studyxgb = optuna.create_study(direction='maximize')
    studyxgb.optimize(objectivexgb, n_trials=100, catch=(ValueError,))

    print('Best params:', studyxgb.best_params)
    print('Best CV F1:', studyxgb.best_value)
    return (studyxgb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Final Model Evaluation

    Train final XGBoost and Decision Tree with their respective Optuna best params on SMOTETomek data, then evaluate both on the held-out test set. Both models tied pre-tuning at F1=0.91 — this confirms whether tuning changed anything.
    """)
    return


@app.cell
def _(
    XGBClassifier,
    classification_report,
    studyxgb,
    xtest,
    xtrainstk,
    ytest,
    ytrainstk,
):
    xgbfinal = XGBClassifier(**studyxgb.best_params, random_state=67, eval_metric='logloss')
    xgbfinal.fit(xtrainstk, ytrainstk)
    ypredxgbfinal = xgbfinal.predict(xtest)
    print(classification_report(ytest, ypredxgbfinal))
    return


@app.cell
def _(
    DecisionTreeClassifier,
    classification_report,
    study,
    xtest,
    xtrainstk,
    ytest,
    ytrainstk,
):
    dtfinal = DecisionTreeClassifier(**study.best_params, random_state=67)
    dtfinal.fit(xtrainstk, ytrainstk)
    ypredfinal = dtfinal.predict(xtest)
    print(classification_report(ytest, ypredfinal))
    return dtfinal, ypredfinal


@app.cell
def _(confusion_matrix, px, ypredfinal, ytest):
    cm = confusion_matrix(ytest, ypredfinal)
    figcm = px.imshow(cm, text_auto=True,
                      labels=dict(x='Predicted', y='Actual'),
                      x=['Not Depressed', 'Depressed'],
                      y=['Not Depressed', 'Depressed'],
                      title='Confusion Matrix - Final Decision Tree')
    figcm.show()
    return


@app.cell
def _(dtfinal, pd, px, x):
    importances = pd.Series(dtfinal.feature_importances_, index=x.columns).sort_values(ascending=False)
    figfi = px.bar(importances, title='Feature Importance - Final Decision Tree')
    figfi.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Robustness Check

    A single train/test split result depends partly on which samples land in the test set. With only 6 depressed teens in test, one hard case can drop F1 from 1.0 to 0.83.

    Cross-validation over 10 folds gives a more honest estimate of real performance.
    """)
    return


@app.cell
def _(cross_val_score, dtfinal, x, y):
    scoress = cross_val_score(dtfinal, x, y, cv=10, scoring='f1')
    print(f'Mean F1: {scoress.mean():.3f} (+/- {scoress.std():.3f})')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Not 100% Accuracy?

    The final model scores **F1=0.91** on this split, not 1.0 — and that's honest, not a failure.

    **The hard case (index 691):**

    | Feature | Her value | Depressed avg | Direction |
    |---|---|---|---|
    | `stress_level` | 9 | 8.48 | ✓ high |
    | `anxiety_level` | 7 | 8.61 | ✗ slightly low |
    | `addiction_level` | 5 | 5.32 | ✗ near non-depressed avg |
    | `sleep_hours` | 5.7 | low | ✓ consistent |
    | `daily_social_media_hours` | 8.0 | high | ✓ consistent |

    She's a genuine outlier — depressed but with moderate anxiety and low addiction, which contradicts the pattern the model learned from the majority of cases. No amount of tuning fixes a statistically ambiguous sample.

    **On the other notebook claiming 100%:**

    Running the same model across 200 random seeds shows F1=1.0 on almost all splits. The single-seed 100% result depends on which 6 depressed teens land in the test set — a favorable split puts easier cases there. One harder split drops it to 0.91.

    Reporting cross-validated mean F1 of **0.930 ± 0.155** is more honest and reproducible than a single lucky seed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Hard Case Analysis

    Identify which depressed teen gets misclassified most often across 200 different train/test splits. This reveals genuine edge cases — samples whose feature profile doesn't match the typical pattern the model learned.
    """)
    return


@app.cell
def _(
    Counter,
    DecisionTreeClassifier,
    SMOTETomek,
    study,
    train_test_split,
    x,
    y,
):
    misclassified = Counter()

    for s in range(200):
        xa, xb, ya, yb = train_test_split(x, y, test_size=0.2, random_state=s, stratify=y)
        sta = SMOTETomek(random_state=67)
        xastk, yastk = sta.fit_resample(xa, ya)
        dta = DecisionTreeClassifier(**study.best_params, random_state=67)
        dta.fit(xastk, yastk)
        preda = dta.predict(xb)
        wrong = xb[(yb == 1) & (preda == 0)].index
        misclassified.update(wrong.tolist())

    print('Most misclassified indices:', misclassified.most_common(5))
    return (misclassified,)


@app.cell
def _(misclassified, sdf):
    hard_idx = misclassified.most_common(1)[0][0]
    print('Hard case profile:')
    print(sdf.loc[[hard_idx]].T)
    print()
    print('Class 1 feature averages:')
    print(sdf.groupby('depression_label')[['stress_level', 'anxiety_level', 'addiction_level']].mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations

    - Depression, anxiety, and addiction labels are assumed self-reported with no
      documented clinical instrument (e.g. PHQ-9, GAD-7, AUDIT). Self-report
      validity for these constructs is low.
    - Numeric scales (1–10) have no documented anchors — a score of 7 carries
      different meaning across respondents, time periods, and cultures.
    - Dataset provenance is unverified. The original source and collection
      methodology are undocumented, making clinical generalization impossible.
    - Model performance metrics reflect pattern-matching on potentially noisy labels,
      not genuine predictive validity for depression screening.
    """)
    return


if __name__ == "__main__":
    app.run()
