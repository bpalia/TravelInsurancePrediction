# Last updated November 15, 2023
# Version 0.1.3

import numpy as np
import pandas as pd
from IPython.display import display
from typing import List
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
    make_scorer,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from imblearn.metrics import (
    macro_averaged_mean_absolute_error,
    classification_report_imbalanced,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    f_regression,
    f_classif,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def model_representation(
    pipe: Pipeline, type: str = "classif", labels: list = None
) -> None:
    """Representation of sklearn classifier or regressor.
    Type can be 'classif' or 'regression'."""
    if len(pipe) > 2:
        imputer = pipe[0]
        scaler = pipe[1]
        model = pipe[2]
        imputer_repr = pd.DataFrame(
            {imputer.strategy: imputer.statistics_},
            index=pipe.feature_names_in_,
        ).T
        print("IMPUTER")
        display(imputer_repr)
    else:
        scaler = pipe[0]
        model = pipe[1]
    scaler_repr = pd.DataFrame(
        {"Mean": scaler.mean_, "Scale": scaler.scale_},
        index=pipe.feature_names_in_,
    ).T
    print("SCALER")
    display(scaler_repr)
    if type == "classif":
        if labels is not None:
            idx = labels
        else:
            idx = pipe.classes_
        model_repr = pd.DataFrame(
            model.coef_, index=idx, columns=pipe.feature_names_in_
        )
        model_repr.index.name = "Class"
    elif type == "regression":
        model_repr = pd.DataFrame(model.coef_, index=pipe.feature_names_in_).T
    model_repr["Intercept"] = model.intercept_
    print("MODEL COEFFICIENTS")
    display(model_repr)
    return


def print_classification_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list = None,
) -> None:
    """Function to print classification results: macro averaged mean absolute
    error, classification metrics for imbalanced data and confusion matrix."""
    print(
        "Classification metrics:"
        f" \n{classification_report_imbalanced(y_true, y_pred, target_names=labels, zero_division=0)}"
    )
    _, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize="true",
        ax=ax,
        colorbar=False,
        cmap="coolwarm",
    )
    if labels is not None:
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
    plt.grid(False)
    plt.show()
    return


def print_regression_results(
    y_true: pd.Series, y_pred: np.ndarray, y_name: str
) -> Figure:
    """Function to print regression results: root mean square error,
    coefficient of determination, and scatterplots of predicted values against
    observed values, and of predicted values against residuals."""
    residuals = y_true - y_pred
    scatter_kws = {"alpha": 0.4, "s": 20}
    line_kws = {"color": "rosybrown"}
    fig, axes = plt.subplots(1, 2, sharex=True)
    sns.regplot(
        x=y_pred,
        y=y_true,
        ax=axes[0],
        scatter_kws=scatter_kws,
        x_jitter=0.2,
        y_jitter=0.2,
        fit_reg=False,
    )
    sns.regplot(
        x=y_pred,
        y=y_true,
        ax=axes[0],
        line_kws=line_kws,
        x_estimator=np.mean,
        x_ci="sd",
    )
    sns.regplot(
        x=y_pred,
        y=residuals,
        ax=axes[1],
        scatter_kws=scatter_kws,
        x_jitter=0.2,
        y_jitter=0.2,
        fit_reg=False,
    )
    sns.regplot(
        x=y_pred,
        y=residuals,
        ax=axes[1],
        line_kws=line_kws,
        x_estimator=np.mean,
        x_ci="sd",
    )
    axes[0].set(xlabel=("Predicted " + y_name), ylabel=("Observed " + y_name))
    axes[1].set(xlabel=("Predicted " + y_name), ylabel="Residuals")
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE = {round(RMSE, 3)}, R^2 = {round(r2, 3)}")
    return fig


def feature_selection_estimates(
    X: pd.DataFrame, y: pd.Series, type: str = "classif"
) -> pd.DataFrame:
    """Function to estimate Spearman's rank correlation coefficient,
    F-statistics and its p-values, and mutual information to accommodate feature
    selection. Type can be 'classif' or 'regression'."""
    measures = pd.DataFrame(index=X.columns)
    measures["Spearman's rho"] = X.corrwith(y, method="spearman")
    if type == "classif":
        f_stat, p_val = f_classif(X, y)
        mi = mutual_info_classif(X, y, random_state=0, discrete_features=False)
    elif type == "regression":
        f_stat, p_val = f_regression(X, y)
        mi = mutual_info_regression(
            X, y, random_state=0, discrete_features=False
        )
    measures["F-statistic"] = f_stat
    measures["p-value"] = p_val
    measures["Mutual Information"] = mi
    return measures


def select_k_best(
    X: pd.DataFrame, y: pd.Series, k: int, type: str = "classif"
) -> list:
    """Function to select top k features based on mutual information.
    Type can be 'classif' or 'regression'."""
    if type == "classif":
        mi = mutual_info_classif(X, y, random_state=0)
    elif type == "regression":
        mi = mutual_info_regression(X, y, random_state=0)
    return pd.Series(mi, index=X.columns).nlargest(n=k).index.to_list()


def vif_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the variance inflation factor for each feature in the dataframe."""
    vif_data = pd.DataFrame({"Feature": df.columns})
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(len(df.columns))
    ]
    return vif_data


def cv_score_binary_classifiers(
    clf_names: List[str],
    clf_models: List[object],
    fold: BaseCrossValidator,
    X: pd.DataFrame,
    y: pd.Series,
    beta: float = 2,
) -> None:
    """Return mean scores on cross-validation of multiple binary classifiers."""
    recall = []
    precision = []
    auc = []
    accuracy = []
    balanced_accuracy = []
    f1 = []
    fbeta = []
    scoring = {
        "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0.0),
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "fbeta": make_scorer(fbeta_score, beta=beta),
    }
    for model in clf_models:
        cv_res = cross_validate(model, X, y, cv=fold, scoring=scoring)
        recall.append(cv_res["test_recall"].mean())
        precision.append(cv_res["test_precision"].mean())
        auc.append(cv_res["test_roc_auc"].mean())
        accuracy.append(cv_res["test_accuracy"].mean())
        balanced_accuracy.append(cv_res["test_balanced_accuracy"].mean())
        f1.append(cv_res["test_f1"].mean())
        fbeta.append(cv_res["test_fbeta"].mean())
    res_df = pd.DataFrame(
        {
            "AUC": auc,
            "Recall": recall,
            "Precision": precision,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "F1": f1,
            "F-beta": fbeta,
        },
        index=clf_names,
    )
    display(
        res_df.style.highlight_max(color="green", axis=0).format(precision=3)
    )
    return


def score_binary_fitted_classifiers(
    clf_names: List[str],
    clf_models: List[object],
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[str] = None,
    beta: float = 2,
) -> Figure:
    """Return scores of multiple binary fitted classifiers.
    Need to predict probabilities also."""
    recall = []
    precision = []
    auc = []
    accuracy = []
    balanced_accuracy = []
    f1 = []
    fbeta = []
    num_models = len(clf_models)
    if num_models > 3:
        num_rows, _ = divmod(num_models, 3)
        fig, axes = plt.subplots(num_rows + 1, 3)
    else:
        fig, axes = plt.subplots(1, num_models)
    axes = axes.flatten()
    for idx, model in enumerate(clf_models):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        recall.append(recall_score(y, y_pred))
        precision.append(precision_score(y, y_pred))
        auc.append(roc_auc_score(y, y_proba[:, 1]))
        accuracy.append(accuracy_score(y, y_pred))
        balanced_accuracy.append(balanced_accuracy_score(y, y_pred))
        f1.append(f1_score(y, y_pred))
        fbeta.append(fbeta_score(y, y_pred, beta=beta))
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred, display_labels=labels, ax=axes[idx], colorbar=False
        )
        axes[idx].set_title(clf_names[idx])
        axes[idx].grid(False)
    fig.suptitle("Confusion Matrices")
    for ax in axes[num_models:]:
        ax.axis("off")
    res_df = pd.DataFrame(
        {
            "AUC": auc,
            "Recall": recall,
            "Precision": precision,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "F1": f1,
            "F-beta": fbeta,
        },
        index=clf_names,
    )
    display(
        res_df.style.highlight_max(color="green", axis=0).format(precision=3)
    )
    return fig
