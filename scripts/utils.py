import os
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------
# File and path utilities
# ---------------------------

def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def resolve_path(*parts: str) -> str:
    """Join path parts with correct separator."""
    return os.path.join(*parts)


# ---------------------------
# Data IO
# ---------------------------

def load_csv(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a CSV with optional datetime parsing.

    Parameters
    ----------
    path : str
        File path to the CSV file.
    parse_dates : Optional[List[str]]
        Column names to parse as datetimes.
    """
    df = pd.read_csv(path)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV with directories created automatically."""
    ensure_directory(os.path.dirname(path))
    df.to_csv(path, index=False)


# ---------------------------
# Cleaning helpers
# ---------------------------

def coerce_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce multiple columns to datetime (inplace-safe pattern)."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce multiple columns to numeric (inplace-safe pattern)."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_engineered_columns(
    df: pd.DataFrame,
    rat_start_col: str = "rat_period_start",
    rat_end_col: str = "rat_period_end",
    bat_to_food_col: str = "bat_landing_to_food",
    hours_after_sunset_col: str = "hours_after_sunset",
) -> pd.DataFrame:
    """Add engineered columns required by the spec.

    Adds:
    - rat_presence_duration = rat_period_end - rat_period_start (seconds)
    - response_delay = bat_landing_to_food (alias for clarity)
    - is_night = hours_after_sunset > 0
    """
    df = df.copy()

    if rat_start_col in df.columns and rat_end_col in df.columns:
        # Try datetime differences; if numeric, fallback to numeric diff
        if np.issubdtype(df[rat_start_col].dtype, np.datetime64) and np.issubdtype(
            df[rat_end_col].dtype, np.datetime64
        ):
            duration = (df[rat_end_col] - df[rat_start_col]).dt.total_seconds()
        else:
            duration = pd.to_numeric(df[rat_end_col], errors="coerce") - pd.to_numeric(
                df[rat_start_col], errors="coerce"
            )
        df["rat_presence_duration"] = duration

    if bat_to_food_col in df.columns:
        df["response_delay"] = pd.to_numeric(df[bat_to_food_col], errors="coerce")

    if hours_after_sunset_col in df.columns:
        df["is_night"] = pd.to_numeric(df[hours_after_sunset_col], errors="coerce") > 0

    return df


def basic_info(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return tuple of (dtypes, missing_counts, describe_numeric)."""
    dtypes = df.dtypes.rename("dtype").to_frame()
    missing = df.isna().sum()
    describe = df.describe(include=[np.number]).T
    return dtypes, missing, describe


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values with a simple strategy.

    Parameters
    ----------
    strategy : {"drop", "median", "mean"}
        - drop: drop rows with any NA
        - median/mean: fill numeric columns with the aggregator; leave non-numeric as-is
    """
    df = df.copy()
    if strategy == "drop":
        return df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "median":
        fill_values = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    elif strategy == "mean":
        fill_values = df[numeric_cols].mean()
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    else:
        raise ValueError("Unknown missing strategy: {0}".format(strategy))
    return df


# ---------------------------
# Visualization helpers
# ---------------------------

def save_figure(fig: plt.Figure, path: str) -> None:
    ensure_directory(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> plt.Figure:
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title(title)
    return fig


def labeled_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig


def labeled_hist(
    data: pd.Series,
    bins: int,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: str = "Count",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data.dropna(), bins=bins, ax=ax, kde=False)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def labeled_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    estimator = np.mean,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=data, x=x, y=y, estimator=estimator, ci=95, ax=ax)
    ax.set_title(title)
    return fig


def labeled_violinplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=data, x=x, y=y, cut=0, inner="box", ax=ax)
    ax.set_title(title)
    return fig


def labeled_lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=data, x=x, y=y, marker="o", ax=ax)
    ax.set_title(title)
    return fig


# ---------------------------
# Statistical helpers
# ---------------------------

def pearson_correlation(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Return (r, p_value)."""
    x_clean = pd.to_numeric(x, errors="coerce")
    y_clean = pd.to_numeric(y, errors="coerce")
    mask = x_clean.notna() & y_clean.notna()
    return stats.pearsonr(x_clean[mask], y_clean[mask])


def logistic_regression(
    df: pd.DataFrame,
    formula: str,
) -> Any:
    """Fit a logistic regression using statsmodels (GLM with binomial family).

    Returns a GLMResultsWrapper; return type kept generic for
    compatibility across statsmodels versions.
    """
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
    return model


def t_test_independent(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Two-sample independent t-test on numeric series (drop NA)."""
    a_clean = pd.to_numeric(a, errors="coerce").dropna()
    b_clean = pd.to_numeric(b, errors="coerce").dropna()
    return stats.ttest_ind(a_clean, b_clean, equal_var=False)


# ---------------------------
# Table saving
# ---------------------------

def save_table(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame table to CSV under outputs/tables."""
    ensure_directory(os.path.dirname(path))
    df.to_csv(path, index=True)
