import pandas as pd
import numpy as np
from langchain_core.tools import tool


CURRENT_DF: pd.DataFrame | None = None


def set_dataframe(df: pd.DataFrame):
    global CURRENT_DF
    CURRENT_DF = df


def _require_df() -> pd.DataFrame:
    if CURRENT_DF is None:
        raise ValueError("No CSV has been loaded yet.")
    return CURRENT_DF


@tool
def describe_dataset() -> str:
    """Describe the currently loaded transaction dataset."""
    df = _require_df()

    return (
        f"Rows: {len(df)}\n"
        f"Columns: {list(df.columns)}\n"
        f"Data types:\n{df.dtypes.to_string()}\n"
        f"Missing values:\n{df.isnull().sum().to_string()}\n\n"
        f"Sample:\n{df.head(5).to_string(index=False)}"
    )


@tool
def transaction_statistics() -> str:
    """Calculate basic transaction amount statistics."""
    df = _require_df()

    if "amount" not in df.columns:
        return "The dataset does not contain an 'amount' column."

    amount = pd.to_numeric(df["amount"], errors="coerce")

    return (
        f"Transaction count: {amount.count()}\n"
        f"Total amount: {amount.sum():.2f}\n"
        f"Average amount: {amount.mean():.2f}\n"
        f"Median amount: {amount.median():.2f}\n"
        f"Maximum amount: {amount.max():.2f}\n"
        f"Minimum amount: {amount.min():.2f}"
    )


@tool
def customer_analysis() -> str:
    """Analyze transaction activity by customer."""
    df = _require_df()

    required = {"account_id", "amount"}
    missing = required - set(df.columns)

    if missing:
        return f"Missing columns: {', '.join(missing)}"

    result = (
        df.groupby("account_id")
        .agg(
            transaction_count=("amount", "count"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
        )
        .sort_values("total_amount", ascending=False)
        .head(10)
    )

    return result.to_string()


@tool
def high_value_transactions(threshold: float = 500000) -> str:
    """
    Find transactions above a specified amount threshold.
    Default threshold is 500000.
    """
    df = _require_df()

    if "amount" not in df.columns:
        return "The dataset does not contain an 'amount' column."

    result = df[df["amount"] >= threshold]

    if result.empty:
        return f"No transactions found above {threshold:.2f}."

    return (
        f"Found {len(result)} transactions above {threshold:.2f}:\n\n"
        f"{result.to_string(index=False)}"
    )


@tool
def transaction_velocity() -> str:
    """Identify customers with unusually high transaction frequency."""
    df = _require_df()

    required = {"account_id"}
    missing = required - set(df.columns)

    if missing:
        return f"Missing columns: {', '.join(missing)}"

    counts = (
        df.groupby("account_id")
        .size()
        .reset_index(name="transaction_count")
        .sort_values("transaction_count", ascending=False)
    )

    return counts.head(10).to_string(index=False)


@tool
def structuring_detection(threshold: float = 500000) -> str:
    """
    Identify customers making multiple transactions below a specified
    threshold, which may indicate potential structuring.
    """
    df = _require_df()

    required = {"account_id", "amount"}
    missing = required - set(df.columns)

    if missing:
        return f"Missing columns: {', '.join(missing)}"

    suspicious = df[df["amount"] < threshold]

    result = (
        suspicious.groupby("account_id")
        .agg(
            transaction_count=("amount", "count"),
            total_amount=("amount", "sum"),
        )
        .query("transaction_count >= 3")
        .sort_values("transaction_count", ascending=False)
    )

    if result.empty:
        return "No potential structuring patterns detected."

    return result.to_string()


TOOLS = [
    describe_dataset,
    transaction_statistics,
    customer_analysis,
    high_value_transactions,
    transaction_velocity,
    structuring_detection,
]