import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_invoice_data(db_path):
    """
    Load invoice and purchase data from SQLite database.
    """

    conn = sqlite3.connect(db_path)

    query = """
    WITH purchases_agg AS (
        SELECT
            PONumber,
            COUNT(DISTINCT Brand) AS total_brands,
            SUM(Quantity) AS total_items_quantity,
            SUM(Dollars) AS total_item_dollars,
            AVG(julianday(ReceivingDate) - julianday(PODate)) AS avg_received_delay
        FROM purchases
        GROUP BY PONumber
    )

    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,

        (julianday(vi.InvoiceDate) - julianday(vi.PODate))
            AS days_po_to_invoice,

        (julianday(vi.PayDate) - julianday(vi.PODate))
            AS days_to_pay,

        pa.total_brands,
        pa.total_items_quantity,
        pa.total_item_dollars,
        pa.avg_received_delay

    FROM vendor_invoice vi

    LEFT JOIN purchases_agg pa
    ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Handle missing values after LEFT JOIN
    df.fillna(0, inplace=True)

    return df


def create_features(df):
    """
    Create additional engineered features.
    """

    df["amount_difference"] = (
        df["invoice_dollars"] - df["total_item_dollars"]
    ).abs()

    if "invoice_quantity" in df.columns and "total_items_quantity" in df.columns:
        df["quantity_difference"] = (
            df["invoice_quantity"] - df["total_items_quantity"]
        ).abs()

    return df


def create_invoice_risk_label(row):
    """
    Rule-based invoice flag.
    """

    if row["amount_difference"] > 5:
        return 1

    if row["avg_received_delay"] > 10:
        return 1

    return 0


def apply_label(df):
    """
    Generate target column.
    """

    df = create_features(df)

    df["flag_invoice"] = df.apply(
        create_invoice_risk_label,
        axis=1
    )

    return df


def split_data(df, features, target):
    """
    Split dataset into train and test.
    """

    X = df[features]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )