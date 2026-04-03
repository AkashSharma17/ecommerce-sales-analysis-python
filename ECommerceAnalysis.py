import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EcommerceAnalysis:
    """
    Cleaned & Professional Ecommerce Data Analysis Pipeline
    """

    def __init__(self, file_path: str, discount_rate=0.05, tax_rate=0.10):
        """
        Initialize dataset safely
        """
        self.raw_df = pd.read_csv(file_path)
        self.df = self.raw_df.copy()

        self.discount_rate = discount_rate
        self.tax_rate = tax_rate

        self._validate_columns()

    # =========================
    # COLUMN VALIDATION
    # =========================
    def _validate_columns(self):
        required_cols = [
            "customer_name", "email", "product",
            "price", "quantity", "order_date", "city"
        ]

        missing = [col for col in required_cols if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

    # =========================
    # DATA INSPECTION
    # =========================
    def inspect_data(self):
        print("\n===== DATA INFO =====\n")
        self.df.info()

        print("\n===== SHAPE =====\n")
        print(self.df.shape)

        print("\n===== MISSING VALUES =====\n")
        print(self.df.isna().sum())

    # =========================
    # CLEANING STEPS
    # =========================
    def convert_types(self):
        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")
        self.df["quantity"] = pd.to_numeric(self.df["quantity"], errors="coerce")
        self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors="coerce")

    def clean_strings(self):
        self.df["customer_name"] = self.df["customer_name"].astype(str).str.strip()
        self.df["email"] = self.df["email"].astype(str).str.strip().str.lower()
        self.df["product"] = self.df["product"].astype(str).str.lower().str.strip()
        self.df["city"] = self.df["city"].astype(str).str.strip()

    def fix_invalid_values(self):
        if "age" in self.df.columns:
            self.df.loc[(self.df["age"] < 0) | (self.df["age"] > 100), "age"] = np.nan

        self.df.loc[self.df["price"] < 0, "price"] = np.nan
        self.df.loc[self.df["quantity"] <= 0, "quantity"] = np.nan

    def handle_missing(self):
        self.df["price"] = self.df["price"].fillna(self.df["price"].median())
        self.df["quantity"] = self.df["quantity"].fillna(self.df["quantity"].median())

        if "age" in self.df.columns:
            self.df["age"] = self.df["age"].fillna(self.df["age"].median())

        self.df["email"] = self.df["email"].fillna("unknown@gmail.com")
        self.df["order_date"] = self.df["order_date"].fillna(pd.Timestamp("2024-01-01"))

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates(
            subset=["customer_name", "order_date", "product"]
        )

    def remove_outliers(self):
        q1 = self.df["price"].quantile(0.25)
        q3 = self.df["price"].quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        self.df = self.df[
            (self.df["price"] >= lower) &
            (self.df["price"] <= upper)
        ]

    # =========================
    # FEATURE ENGINEERING
    # =========================
    def create_revenue_features(self):
        self.df["total_amount"] = self.df["price"] * self.df["quantity"]

        self.df["discount"] = self.df["total_amount"] * self.discount_rate
        self.df["final_amount"] = self.df["total_amount"] - self.df["discount"]

        self.df["tax"] = self.df["final_amount"] * self.tax_rate
        self.df["net_revenue"] = self.df["final_amount"] - self.df["tax"]

    def extract_date_features(self):
        self.df["order_month"] = self.df["order_date"].dt.month
        self.df["order_day"] = self.df["order_date"].dt.day

    def extract_email_domain(self):
        self.df["email_domain"] = self.df["email"].str.split("@").str[-1]

    # =========================
    # ANALYSIS
    # =========================
    def top_customers(self):
        return self.df.sort_values(by="final_amount", ascending=False).head(5)

    def sales_by_product(self):
        return self.df.groupby("product")["final_amount"].sum().sort_values(ascending=False)

    def sales_by_city(self):
        return self.df.groupby("city")["final_amount"].sum().sort_values(ascending=False)

    def monthly_sales(self):
        return self.df.groupby("order_month")["final_amount"].sum().sort_index()

    def product_stats(self):
        return self.df.groupby("product")["final_amount"].agg(
            total_sales="sum",
            avg_sales="mean",
            order_count="count"
        )

    # =========================
    # VISUALIZATION (UPGRADED)
    # =========================
    def plot_monthly_sales(self):
        """
        Plots monthly revenue using a clean bar chart.
        """
        data = self.monthly_sales()

        plt.figure(figsize=(10, 5))

        data.sort_values().plot(kind="barh")

        plt.title("Monthly Sales Analysis", fontsize=14, weight="bold")
        plt.xlabel("Month")
        plt.ylabel("Revenue")

        plt.xticks(rotation=0)
        plt.grid(alpha=0.3)
        
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_top_products(self):
        data = self.sales_by_product().head(10)

        plt.figure(figsize=(10, 5))

        data.plot(kind="bar")

        plt.title("Top 10 Products by Revenue", fontsize=14, weight="bold")
        plt.xlabel("Product")
        plt.ylabel("Revenue")

        plt.xticks(rotation=0)
        plt.grid(alpha=0.3)
        
        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.show()

    # =========================
    # EXPORT
    # =========================
    def export_data(self, file_name="cleaned_ecommerce_data.csv"):
        self.df.to_csv(file_name, index=False)

    # =========================
    # PIPELINE
    # =========================
    def run_pipeline(self):
        self.convert_types()
        self.clean_strings()
        self.fix_invalid_values()
        self.handle_missing()
        self.remove_duplicates()
        self.remove_outliers()
        self.create_revenue_features()
        self.extract_date_features()
        self.extract_email_domain()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    project = EcommerceAnalysis("ecommerce_raw_data.csv")

    # Inspect
    project.inspect_data()

    # Run pipeline
    project.run_pipeline()

    # Analysis
    print("\nTop Customers:\n", project.top_customers())
    print("\nSales by Product:\n", project.sales_by_product())
    print("\nSales by City:\n", project.sales_by_city())
    print("\nMonthly Sales:\n", project.monthly_sales())

    # Visuals
    project.plot_monthly_sales()
    project.plot_top_products()

    # Export
    project.export_data()