import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


class EcommerceAnalysis:
    """
    Professional Ecommerce Data Analysis Pipeline
    """

    def __init__(self, file_path: str, discount_rate=0.05, tax_rate=0.10):

        self.raw_df = pd.read_csv(file_path)
        self.df = self.raw_df.copy()

        self.discount_rate = discount_rate
        self.tax_rate = tax_rate

        # =========================
        # ONLY CHANGE: PRINT → LOGGING
        # =========================
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline Started")

        self._validate_columns()

    # =========================
    # VALIDATION
    # =========================
    def _validate_columns(self):
        required_cols = [
            "customer_name", "email", "product",
            "price", "quantity", "order_date", "city"
        ]

        missing = [col for col in required_cols if col not in self.df.columns]

        if missing:
            self.logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Missing columns: {missing}")

    def validate_clean_data(self):
        assert self.df["price"].isna().sum() == 0
        assert self.df["quantity"].isna().sum() == 0
        self.logger.info("Validation Passed")

    # =========================
    # CLEANING
    # =========================
    def convert_types(self):
        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")
        self.df["quantity"] = pd.to_numeric(self.df["quantity"], errors="coerce")
        self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors="coerce")

    def clean_strings(self):
        self.df["customer_name"] = self.df["customer_name"].astype(str).str.strip()
        self.df["email"] = self.df["email"].astype(str).str.strip().str.lower()
        self.df["product"] = self.df["product"].astype(str).str.strip().str.lower()
        self.df["city"] = self.df["city"].astype(str).str.strip()

    def fix_invalid_values(self):
        self.df.loc[self.df["price"] < 0, "price"] = np.nan
        self.df.loc[self.df["quantity"] <= 0, "quantity"] = np.nan

    def handle_missing(self):
        self.df["price"] = self.df["price"].fillna(self.df["price"].median())
        self.df["quantity"] = self.df["quantity"].fillna(self.df["quantity"].median())
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
    def create_customer_id(self):
        self.df["customer_id"] = self.df["email"].factorize()[0]

    def create_revenue_features(self):
        self.df["total_amount"] = self.df["price"] * self.df["quantity"]
        self.df["discount"] = self.df["total_amount"] * self.discount_rate
        self.df["final_amount"] = self.df["total_amount"] - self.df["discount"]
        self.df["tax"] = self.df["final_amount"] * self.tax_rate
        self.df["net_revenue"] = self.df["final_amount"] - self.df["tax"]

    def extract_date_features(self):
        self.df["order_month"] = self.df["order_date"].dt.month

    def extract_email_domain(self):
        self.df["email_domain"] = self.df["email"].str.split("@").str[-1]

    # =========================
    # ANALYSIS
    # =========================
    def sales_by_product(self):
        return self.df.groupby("product")["final_amount"].sum().sort_values(ascending=False)

    def sales_by_city(self):
        return self.df.groupby("city")["final_amount"].sum().sort_values(ascending=False)

    def monthly_sales(self):
        return self.df.groupby("order_month")["final_amount"].sum().sort_index()

    # =========================
    # ADVANCED ANALYSIS
    # =========================
    def customer_lifetime_value(self):
        return self.df.groupby("customer_id")["final_amount"].sum().sort_values(ascending=False)

    def top_20_percent_customers(self):
        clv = self.customer_lifetime_value()
        cutoff = int(0.2 * len(clv))
        return clv.head(cutoff)

    def email_domain_analysis(self):
        return self.df["email_domain"].value_counts()

    # =========================
    # BUSINESS INSIGHTS
    # =========================
    def generate_insights(self):
        self.logger.info("===== BUSINESS INSIGHTS =====")
        self.logger.info(f"Top Product: {self.sales_by_product().idxmax()}")
        self.logger.info(f"Top City: {self.sales_by_city().idxmax()}")
        self.logger.info(f"Total Revenue: {round(self.df['final_amount'].sum(), 2)}")

    # =========================
    # VISUALIZATION (NO CHANGE)
    # =========================
    def plot_monthly_sales(self):
        data = self.monthly_sales()

        plt.figure(figsize=(10, 5))
        data.plot(kind="bar")

        plt.title("Monthly Sales", fontsize=14, weight="bold")
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

        plt.title("Top Products", fontsize=14, weight="bold")
        plt.xlabel("Product")
        plt.ylabel("Revenue")

        plt.xticks(rotation=0)
        plt.grid(alpha=0.3)

        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.show()

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
        self.create_customer_id()
        self.create_revenue_features()
        self.extract_date_features()
        self.extract_email_domain()
        self.validate_clean_data()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    project = EcommerceAnalysis("ecommerce_raw_data.csv")

    project.run_pipeline()

    print("\nCLV:\n", project.customer_lifetime_value().head())
    print("\nTop 20% Customers:\n", project.top_20_percent_customers())
    print("\nEmail Domains:\n", project.email_domain_analysis())

    project.generate_insights()

    project.plot_monthly_sales()
    project.plot_top_products()