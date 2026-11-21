import streamlit as st
import pandas as pd
import plotly.express as px

from tools.data_analysis import set_dataset
from agent.agent import create_data_analyst


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")
st.write("Upload your dataset and ask questions about your data.")


# -----------------------------
# UPLOAD DATASET
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Make dataset available to tools
        set_dataset(df)

        st.success("Dataset uploaded successfully!")

        # -----------------------------
        # DATASET PREVIEW
        # -----------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )


        # -----------------------------
        # DATASET INFORMATION
        # -----------------------------

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )


        # -----------------------------
        # BASIC REVENUE DASHBOARD
        # -----------------------------

        if {"Quantity", "Price"}.issubset(df.columns):

            df["Quantity"] = pd.to_numeric(
                df["Quantity"],
                errors="coerce"
            )

            df["Price"] = pd.to_numeric(
                df["Price"],
                errors="coerce"
            )

            df["Revenue"] = (
                df["Quantity"] * df["Price"]
            )

            st.divider()

            st.header("💰 Revenue Dashboard")

            total_revenue = df["Revenue"].sum()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Revenue",
                    f"₹{total_revenue:,.2f}"
                )

            with col2:
                st.metric(
                    "Total Orders",
                    f"{len(df):,}"
                )


            # Category revenue
            if "Category" in df.columns:

                category_revenue = (
                    df.groupby("Category")["Revenue"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                st.subheader("Revenue by Category")

                fig = px.bar(
                    category_revenue,
                    x="Category",
                    y="Revenue",
                    text_auto=".2s"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


            # Product revenue
            if "Product" in df.columns:

                product_revenue = (
                    df.groupby("Product")["Revenue"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index()
                )

                st.subheader("Top 5 Products")

                fig = px.bar(
                    product_revenue,
                    x="Product",
                    y="Revenue",
                    text_auto=".2s"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        # -----------------------------
        # AI DATA ANALYST
        # -----------------------------

        st.divider()

        st.header("🤖 Ask AI Data Analyst")

        question = st.text_input(
            "Ask a question about your dataset",
            placeholder=(
                "e.g. Which product sold the most in Bangalore?"
            )
        )


        if st.button("Analyze"):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Analyzing your dataset..."
                ):

                    try:

                        # Create agent
                        agent = create_data_analyst()

                        # Ask agent
                        response = agent.invoke(
                            {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": question
                                    }
                                ]
                            }
                        )


                        # Get final response
                        final_message = response[
                            "messages"
                        ][-1]

                        content = final_message.content


                        # Handle different content formats
                        if isinstance(content, str):

                            answer = content

                        else:

                            answer = ""

                            for block in content:

                                if isinstance(block, dict):

                                    if block.get("type") == "text":

                                        answer += block.get(
                                            "text",
                                            ""
                                        )


                        # Display result
                        st.subheader(
                            "📋 AI Analysis"
                        )

                        st.markdown(answer)


                    except Exception as e:

                        st.error(
                            f"Unable to analyze the dataset: {e}"
                        )


    except Exception as e:

        st.error(
            f"Unable to process the dataset: {e}"
        )