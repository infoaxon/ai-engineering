import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CDP Segmentation Demo", layout="wide")


@st.cache_data
def load_data(path):
    # Load key sheets from the Excel file
    demographics = pd.read_excel(path, sheet_name='Demographics')
    actions = pd.read_excel(path, sheet_name='UserActionProfile')
    metrics = pd.read_excel(path, sheet_name='LifetimeMetrics')
    # Merge into master dataframe
    df = demographics[['recordId', 'age']].merge(
        actions[['recordId', 'numOfLogins30d']], on='recordId', how='left'
    ).merge(
        metrics[['recordId', 'totalCustomerLifetimeValue']], on='recordId', how='left'
    )
    return df


def assign_segment(row, high_ltv, high_logins):
    if row['totalCustomerLifetimeValue'] >= high_ltv and row['numOfLogins30d'] >= high_logins:
        return 'High-Value & Engaged'
    elif row['totalCustomerLifetimeValue'] < (high_ltv * 0.3) and row['numOfLogins30d'] < (high_logins * 0.5):
        return 'Low-Value & Low Engagement'
    else:
        return 'Mid-Tier'


def main():
    st.title("CDP Customer Segmentation Demo")
    st.markdown(
        "Visualize segments based on Lifetime Value vs. Recent Engagement")

    # File uploader
    uploaded = st.file_uploader("Upload your Excel file", type=['xlsx'])
    if uploaded:
        df = load_data(uploaded)

        # Sidebar controls
        st.sidebar.header("Segment Thresholds")
        high_ltv = st.sidebar.slider(
            "High LTV Threshold",
            int(df['totalCustomerLifetimeValue'].min()),
            int(df['totalCustomerLifetimeValue'].max()),
            100000
        )
        high_logins = st.sidebar.slider(
            "High Logins Threshold (30d)",
            int(df['numOfLogins30d'].min()),
            int(df['numOfLogins30d'].max()),
            10
        )

        # Assign segments
        df['segment'] = df.apply(
            assign_segment, axis=1, args=(high_ltv, high_logins))

        # Show data and metrics
        st.subheader("Segment Distribution")
        st.bar_chart(df['segment'].value_counts())

        # Scatter plot
        st.subheader("LTV vs. Logins Scatter")
        fig, ax = plt.subplots()
        for seg, group in df.groupby('segment'):
            ax.scatter(
                group['numOfLogins30d'],
                group['totalCustomerLifetimeValue'],
                label=seg
            )
        ax.set_xlabel('Logins in Last 30 Days')
        ax.set_ylabel('Customer Lifetime Value')
        ax.legend()
        st.pyplot(fig)

        # Show sample records
        st.subheader("Sample Records")
        st.dataframe(df.head(10))


if __name__ == "__main__":
    main()
