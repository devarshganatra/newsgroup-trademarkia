"""
Semantic Search Observability Dashboard.

Run with:
    streamlit run dashboard/cache_dashboard.py

Requires the FastAPI server running at http://localhost:8000
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://localhost:8000/dashboard/metrics"

st.set_page_config(
    page_title="Semantic Search Observability",
    layout="wide"
)

st.title(" Semantic Search Observability Dashboard")
st.markdown("Live performance metrics for the cluster-aware semantic cache.")

# Sidebar controls
st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if st.sidebar.button(" Refresh Now"):
    st.rerun()


@st.cache_data(ttl=2)
def fetch_metrics():
    """Fetch metrics from the API with a 2-second cache."""
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
    return None


metrics = fetch_metrics()

if metrics:
    # Top-level KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Queries", metrics["total_queries"])
    kpi2.metric("Cache Hit Rate", f"{metrics['hit_rate'] * 100:.1f}%")
    kpi3.metric("Avg Latency", f"{metrics['average_latency_ms']:.1f} ms")
    kpi4.metric("Cache Entries", metrics["cache_hits"] + metrics["cache_misses"])

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Query Distribution")
        dist = metrics["cluster_distribution"]
        # Filter to only clusters with queries
        active = {k: v for k, v in dist.items() if v > 0}
        if active:
            df = pd.DataFrame({"Cluster": list(active.keys()), "Queries": list(active.values())})
            df = df.sort_values("Cluster")
            st.bar_chart(df.set_index("Cluster"))
        else:
            st.info("No queries routed to clusters yet.")

    with col2:
        st.subheader("Cache Performance")
        hits = metrics["cache_hits"]
        misses = metrics["cache_misses"]
        if hits + misses > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                [hits, misses],
                labels=["Hits", "Misses"],
                autopct="%1.1f%%",
                startangle=140,
                colors=["#4CAF50", "#FFC107"],
            )
            ax.set_title("Hit / Miss Ratio")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No queries processed yet.")

    st.caption(f"Last updated: {pd.Timestamp.now()}")
else:
    st.warning("Could not reach the API. Make sure the server is running on http://localhost:8000")

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(3)
    st.rerun()
