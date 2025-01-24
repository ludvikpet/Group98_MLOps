import streamlit as st

home_page = st.Page("welcome.py",title="Cleaninbox", icon=":material/home_app_logo:",default=True)
predict_page = st.Page(
    "prediction/predict.py", title="Predict", icon=":material/computer:", default=False
)

data_drift_page = st.Page(
    "monitoring/data_drifting.py", title="Monitoring", icon=":material/bar_chart:", default=False
)

statistics_page = st.Page(
    "monitoring/user_statistics.py", title="Usage statistics", icon=":material/pie_chart:", default=False    
)

metrics_page = st.Page(
    "monitoring/metrics.py", title="Internal Metrics", icon=":material/access_alarm:", default=False    
)

pg = st.navigation(
    {
        "Home": [home_page],
        "Predict": [predict_page],
        "Monitoring": [data_drift_page, metrics_page],
        "Usage statistics": [statistics_page]
        #"Tools": [search, history],
    }
)

pg.run()