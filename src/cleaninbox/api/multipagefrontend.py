import streamlit as st

#icon list: https://materialdesignicons.com/ ChartBellCurve
home_page = st.Page("welcome.py",title="Cleaninbox", icon=":material/home_app_logo:",default=True)
predict_page = st.Page(
    "prediction/predict.py", title="Predict", icon=":material/computer:", default=False
)

data_drift_page = st.Page(
    "monitoring/data_drifting.py", title="Monitoring", icon=":material/bar_chart:", default=False
)

pg = st.navigation(
    {
        "Home": [home_page],
        "Predict": [predict_page],
        "Monitoring": [data_drift_page],
        #"Tools": [search, history],
    }
)
pg.run()