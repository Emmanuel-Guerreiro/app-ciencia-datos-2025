import altair as alt
from numpy.random import default_rng as rng
import streamlit as st
import numpy as np
import pandas as pd

main_page = st.Page("main.py", title="Main Page", icon="🎈")
plots_page = st.Page("plots.py", title="Plots Page", icon="🎈")
model_page = st.Page("model.py", title="Model Page", icon="🎈")
pg = st.navigation([main_page, plots_page, model_page])
pg.run()
