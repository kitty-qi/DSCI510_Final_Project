import plotly.express as px
import pandas as pd
# https://plotly.com/python/time-series/
df = pd.read_csv('covid_2021_losangeles_data.csv')

fig = px.line(df, x='date', y=['cases',"deaths"], title='Time Series with Range Slider and Selectors')



fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()