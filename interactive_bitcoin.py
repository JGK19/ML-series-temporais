import pickle
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ======================
# LOAD DATA
# ======================

with open("df.pkl", "rb") as f:
    df = pickle.load(f)

with open("hip.pkl", "rb") as f:
    hip = pickle.load(f)

with open("alphas.pkl", "rb") as f:
    alphas = pickle.load(f)

df["date"] = pd.to_datetime(df["Timestamp"], unit="s")

Y = df["Close"]

# derivada
df["grad"] = np.gradient(Y)

# integral total
df["y_cum"] = np.cumsum(Y)

# features calculadas antes
df["y_cum_50"] = hip["Y_cum_50"]
df["alpha_50"] = alphas["alpha_50"]

# ======================
# APP
# ======================

app = dash.Dash(__name__)

years = sorted(df["date"].dt.year.unique())

app.layout = html.Div([

    html.H1("Bitcoin Feature Explorer"),

    html.H3("Selecionar intervalo antes de carregar dados"),

    html.Label("Ano"),
    dcc.Dropdown(years, id="year"),

    html.Label("Semestre"),
    dcc.Dropdown(
        [
            {"label": "Ano inteiro", "value": "all"},
            {"label": "1º semestre", "value": 1},
            {"label": "2º semestre", "value": 2}
        ],
        value="all",
        id="semester"
    ),

    html.Label("Mês"),
    dcc.Input(id="month", type="number", min=1, max=12),

    html.Label("Dia"),
    dcc.Input(id="day", type="number", min=1, max=31),

    html.Button("Carregar intervalo", id="load"),

    html.Hr(),

    html.Div(id="graphs"),

    html.H2("Selecionar ponto"),

    dcc.Input(id="index-input", type="number"),

    html.Div(id="values")

])

# ======================
# LOAD INTERVAL
# ======================

@app.callback(
    Output("graphs", "children"),
    Input("load", "n_clicks"),
    State("year", "value"),
    State("semester", "value"),
    State("month", "value"),
    State("day", "value")
)
def load_interval(n, year, semester, month, day):

    if n is None or year is None:
        return ""

    dff = df[df["date"].dt.year == year]

    if semester == 1:
        dff = dff[dff["date"].dt.month <= 6]

    if semester == 2:
        dff = dff[dff["date"].dt.month > 6]

    if month:
        dff = dff[dff["date"].dt.month == month]

    if day:
        dff = dff[dff["date"].dt.day == day]

    fig_price = go.Figure(go.Scatter(x=dff["date"], y=dff["Close"], mode="lines"))
    fig_price.update_layout(title="Preço (Close)")

    fig_grad = go.Figure(go.Scatter(x=dff["date"], y=dff["grad"], mode="lines"))
    fig_grad.update_layout(title="Gradiente do preço")

    fig_cum = go.Figure(go.Scatter(x=dff["date"], y=dff["y_cum"], mode="lines"))
    fig_cum.update_layout(title="Integral acumulada (y_cum)")

    fig_cum50 = go.Figure(go.Scatter(x=dff["date"], y=dff["y_cum_50"], mode="lines"))
    fig_cum50.update_layout(title="Integral janela 50 dias (y_cum_50)")

    fig_alpha = go.Figure(go.Scatter(x=dff["date"], y=dff["alpha_50"], mode="lines"))
    fig_alpha.update_layout(title="Alpha 50 = y_cum_50 / grad")

    return html.Div([

        dcc.Store(id="current-data", data=dff.to_dict("records")),

        dcc.Graph(id="price", figure=fig_price),
        dcc.Graph(id="grad", figure=fig_grad),
        dcc.Graph(id="cum", figure=fig_cum),
        dcc.Graph(id="cum50", figure=fig_cum50),
        dcc.Graph(id="alpha", figure=fig_alpha)

    ])

# ======================
# SHOW VALUES
# ======================

@app.callback(
    Output("values", "children"),

    Input("price", "clickData"),
    Input("grad", "clickData"),
    Input("cum", "clickData"),
    Input("cum50", "clickData"),
    Input("alpha", "clickData"),

    Input("index-input", "value"),

    State("current-data", "data")
)
def show_values(p, g, c, c50, a, idx, data):

    if data is None:
        return ""

    dff = pd.DataFrame(data)

    click = None

    for item in [p, g, c, c50, a]:
        if item is not None:
            click = item
            break

    if click:
        idx = click["points"][0]["pointIndex"]

    if idx is None or idx >= len(dff):
        return ""

    row = dff.iloc[idx]

    price = row["Close"]
    grad = row["grad"]
    integral = row["y_cum_50"]
    alpha = row["alpha_50"]

    calc_alpha = integral / grad if grad != 0 else np.nan

    return html.Div([

        html.H3(f"Ponto {idx}"),
        html.P(f"Data: {row['date']}"),

        html.H4("Valores"),

        html.P(f"Preço = {price}"),
        html.P(f"Gradiente = {grad}"),
        html.P(f"Integral acumulada = {row['y_cum']}"),
        html.P(f"Integral 50 dias = {integral}"),
        html.P(f"Alpha salvo = {alpha}"),

        html.H4("Verificação"),

        html.P("alpha = integral / grad"),
        html.P(f"{integral} / {grad} = {calc_alpha}")

    ])

# ======================
# RUN
# ======================

if __name__ == "__main__":
    app.run(debug=True)