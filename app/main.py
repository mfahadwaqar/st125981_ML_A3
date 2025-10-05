import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import pickle
from custom_models import LinearRegression
from custom_models import LinearRegression, NoRegularization, RidgeRegularization, LassoRegularization

# ==== Features ====
NUMERIC_FEATURES = ["year", "km_driven", "mileage", "engine", "max_power"]
CATEGORICAL_FEATURES = ["brand", "transmission", "owner"]
BRANDS = [
    'Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun',
    'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep',
    'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz',
    'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata',
    'Toyota', 'Volkswagen', 'Volvo'
]

# ==== Load pipelines ====
with open("app/rf_pipeline.pkl", "rb") as f:
    rf_pipeline = pickle.load(f)

with open("app/custom_lr_pipeline.pkl", "rb") as f:
    custom_pipeline = pickle.load(f)

# ==== Helper: Preprocessing & Prediction ====
def preprocess_input(df_input: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    scaler = pipeline["scaler"]
    encoder = pipeline["encoder"]
    num_medians = pipeline["num_medians"]
    cat_modes = pipeline["cat_modes"]

    X_num = df_input[NUMERIC_FEATURES].copy().fillna(num_medians)
    X_cat = df_input[CATEGORICAL_FEATURES].copy().fillna(cat_modes)

    X_num_scaled = pd.DataFrame(
        scaler.transform(X_num), columns=NUMERIC_FEATURES, index=df_input.index
    )
    X_cat_encoded = pd.DataFrame(
        encoder.transform(X_cat),
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df_input.index,
    )
    return pd.concat([X_num_scaled, X_cat_encoded], axis=1)

def predict_with_pipeline(n_clicks, year, km_driven, mileage, engine, max_power, brand, transmission, owner, pipeline):
    if not n_clicks:
        return ""
    if None in (year, km_driven, mileage, engine, max_power, brand, transmission, owner):
        return "Please fill all fields before predicting."

    try:
        # ---- Raw user input as DataFrame ----
        df_input = pd.DataFrame([{
            "year": float(year),
            "km_driven": float(km_driven),
            "mileage": float(mileage),
            "engine": float(engine),
            "max_power": float(max_power),
            "brand": brand,
            "transmission": transmission,
            "owner": owner
        }])

        # ---- Apply same imputations ----
        for col, median in pipeline["num_medians"].items():
            if col in df_input:
                df_input[col] = df_input[col].fillna(median)
        for col, mode in pipeline["cat_modes"].items():
            if col in df_input:
                df_input[col] = df_input[col].fillna(mode)

        # ---- Scale + encode ----
        X_num = pipeline["scaler"].transform(df_input[NUMERIC_FEATURES])
        
        # If polynomial transformer exists, apply it
        if "poly" in pipeline and pipeline["poly"] is not None:
            X_num = pipeline["poly"].transform(X_num)

        # X_cat = pipeline["encoder"].transform(df_input[CATEGORICAL_FEATURES])
        # X_processed = np.hstack([X_num, X_cat])
        # print(X_num.shape)
        # Add bias column (intercept term) if using custom model
        # if isinstance(pipeline["model"], LinearRegression):
            # X_processed = np.hstack([np.ones((X_processed.shape[0], 1)), X_processed])
        
        # ---- Predict ----
        model = pipeline["model"]
        log_price = model.predict(X_num)[0]
        price = float(np.exp(log_price))
        return f"Predicted Selling Price: {price:,.0f}"

    except Exception as e:
        return f"Error during prediction: {e}"

# ==== Shared Input Form ====
def input_form(model_name):
    return html.Div([
        html.H1(f"Car Price Prediction – {model_name}", style={"textAlign": "center"}),
        html.P("Fill in the details below to predict the car's selling price.", style={"textAlign": "center"}),
        html.Div([
            html.Div([
                html.Label("Year"), dcc.Input(id=f"year-{model_name}", type="number"),
                html.Label("Kilometers Driven"), dcc.Input(id=f"km_driven-{model_name}", type="number"),
                html.Label("Mileage (kmpl)"), dcc.Input(id=f"mileage-{model_name}", type="number"),
                html.Label("Engine (CC)"), dcc.Input(id=f"engine-{model_name}", type="number"),
                html.Label("Max Power (bhp)"), dcc.Input(id=f"max_power-{model_name}", type="number"),
            ], style={"display": "grid", "gap": "10px"}),
            html.Div([
                html.Label("Brand"),
                dcc.Dropdown(id=f"brand-{model_name}", options=[{"label": b, "value": b} for b in BRANDS]),
                html.Label("Transmission"),
                dcc.Dropdown(id=f"transmission-{model_name}", options=[{"label": x, "value": x} for x in ["Manual", "Automatic"]]),
                html.Label("Owner"),
                dcc.Dropdown(id=f"owner-{model_name}", options=[{"label": o, "value": o} for o in ["First Owner","Second Owner","Third Owner","Fourth & Above Owner"]]),
            ], style={"display": "grid", "gap": "10px"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "40px", "maxWidth": "900px", "margin": "0 auto"}),
        html.Div([
            html.Button("Predict Price", id=f"predict-btn-{model_name}", n_clicks=0)
        ], style={"textAlign": "center", "marginTop": "20px"}),
        html.Div(id=f"output-{model_name}", style={"textAlign": "center", "fontSize": 24, "marginTop": "20px"})
    ])

# ==== App Layout with Pages ====
app = dash.Dash(__name__)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Car Price Prediction</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background-color: #121212;
                color: white;
                font-family: Arial, sans-serif;
                height: 100%;
            }
            #_dash-app-content {
                background-color: #121212;
                min-height: 100vh;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style={
    "backgroundColor": "#121212",
    "minHeight": "100vh",
    "color": "white",
    "fontFamily": "Arial, sans-serif",
    "padding": "20px"
})

navbar = html.Div([
    dcc.Link(html.Button("Home", style={
        "backgroundColor": "#333", "color": "white",
        "border": "none", "padding": "10px 20px",
        "borderRadius": "8px", "cursor": "pointer",
        "marginRight": "10px"
    }), href='/'),

    dcc.Link(html.Button("Old Model (A1)", style={
        "backgroundColor": "#007acc", "color": "white",
        "border": "none", "padding": "10px 20px",
        "borderRadius": "8px", "cursor": "pointer",
        "marginRight": "10px"
    }), href='/old'),

    dcc.Link(html.Button("Custom Model (A2)", style={
        "backgroundColor": "#ff5722", "color": "white",
        "border": "none", "padding": "10px 20px",
        "borderRadius": "8px", "cursor": "pointer"
    }), href='/new'),
], style={
    "textAlign": "center",
    "padding": "15px",
    "backgroundColor": "#1e1e1e",
    "position": "sticky",
    "top": "0",
    "zIndex": "1000",
    "marginBottom": "30px"
})

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/old':
        return html.Div([navbar, input_form("old")])
    elif pathname == '/new':
        return html.Div([navbar, input_form("new")])
    else:
        return html.Div([navbar, html.H1("Choose a model to start predicting.", style={"textAlign": "center"})])

# ==== Callbacks for Predictions ====
@app.callback(
    Output("output-old", "children"),
    Input("predict-btn-old", "n_clicks"),
    State("year-old", "value"), State("km_driven-old", "value"),
    State("mileage-old", "value"), State("engine-old", "value"),
    State("max_power-old", "value"), State("brand-old", "value"),
    State("transmission-old", "value"), State("owner-old", "value"))
def predict_old(n_clicks, year, km, mileage, engine, power, brand, trans, owner):
    return predict_with_pipeline(n_clicks, year, km, mileage, engine, power, brand, trans, owner, rf_pipeline)

@app.callback(
    Output("output-new", "children"),
    Input("predict-btn-new", "n_clicks"),
    State("year-new", "value"), State("km_driven-new", "value"),
    State("mileage-new", "value"), State("engine-new", "value"),
    State("max_power-new", "value"), State("brand-new", "value"),
    State("transmission-new", "value"), State("owner-new", "value"))
def predict_new(n_clicks, year, km, mileage, engine, power, brand, trans, owner):
    return predict_with_pipeline(n_clicks, year, km, mileage, engine, power, brand, trans, owner, custom_pipeline)

# ==== Run App ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
