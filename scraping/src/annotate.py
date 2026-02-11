import os
import json
import numpy as np
from PIL import Image, ImageDraw

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback, no_update, ctx

# =========================
# CONFIGURATION
# =========================
IMAGES_DIR = "/home/vince/ENSTA/4A/projet/scraping/images_a_annoter"
MASKS_DIR = "/home/vince/ENSTA/4A/projet/scraping/masks_produits"

os.makedirs(MASKS_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

image_files = sorted(
    [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
)

if len(image_files) == 0:
    raise RuntimeError("No images found")

# =========================
# HELPERS
# =========================
def load_image(path):
    return np.array(Image.open(path))

def shapes_to_mask(shapes, img_shape):
    h, w = img_shape[:2]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in shapes:
        if shape["type"] == "path":
            path = shape["path"]
            points = []
            for token in path.replace("M", "").replace("Z", "").split("L"):
                parts = token.strip().split(",")
                if len(parts) == 2:
                    x, y = parts
                    points.append((float(x), float(y)))
            if len(points) > 1:
                draw.polygon(points, fill=255)

        elif shape["type"] == "rect":
            draw.rectangle(
                [shape["x0"], shape["y0"], shape["x1"], shape["y1"]],
                fill=255,
            )

        elif shape["type"] == "circle":
            draw.ellipse(
                [shape["x0"], shape["y0"], shape["x1"], shape["y1"]],
                fill=255,
            )

    return np.array(mask)

def make_figure(img, shapes=None):
    fig = px.imshow(img)

    fig.update_layout(
        dragmode="drawclosedpath", # Mode dessin par défaut
        shapes=shapes or [],
        newshape=dict(
            line=dict(color="yellow", width=2),
            fillcolor="rgba(255,255,0,0.3)",
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    # Configuration des axes pour permettre le zoom fluide
    fig.update_xaxes(visible=False, fixedrange=False)
    fig.update_yaxes(visible=False, fixedrange=False)
    
    return fig

# =========================
# APP INIT
# =========================
app = Dash(__name__)

config = {
    "scrollZoom": True,        # ✅ ACTIVE LE ZOOM AVEC LA MOLETTE
    "modeBarButtonsToAdd": [
        "drawopenpath",
        "drawclosedpath",
        "drawrect",
        "drawcircle",
        "eraseshape",
        "pan2d",               # Ajout de l'outil "pan" pour se déplacer une fois zoomé
    ],
    "displayModeBar": True,
    "displaylogo": False,
}

# =========================
# LAYOUT
# =========================
app.layout = html.Div(
    style={"height": "100vh", "margin": "0", "padding": "0", "overflow": "hidden"},
    children=[
        dcc.Store(id="image-index", data=0),
        dcc.Store(id="shapes-store", data=[]),
        dcc.Store(id="done", data=False),

        html.Div(
            id="main-ui",
            style={
                "display": "flex", 
                "flexDirection": "row", 
                "height": "100vh", 
                "width": "100vw"
            },
            children=[
                
                # ZONE DE GAUCHE : IMAGE
                html.Div(
                    style={"flex": "1", "height": "100%", "backgroundColor": "#1e1e1e"},
                    children=[
                        dcc.Graph(
                            id="fig-image",
                            figure=make_figure(load_image(os.path.join(IMAGES_DIR, image_files[0]))),
                            config=config,
                            style={"height": "100%", "width": "100%"},
                            responsive=True,
                        ),
                    ]
                ),

                # ZONE DE DROITE : CONTROLES
                html.Div(
                    style={
                        "width": "300px",
                        "padding": "20px",
                        "display": "flex",
                        "flexDirection": "column",
                        "borderLeft": "1px solid #ccc",
                        "backgroundColor": "#f8f9fa"
                    },
                    children=[
                        html.H4("Annotation"),
                        html.Div(id="title", style={"fontWeight": "bold", "marginBottom": "20px"}),
                        
                        html.Hr(),
                        
                        html.Button(
                            "Undo (Ctrl+Z)", 
                            id="undo", 
                            n_clicks=0, 
                            style={"marginBottom": "10px", "padding": "10px", "cursor": "pointer"}
                        ),
                        html.Button(
                            "Save & Next (Enter)", 
                            id="save-next", 
                            n_clicks=0, 
                            style={
                                "padding": "15px", 
                                "backgroundColor": "#28a745", 
                                "color": "white", 
                                "border": "none",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "fontWeight": "bold"
                            }
                        )
                        
                    ],
                ),
            ],
        ),

        html.Div(
            id="end-message",
            style={"display": "none", "textAlign": "center", "paddingTop": "100px", "height": "100vh"},
            children=html.H2("Toutes les images ont été annotées !"),
        ),
    ]
)

# =========================
# RACCOURCIS CLAVIER (JS)
# =========================
app.clientside_callback(
    """
    function(id) {
        document.addEventListener('keydown', function(event) {
            if (event.ctrlKey && event.key.toLowerCase() === 'z') {
                event.preventDefault();
                const btnUndo = document.getElementById('undo');
                if (btnUndo) btnUndo.click();
            }
            if (event.key === 'Enter') {
                event.preventDefault();
                const btnNext = document.getElementById('save-next');
                if (btnNext) btnNext.click();
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("main-ui", "id"), 
    Input("main-ui", "id"),
)

# =========================
# CALLBACKS PYTHON
# =========================
@callback(
    Output("title", "children"),
    Input("image-index", "data"),
)
def update_title(idx):
    return f"Fichier : {image_files[idx]}"

@callback(
    Output("fig-image", "figure"),
    Output("image-index", "data"),
    Output("shapes-store", "data"),
    Output("done", "data"),
    Input("fig-image", "relayoutData"),
    Input("undo", "n_clicks"),
    Input("save-next", "n_clicks"),
    State("shapes-store", "data"),
    State("image-index", "data"),
    State("done", "data"),
    prevent_initial_call=True,
)
def handle_all(relayout_data, undo_clicks, save_clicks, shapes, idx, done):
    if done: return no_update, idx, shapes, True

    trigger = ctx.triggered_id
    img_path = os.path.join(IMAGES_DIR, image_files[idx])
    img = load_image(img_path)

    # Si on dessine ou si on zoom/déplace, on met à jour les formes
    if trigger == "fig-image" and relayout_data:
        if "shapes" in relayout_data:
            return no_update, idx, relayout_data["shapes"], False
        return no_update, idx, shapes, False

    if trigger == "undo":
        if not shapes: return no_update, idx, shapes, False
        shapes = shapes[:-1]
        return make_figure(img, shapes), idx, shapes, False

    if trigger == "save-next":
        mask = shapes_to_mask(shapes, img.shape) if shapes else np.zeros(img.shape[:2], dtype=np.uint8)
        mask_name = os.path.splitext(image_files[idx])[0] + ".png"
        Image.fromarray(mask).save(os.path.join(MASKS_DIR, mask_name))

        next_idx = idx + 1
        if next_idx >= len(image_files):
            return no_update, idx, [], True

        next_img = load_image(os.path.join(IMAGES_DIR, image_files[next_idx]))
        return make_figure(next_img), next_idx, [], False

    return no_update, idx, shapes, False

@callback(
    Output("main-ui", "style"),
    Output("end-message", "style"),
    Input("done", "data"),
)
def toggle_ui(done):
    if done: return {"display": "none"}, {"display": "block"}
    return {"display": "flex", "flexDirection": "row", "height": "100vh", "width": "100vw"}, {"display": "none"}

if __name__ == "__main__":
    app.run(debug=True)