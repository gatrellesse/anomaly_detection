import os
import numpy as np
from PIL import Image, ImageDraw
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback, no_update, ctx

"""
This script provides a web-based Interactive Image Annotation Tool designed to create ground-truth masks
for object detection and anomaly segmentation. Users can draw shapes (polygons, rectangles) on images to
define "Object" areas and "Anomaly" areas, which are then exported as binary PNG masks.

The tool is built using Dash, a productive Python framework for building web applications. Dash leverages
Plotly for interactive graphing, allowing for precise zooming and drawing directly on images, and uses a
reactive callback system to handle complex UI logic (like undoing actions or switching modes) entirely in
Python.
"""

IMAGES_DIR = "/home/vince/ENSTA/4A/projet/scraping/images_a_annoter"
MASKS_OBJ_DIR = "/home/vince/ENSTA/4A/projet/scraping/masks_produits_objets"
MASKS_ANO_DIR = "/home/vince/ENSTA/4A/projet/scraping/masks_produits_anomalies"

for d in [MASKS_OBJ_DIR, MASKS_ANO_DIR]:
    os.makedirs(d, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)])

if len(image_files) == 0:
    raise RuntimeError("No images found")

def load_image(path):
    """Load image as a numpy array."""
    return np.array(Image.open(path))

def shapes_to_mask(shapes, img_shape):
    """Convert Plotly shapes (paths, rects, circles) into a binary 8-bit mask."""
    h, w = img_shape[:2]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in shapes:
        if shape["type"] == "path":
            points = []
            for token in shape["path"].replace("M", "").replace("Z", "").split("L"):
                parts = token.strip().split(",")
                if len(parts) == 2:
                    points.append((float(parts[0]), float(parts[1])))
            if len(points) > 1:
                draw.polygon(points, fill=255)

        elif shape["type"] in ["rect", "circle"]:
            method = draw.rectangle if shape["type"] == "rect" else draw.ellipse
            method([shape["x0"], shape["y0"], shape["x1"], shape["y1"]], fill=255)

    return np.array(mask)

def make_figure(img, shapes_obj, shapes_ano, current_mode, draw_mode, polygon_points):
    """Generate the Plotly figure with image and current annotations."""
    fig = px.imshow(img, binary_string=True)

    draw_color = "yellow" if current_mode == "OBJ" else "#00FF00"
    draw_fill = "rgba(255,255,0,0.3)" if current_mode == "OBJ" else "rgba(0,255,0,0.3)"
    
    # Toggle between drawing mode and navigation (zoom/pan)
    dragmode = "drawclosedpath" if draw_mode == "DRAG" else "zoom"    
    
    fig.update_layout(
        dragmode=dragmode,
        shapes=shapes_obj + shapes_ano,
        newshape=dict(line=dict(color=draw_color, width=2), fillcolor=draw_fill),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        autosize=True
    )

    # Visual feedback for manual click-by-click polygon mode
    if len(polygon_points) > 0:
        xs = [p[0] for p in polygon_points]
        ys = [p[1] for p in polygon_points]

        fig.add_scatter(
            x=xs,
            y=ys,
            mode="markers+lines",
            marker=dict(size=8, color=draw_color),
            line=dict(color=draw_color),
        )

    return fig

app = Dash(__name__)

config = {
    "scrollZoom": True,
    "modeBarButtonsToAdd": ["drawclosedpath", "drawrect", "eraseshape", "pan2d"],
    "displayModeBar": True,
    "displaylogo": False,
}

app.layout = html.Div(style={"height": "100vh", "margin": "0", "backgroundColor": "#1e1e1e", "overflow": "hidden"}, children=[

    dcc.Store(id="image-index", data=0),
    dcc.Store(id="shapes-obj-store", data=[]),
    dcc.Store(id="shapes-ano-store", data=[]),
    dcc.Store(id="mode-store", data="OBJ"),
    dcc.Store(id="draw-mode-store", data="DRAG"),
    dcc.Store(id="polygon-points", data=[]),
    dcc.Store(id="done", data=False),

    html.Div(id="main-ui", style={"display": "flex", "height": "100vh", "width": "100vw"}, children=[

        html.Div(style={"flex": "1", "height": "100vh", "overflow": "hidden"}, children=[
            dcc.Graph(
                id="fig-image",
                config=config,
                style={"height": "100%", "width": "100%"},
                responsive=True
            ),
        ]),

        html.Div(style={
            "width": "320px", "padding": "20px", "backgroundColor": "#f8f9fa",
            "display": "flex", "flexDirection": "column", "zIndex": "10"
        }, children=[

            html.H4("Annotation"),

            html.Div(id="title", style={"fontWeight": "bold", "marginBottom": "10px"}),

            html.Hr(),

            html.Label("Mode actuel :"),

            html.Button(id="btn-mode", n_clicks=0),

            html.Label("Mode dessin :", style={"marginTop": "20px"}),

            html.Button(id="btn-draw-mode", n_clicks=0),

            html.Button("Undo (Ctrl+Z)", id="undo", n_clicks=0),

            html.Div(style={"marginTop": "auto"}, children=[
                html.Button("Save & Next (Enter)", id="save-next", n_clicks=0)
            ])
        ])
    ]),

    html.Div(id="end-message", style={"display": "none", "textAlign": "center", "color": "white", "paddingTop": "100px"}, children=html.H2("Session terminée !"))
])

@callback(
    Output("draw-mode-store","data"),
    Input("btn-draw-mode","n_clicks"),
    State("draw-mode-store","data"),
    prevent_initial_call=True
)
def toggle_draw(n,mode):
    return "CLICK" if mode=="DRAG" else "DRAG"

@callback(
    Output("btn-draw-mode","children"),
    Input("draw-mode-store","data")
)
def update_draw_label(mode):
    return "MODE DESSIN : CLICS SUCCESSIFS" if mode=="CLICK" else "MODE DESSIN : DRAG"

@callback(
    Output("mode-store","data",allow_duplicate=True),
    Input("btn-mode","n_clicks"),
    State("mode-store","data"),
    prevent_initial_call=True
)
def toggle_mode(n,mode):
    return "ANO" if mode=="OBJ" else "OBJ"

@callback(
    [Output("btn-mode","children"),
     Output("btn-mode","style")],
    Input("mode-store","data")
)
def update_button(mode):
    if mode=="ANO":
        return "MODE: ANOMALIE",{"backgroundColor":"#00FF00"}
    return "MODE: OBJET",{"backgroundColor":"yellow"}


@callback(
    Output("fig-image","figure"),
    Output("image-index","data"),
    Output("shapes-obj-store","data"),
    Output("shapes-ano-store","data"),
    Output("polygon-points","data"),
    Output("mode-store","data"),
    Output("done","data"),

    Input("fig-image","relayoutData"),
    Input("fig-image","clickData"),
    Input("undo","n_clicks"),
    Input("save-next","n_clicks"),
    Input("mode-store","data"),
    Input("draw-mode-store","data"),

    State("shapes-obj-store","data"),
    State("shapes-ano-store","data"),
    State("polygon-points","data"),
    State("image-index","data"),
    State("done","data"),
    prevent_initial_call=True
)
def sync(relayout,clickData,undo,save,mode,draw_mode,s_obj,s_ano,poly,idx,done):

    if done:
        return no_update,idx,s_obj,s_ano,poly,mode,True

    trig=ctx.triggered_id
    img_path=os.path.join(IMAGES_DIR,image_files[idx])
    img=load_image(img_path)

    # CLICK MODE
    if trig=="fig-image" and draw_mode=="CLICK" and clickData:

        x=clickData["points"][0]["x"]
        y=clickData["points"][0]["y"]

        if len(poly)>2:
            x0,y0=poly[0]
            if np.hypot(x-x0,y-y0)<10:

                path="M "+" L ".join([f"{p[0]},{p[1]}" for p in poly])+" Z"

                shape=dict(
                    type="path",
                    path=path,
                    line=dict(color="yellow" if mode=="OBJ" else "#00FF00"),
                    fillcolor="rgba(255,255,0,0.3)" if mode=="OBJ" else "rgba(0,255,0,0.3)"
                )

                if mode=="OBJ":
                    s_obj.append(shape)
                else:
                    s_ano.append(shape)

                poly=[]

                return make_figure(img,s_obj,s_ano,mode,draw_mode,poly),idx,s_obj,s_ano,poly,mode,False

        poly.append((x,y))

        return make_figure(img,s_obj,s_ano,mode,draw_mode,poly),idx,s_obj,s_ano,poly,mode,False

    # DRAG MODE
    if trig=="fig-image" and relayout and "shapes" in relayout:

        new_shapes=relayout["shapes"]

        current_obj=[s for s in new_shapes if s.get("line",{}).get("color")=="yellow"]
        current_ano=[s for s in new_shapes if s.get("line",{}).get("color")=="#00FF00"]

        return no_update,idx,current_obj,current_ano,poly,mode,False

    if trig=="undo":

        if draw_mode=="CLICK" and poly:
            poly.pop()
        elif mode=="OBJ" and s_obj:
            s_obj.pop()
        elif mode=="ANO" and s_ano:
            s_ano.pop()

        return make_figure(img,s_obj,s_ano,mode,draw_mode,poly),idx,s_obj,s_ano,poly,mode,False

    if trig=="save-next":

        mask_name=os.path.splitext(image_files[idx])[0]+".png"

        Image.fromarray(shapes_to_mask(s_obj,img.shape)).save(os.path.join(MASKS_OBJ_DIR,mask_name))
        Image.fromarray(shapes_to_mask(s_ano,img.shape)).save(os.path.join(MASKS_ANO_DIR,mask_name))

        if idx+1>=len(image_files):
            return no_update,idx,[],[],[],mode,True

        new_idx=idx+1
        new_img=load_image(os.path.join(IMAGES_DIR,image_files[new_idx]))

        return make_figure(new_img,[],[],mode,draw_mode,[]),new_idx,[],[],[],mode,False

    return make_figure(img,s_obj,s_ano,mode,draw_mode,poly),idx,s_obj,s_ano,poly,mode,False

# =========================
# TITLE
# =========================
@callback(
    Output("title","children"),
    Input("image-index","data")
)
def set_title(idx):
    return f"Image {idx+1}/{len(image_files)}: {image_files[idx]}"

# =========================
# END
# =========================
@callback(
    [Output("main-ui","style"),Output("end-message","style")],
    Input("done","data")
)
def finish(done):
    if done:
        return {"display":"none"},{"display":"block"}
    return {"display":"flex","height":"100vh","width":"100vw"},{"display":"none"}

# =========================
# SHORTCUTS
# =========================
app.clientside_callback(
"""
function(id){
document.addEventListener('keydown',function(e){

if(e.ctrlKey && e.key==='z'){
e.preventDefault()
document.getElementById('undo').click()
}

if(e.key==='Enter'){
e.preventDefault()
document.getElementById('save-next').click()
}

})
return window.dash_clientside.no_update
}
""",
Output("main-ui","id"),
Input("main-ui","id")
)

if __name__=="__main__":
    app.run(debug=True,use_reloader=False)