#!python3.8

from fastcore.all import *
from fastai.vision.all import *
import streamlit as st
from streamlit_image_select import image_select


## Import FastAI models (inculding related methods)
def get_label_reg(fname) -> float:
    scr, xrr = re.findall(r'^SCR=(\d+)_XRR=(\d+).+.png$', fname.name)[0]
    return scr

def get_label_clf(fname) -> str:
    scr, xrr = re.findall(r'^SCR=(\d+)_XRR=(\d+).+.png$', fname.name)[0]
    label = 'null'
    scr = float(scr)
    if scr<10:
        label = 'Very Weak'
    elif scr<=20:
        label = 'Weak'
    else:
        label = 'Strong'
    return label

learn_reg = load_learner("models/reg_learn.pkl")
learn_clf = load_learner("models/clf_learn.pkl")


## Prediction methods
def classify_grid(data):
    pred, pred_idx, probs = learn_clf.predict(data)
    return pred, probs[pred_idx]

def predict_grid(data):
    pred, _, _ = learn_reg.predict(data)
    return pred[0]


## App
st.set_page_config("SCR Prediction", ":material/power:")
st.title("Power Grid Classifier! 📈")
bytes_data = None

uploaded_image = st.file_uploader("Input a Power Grid spectrum:")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    st.image(bytes_data, caption="Uploaded image")   

test_image = image_select(
    label="or select a test spectrum from the gallery",
    images=[
        Image.open("data/SCR=4_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=7_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=12_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=16_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=30_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=40_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=60_XRR=3_SpP=30%_SpQ=0%.png"),
        Image.open("data/SCR=80_XRR=3_SpP=30%_SpQ=0%.png"),
    ],
    captions=["SCR=4", "SCR=7", "SCR=12", "SCR=16", "SCR=30", "SCR=40", "SCR=60", "SCR=80"],
    use_container_width=False,
)
if test_image:
    bytes_data = test_image

if bytes_data:
    analysis = st.button("Analyse!")
    if analysis:
        st.header("Prediction")
        label, confidence = classify_grid(bytes_data)
        y_pred = predict_grid(bytes_data)

        col1, col2= st.columns(2)
        col1.metric("Grid Type", f"{label} ({confidence:.02%})", "")
        col2.metric("SCR", f"{y_pred:.01f}", "")