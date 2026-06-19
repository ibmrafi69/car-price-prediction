import gradio as gr
import pandas as pd
import pandas as pd
import pickle

#load model
with open("R:\Machine learning\car_price_project\model.pkl", "rb") as file:
    model=pickle.load(file)
    
#main logic
def predict(symboling,fueltype,aspiration,doornumber,carbody,
       drivewheel,enginelocation,wheelbase,carlength,carwidth,
       carheight,curbweight,enginetype,cylindernumber,enginesize,
       fuelsystem,boreratio,stroke,compressionratio,horsepower,
       peakrpm,citympg,highwaympg,Brand):
    
    inputs_df = pd.DataFrame({
        "symboling": [symboling],
        "fueltype": [fueltype],
        "aspiration": [aspiration],
        "doornumber": [doornumber],
        "carbody": [carbody],
        "drivewheel": [drivewheel],
        "enginelocation": [enginelocation],
        "wheelbase": [wheelbase],
        "carlength": [carlength],
        "carwidth": [carwidth],
        "carheight": [carheight],
        "curbweight": [curbweight],
        "enginetype": [enginetype],
        "cylindernumber": [cylindernumber],
        "enginesize": [enginesize],
        "fuelsystem": [fuelsystem],
        "boreratio": [boreratio],
        "stroke": [stroke],
        "compressionratio": [compressionratio],
        "horsepower": [horsepower],
        "peakrpm": [peakrpm],
        "citympg": [citympg],
        "highwaympg": [highwaympg],
        "Brand": [Brand]
    })
    #prediction
    prediction=model.predict(inputs_df)[0]
    return prediction
#inputs
inputs = [
    gr.Slider(-2,3,step=1,label="Symboling"),
    gr.Dropdown(["gas", "diesel"], label="Fuel Type"),
    gr.Dropdown(["std", "turbo"], label="Aspiration"),
    gr.Dropdown(["two", "four"], label="Door Number"),
    gr.Dropdown(
        ["sedan", "hatchback", "wagon", "hardtop", "convertible"],
        label="Car Body"
    ),
    gr.Dropdown(
        ["fwd", "rwd", "4wd"],
        label="Drive Wheel"
    ),
    gr.Dropdown(
        ["front", "rear"],
        label="Engine Location"
    ),
    gr.Number(label="Wheel Base"),
    gr.Number(label="Car Length"),
    gr.Number(label="Car Width"),
    gr.Number(label="Car Height"),
    gr.Number(label="Curb Weight"),

    gr.Dropdown(
        ["ohc", "ohcf", "ohcv", "dohc", "l", "rotor"],
        label="Engine Type"
    ),
    gr.Dropdown(
        ["two", "three", "four", "five", "six", "eight", "twelve"],
        label="Cylinder Number"
    ),
    gr.Number(label="Engine Size"),
    gr.Dropdown(
        ["mpfi", "2bbl", "1bbl", "idi", "spdi", "spfi"],
        label="Fuel System"
    ),
    gr.Number(label="Bore Ratio"),
    gr.Number(label="Stroke"),
    gr.Number(label="Compression Ratio"),
    gr.Number(label="Horse Power"),
    gr.Number(label="Peak RPM"),
    gr.Number(label="City MPG"),
    gr.Number(label="Highway MPG"),
    gr.Textbox(label="Brand")
]
#interface
app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Price"),

    title="🚗 Car Price Prediction",
    description="Enter car details and click Submit to predict the car price.",

    theme=gr.themes.Soft(),

    submit_btn="Predict Price",
    clear_btn="Clear",

    article="""
    ### About
    This machine learning model predicts car prices based on
    brand, year, mileage, fuel type, and other features.
    """
)

#app launch
app.launch(share=True)
