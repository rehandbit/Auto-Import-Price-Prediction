from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = CustomData(
        symboling         = int(request.form.get("symboling")),
        normalized_losses = int(request.form.get("normalized_losses")),
        make              = request.form.get("make"),
        fuel_type         = request.form.get("fuel_type"),
        aspiration        = request.form.get("aspiration"),
        num_of_doors      = int(request.form.get("num_of_doors")),
        body_style        = request.form.get("body_style"),
        drive_wheels      = request.form.get("drive_wheels"),
        engine_location   = request.form.get("engine_location"),
        wheel_base        = float(request.form.get("wheel_base")),
        length            = float(request.form.get("length")),
        width             = float(request.form.get("width")),
        height            = float(request.form.get("height")),
        curb_weight       = int(request.form.get("curb_weight")),
        engine_type       = request.form.get("engine_type"),
        num_of_cylinders  = int(request.form.get("num_of_cylinders")),
        engine_size       = int(request.form.get("engine_size")),
        fuel_system       = request.form.get("fuel_system"),
        bore              = float(request.form.get("bore")),
        stroke            = float(request.form.get("stroke")),
        compression_ratio = float(request.form.get("compression_ratio")),
        horsepower        = int(request.form.get("horsepower")),
        peak_rpm          = int(request.form.get("peak_rpm")),
        city_mpg          = int(request.form.get("city_mpg")),
        highway_mpg       = int(request.form.get("highway_mpg")),
    )

    df          = data.get_data_as_dataframe()
    pipeline    = PredictionPipeline()
    price       = pipeline.predict(df)

    return render_template('index.html', prediction = f'Predicted Price: $ {price[0]:.2f}')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))

    app.run(host='0.0.0.0' ,debug=False)