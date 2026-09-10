import sys
import numpy as np
import pandas as pd
from src.logger import logger
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        self.model_path = 'artifacts/model/model.pkl'
        self.scaler_path = 'artifacts/preprocessor/scaler.pkl'
        self.ohe_path = 'artifacts/preprocessor/ohe.pkl'


    def predict(self, data):
        try:
            logger.info('Prediction has started -\/')

            #load 
            model  =load_object(self.model_path)
            scaler =load_object(self.scaler_path)
            ohe    =load_object(self.ohe_path)

            #numeric feature
            numeric_feature      = data.select_dtypes(include = [np.number]).columns.tolist()
            categoorical_feature = data.select_dtypes(exclude = [np.number]).columns.tolist()

            scaled  = scaler.transform(data[numeric_feature])
            encoded = ohe.transform(data[categoorical_feature])
            final   = np.hstack([scaled, encoded])

            price = model.predict(final)
            logger.info(f'Prediction done : {price}')

            return price

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, symboling, normalized_losses, make, fuel_type, aspiration, num_of_doors,
                 body_style, drive_wheels, engine_location, wheel_base, length, width, height,
                 curb_weight, engine_type, num_of_cylinders, engine_size, fuel_system, bore, stroke,
                 compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg):
        
        self.symboling         = symboling
        self.normalized_losses = normalized_losses
        self.make              = make
        self.fuel_type         = fuel_type
        self.aspiration        = aspiration
        self.num_of_doors      = num_of_doors
        self.body_style        = body_style
        self.drive_wheels      = drive_wheels
        self.engine_location   = engine_location
        self.wheel_base        = wheel_base
        self.length            = length
        self.width             = width
        self.height            = height
        self.curb_weight       = curb_weight
        self.engine_type       = engine_type
        self.num_of_cylinders  = num_of_cylinders
        self.engine_size       = engine_size
        self.fuel_system       = fuel_system
        self.bore              = bore
        self.stroke            = stroke
        self.compression_ratio = compression_ratio
        self.horsepower        = horsepower
        self.peak_rpm          = peak_rpm
        self.city_mpg          = city_mpg
        self.highway_mpg       = highway_mpg

    def get_data_as_dataframe(self):
        data = {
            'symboling'         : [self.symboling],
            'normalized_losses' : [self.normalized_losses],
            'make'              : [self.make],
            'fuel_type'         : [self.fuel_type],
            'aspiration'        : [self.aspiration],
            'num_of_doors'      : [self.num_of_doors],
            'body_style'        : [self.body_style],
            'drive_wheels'      : [self.drive_wheels],
            'engine_location'   : [self.engine_location],
            'wheel_base'        : [self.wheel_base],
            'length'            : [self.length],
            'width'             : [self.width],
            'height'            : [self.height],
            'curb_weight'       : [self.curb_weight],
            'engine_type'       : [self.engine_type],
            'num_of_cylinders'  : [self.num_of_cylinders],
            'engine_size'       : [self.engine_size],
            'fuel_system'       : [self.fuel_system],
            'bore'              : [self.bore],
            'stroke'            : [self.stroke],
            'compression_ratio' : [self.compression_ratio],
            'horsepower'        : [self.horsepower],
            'peak_rpm'          : [self.peak_rpm],
            'city_mpg'          : [self.city_mpg],
            'highway_mpg'       : [self.highway_mpg],
        }
        return pd.DataFrame(data)

if __name__ == '__main__':
    car = CustomData(
        symboling=0, normalized_losses=100, make="toyota",
        fuel_type="gas", aspiration="std", num_of_doors=4,
        body_style="sedan", drive_wheels="fwd",
        engine_location="front", wheel_base=98.0,
        length=170.0, width=65.0, height=54.0,
        curb_weight=2400, engine_type="ohc",
        num_of_cylinders=4, engine_size=110,
        fuel_system="mpfi", bore=3.19, stroke=3.40,
        compression_ratio=10.0, horsepower=102,
        peak_rpm=5500, city_mpg=24, highway_mpg=30
    )

    df = car.get_data_as_dataframe()
    print(df)
    pipeline = PredictionPipeline()
    price = pipeline.predict(df)
    print(f'\n Predicted Price : $ {round(price[0],0)}')