import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
import math
import matplotlib.pyplot as plt

def get_data():
	data = pd.read_csv("https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_Houses_dataset.csv")
	data = data[['GrLivArea','BedroomAbvGr','KitchenAbvGr', 'OverallCond','RoofSurface','GarageFinish','CentralAir','ChimneyStyle','MoSold','SalePrice']].copy()
	return data


def clean_data(data):
	data = data.drop_duplicates() # Remove duplicates
	data.GarageFinish.replace(np.nan, "NoGarage", inplace=True)
	data.RoofSurface.replace(np.nan, data.RoofSurface.mean())
	data = data.dropna(subset=['RoofSurface'])
	data.drop(columns='ChimneyStyle', inplace=True) 
	return data

def scale_data(data):
	scaler = MinMaxScaler()
	scaler.fit(data[['KitchenAbvGr']])
	scaler.transform(data[['KitchenAbvGr']])
	data.KitchenAbvGr = scaler.transform(data[['KitchenAbvGr']])
	ohe = OneHotEncoder(sparse = False)
	ohe.fit(data[['GarageFinish']])
	garage_encoded = ohe.transform(data[['GarageFinish']])
	data["Fin"],data["NoGarage"],data['RFn'], data['Unf'] = garage_encoded.T
	ohe2 = OneHotEncoder(drop='if_binary',sparse = False)
	ohe2.fit(data[['CentralAir']])
	central_air_encoded = ohe2.transform(data[['CentralAir']])
	data["CentralAir"] = central_air_encoded
	return data
