import os
import pickle

import numpy as np
import pandas as pd
import redis
import xgboost as xgb
import yaml
from flask_restful import Resource, fields, marshal_with, reqparse

from api import api
from api.tasks.ml_model import (feature_engineering, make_new_dataframe,
                                xgboost_model)
from config import basedir
from log import logger

# from webargs.fields import Int, Str
# from webargs.flaskparser import use_kwargs


log = logger(__name__)

message_fields = {   
        'company_name': fields.String,     
        'start_date': fields.String,   
        'end_date': fields.String,  
        'category': fields.String,    
        'city': fields.String,   
        'state': fields.String,    
        'country': fields.String,
        'price': fields.Integer,   
        'product_id': fields.Integer,    
        'product_name': fields.String
        }
        


class DefaultResource(Resource):
    """Handle default route."""

    def get(self):
        """Get request for home page or response."""
        return {
            "status": "success",
            "data": {"msg": "Welcome to Linode Sales Training System- ADPS"},
        }


class MakePredictions(Resource):
    """Make new predictions"""
    # @marshal_with(message_fields)
    def post(self):
        parser = reqparse.RequestParser()        
        parser.add_argument('company_name', type=str, required=True,help='company name cannot be blank!')       
        parser.add_argument('start_date', type=str, required=True,help='start date cannot be blank!')        
        parser.add_argument('end_date', type=str, required=True,help='end date  cannot be blank!')  
        parser.add_argument('category', type=str, required=True,help='category cannot be blank!')       
        parser.add_argument('city', type=str, required=True,help='city cannot be blank!')        
        parser.add_argument('state', type=str, required=True,help='state cannot be blank!')   
        parser.add_argument('country', type=str, required=True,help='country cannot be blank!')        
        parser.add_argument('price', type=int, required=True,help='price cannot be blank!')   
        parser.add_argument('product_id', type=int, required=True,help='product_id cannot be blank!')  
        parser.add_argument('product_name', type=str, required=False,help='product name cannot be blank!')        
        args = parser.parse_args()

        start_date = args['start_date']
        company_name = args['company_name']
        end_date = args['end_date']
        category = args['category']
        city = args['city']
        state = args['state']
        country = args['country']
        price = args['price']
        product_id = args['product_id']
        product_name = args['product_name']

        # create a new dataset and make predictions
        datasets = make_new_dataframe(
            start_date,
            end_date,
            product_name,
            product_id,
            category,
            city,
            state,
            country,
            price,
        )

        """Write a function that calls the files uploaded to AWS S3 bucket.
           But for now we would be working with local files until such 
           options are available.
           If the company_name is passed in json, it is used to filter out the 
           the datasets from AWS S3 bucket
        """

        def load_files_from_aws(company_name):
            pass

        ords = os.path.join(basedir, "data/orders.csv")
        trans = os.path.join(basedir, "data/transactions.csv")
        prods = os.path.join(basedir, "data/products.csv")

        # background task for model_training
        background_task = xgboost_model.delay(ords, trans, prods, company_name)

        datasets = make_new_dataframe(
            start_date,
            end_date,
            product_name,
            product_id,
            category,
            city,
            state,
            country,
            price,
        )
        train_data = datasets.copy()
        # apply feature engineering and make predictions
        train_data = feature_engineering(datasets)
        train_data = train_data[
            [
                "year",
                "day",
                "weekofyear",
                "month",
                "product_id",
                "city",
                "category",
                "country",
                "state",
            ]
        ]

        train_data = xgb.DMatrix(train_data)

        #load pickled model for company 
        try:
            filename = os.path.join(basedir, "models", f"pickled_model_{company_name}")
            model = open(filename, "rb")
            model = pickle.load(model)
        except (FileNotFoundError):
            raise ("model has not been trained, retry requests in few minute time")

        results = model.predict(train_data)
        datasets["forecasted_sales"] = results
        datasets["quantity_predicted"] = round(
            datasets["forecasted_sales"] / datasets["price"]
        )

        # Datetime are not JSON serializabe, so we change dates to string
        datasets["date"] = datasets["date"].dt.strftime("%Y-%b-%d")
        results = datasets.to_dict(orient="list")
        

        if results:
            log.info("succesfully made new predictions")
            return {"status": "success", "data": results}, 200

        log.warning("failed to make new predictions ,sales model is still running,retry in few minute time")
        return {"status": "failed"}, 400


api.add_resource(DefaultResource, "/", endpoint="home")
api.add_resource(MakePredictions, "/prediction/", endpoint="prediction")
