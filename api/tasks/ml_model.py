"""
Module that handles data operations such as merging different datasets,
apply feature engineering , file validation and
trains the xgboost model on the data as an asychronous task using celery
"""


import json
import mimetypes
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from celery.schedules import crontab
from sklearn.preprocessing import LabelEncoder
from werkzeug.datastructures import FileStorage

from celery_worker import celery
from config import VALID_FILE_TYPES, basedir


def _validate(data):
    if mimetypes.MimeTypes().guess_type(data)[0] == "text/csv":
        return pd.read_csv(data)
    elif (
        mimetypes.MimeTypes().guess_type(data)[0]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        return pd.read_excel(data)
    elif mimetypes.MimeTypes().guess_type(data)[0] == "application/vnd.ms-excel":
        return pd.read_excel(data)
    else:
        raise ("invalid file format")



def read_file(value):
    """read incoming file.
    """
    try:
        file = pd.read_csv(value)
    except (UnicodeDecodeError):
        try:
            file = pd.read_excel(value)
        except (UnicodeDecodeError):
            return "Invalid file format"


def merge_datasets(orders, transaction, products):

    new_dataset = orders.merge(
        transaction,
        left_on="id",
        right_on="order_id",
        suffixes=("_orders", "_transaction"),
        sort=False,
    ).merge(products, left_on="product_id", right_on="id", sort=False)
    new_dataset = new_dataset.rename(
        columns={"created_on": "date", "total_cost": "sales"}
    )

    return new_dataset[
        [
            "date",
            "customer_id",
            "country",
            "city",
            "state",
            "order_id",
            "product_id",
            "category",
            "sales",
            "quantity",
            "cost_price",
        ]
    ]


def feature_engineering(data):

    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").sort_index().reset_index()
    # extract some features from date column
    data["month"] = data.date.dt.month
    data["year"] = data.date.dt.year
    data["day"] = data.date.dt.day
    data["weekofyear"] = data.date.dt.weekofyear

    label_encoder = LabelEncoder()
    data["city"] = label_encoder.fit_transform(data["city"])
    data["category"] = label_encoder.fit_transform(data["category"])
    data["country"] = label_encoder.fit_transform(data["country"])
    data["state"] = label_encoder.fit_transform(data["state"])

    return data


@celery.task(name="sales-training")
def xgboost_model(ords, trans, prods, company_name):
    """
    ===================================
    trans, prods and ords are dataframes (csv or excel) for the machine
    learning
    """
    ords = pd.read_csv(ords)
    trans = pd.read_csv(trans)
    prods = pd.read_csv(prods)

    data = merge_datasets(ords, trans, prods)
    data = feature_engineering(data)

    train_data = data[
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
    test_data = data["sales"]

    dtrain = xgb.DMatrix(train_data, test_data)

    """setting up parameters"""

    params = {
        "objective": "reg:linear",  # for linear regression
        "booster": "gbtree",  # use tree based models
        "eta": 0.03,  # learning rate
        "max_depth": 10,  # maximum depth of a tree
        "subsample": 0.9,  # Subsample ratio of the training instances
        "colsample_bytree": 0.7,  # Subsample ratio of columns when constructing each tree
        "silent": 1,  # silent mode
        "seed": 10,  # Random number seed
    }
    num_boost_round = 300
    model = xgb.train(params, dtrain, num_boost_round)

    # save model and pickle it to a folder
    filename = f"pickled_model_{company_name}"
    model_folder = os.path.join(basedir, "models", filename)
    pickled_model = open(model_folder, "wb")
    pickle.dump(model, pickled_model)
    pickled_model.close()

    return {"task": "sales training completed"}


def make_new_dataframe(
    start_date,
    end_date,
    product_name,
    product_id,
    category,
    city,
    state,
    country,
    price,
):
    data = {"date": pd.date_range(start=start_date, end=end_date)}
    data = pd.DataFrame(data=data)
    data["name"] = product_name
    data["product_id"] = product_id
    data["category"] = category
    data["city"] = city
    data["state"] = state
    data["country"] = country
    data["price"] = price
    return data
