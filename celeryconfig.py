from __future__ import absolute_import, unicode_literals

import os

import pandas as pd
from celery.schedules import crontab

from config import basedir

# call the file locally, in the future it would come from aws S3 bucket
# ords = os.path.join(basedir, "data/orders.csv")
# trans = os.path.join(basedir, "data/transactions.csv")
# prods = os.path.join(basedir, "data/products.csv")

broker_url = "redis://localhost:6379/0"
result_backend = "redis://localhost:6379/0"

# List of modules to import when the Celery worker starts.
imports = ("api.tasks.ml_model",)

# # using serializer name
# accept_content = ["pickle"]

# task_serializer = ["pickle"]
# result_serializer = ["pickle"]


timezone = "Africa/Lagos"

result_expires = 3600

 #we set this background, it trains every 5 hours
# beat_schedule = {
#     #Executes sales training every minutes.
#     "execute-training-every-minute": {
#         "task": 'api.tasks.ml_model.xgboost_model',
#         "schedule": crontab(),  #Execute every five hours
#         "args": (ords, trans, prods),
#     },
    
# }
