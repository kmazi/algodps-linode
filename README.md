### Description

A flask app for training data sources with xgboost for sales prediction.


### Production app on Linode
[Linode](http://45.33.127.194/)


### To work with the app via linode
> - SSH into the server by opening a terminal and typing this  ssh ####################
> - If asked for a password type this  ###############
> - cd into linode-sales-training-setup
>- activate the virtualenv there by typing source linode_setup/bin/activate
>- To pull in recent changes, just do `git pull`



### To work with the app locally on your system
You need to have python 3.7 installed on your system, you can use pyenv to install various python versions, follow the instruction here [RealPython](https://realpython.com/intro-to-pyenv/)

* Clone this repository by running this git command `git clone https://user_name@bitbucket.org/algostacks/linode-sales-training-setup.git`. Replace the username with your bitbucket username.

* [Create a virtual environment and activate it](https://docs.python.org/3/tutorial/venv.html).
* Run `pip install -r requirements.txt` to install  packages required.
*  Create a .env file in your root directory and provide the environment variables as required by the .env_sample.
* Open a terminal and type `flask run`
* Open another terminal  and type `celery -A celery_worker worker -l info`  to start celery app 









