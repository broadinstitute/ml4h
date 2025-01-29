# MLflow Usage and Maintance


## Using MLflow on App Engine
### To see the MLflow UI
ML4H has a GAE url for MLflow

The UI needs a username and password to access
These are stored in the GCP Secret Manager

### To add your experiments to this MLflow instance
You can use this from your local machine or from your GCP VM

To start, you will want to use the .env.template to make your own .env file
cp .env.template .env
and then update your python script with this path

you will also need to log in using 
gcloud auth application-default login 

note the path that your auth json is stored at and put that into your .env file as well


## General Architecture

MLflow is using Cloud SQL as a backend database-- this stores all of the experiment runs that show up in the UI
MLflow is stood up on App Engine. The server is packaged up as a docker image deployed. It is also running on CloudRun (though likely this will be taken down in the future)
MLflow is uses a Google bucket as an artifact store.  



## Setup and Maintanence

The database for this instance is in GCP Cloud SQL
To access the database, navigate to the GCP SQL dashboard, then navigate to the <database name> database 
Next you want to connect by clicking OPEN CLOUD SHELL
Once there you can log in with
gcloud sql connect <username> --user=postgres --quiet

Use the credentials from the GCP secret manager

Once in the database, you can use \d to get a full list of tables
Likely the tables you are interested in will be `runs` and `tags`

In the future, to maximize security, the app can live behind a Identity-Aware Proxy.
Note that this cannot be applied so specific services, so if in the future a different service as added, a different security set-up may be necessary




### For the App Engine setup
The default service account's name is PROJECT_ID@appspot.gserviceaccount.com

Ensure that the service account your app is using to authenticate calls to Cloud SQL has the Cloud SQL Client IAM role.


From the mlflow directory, run:
gcloud app deploy
gcloud sql instances describe broad-ml4cvd-staging-db



(is this even necessary to include?)
gcloud app deploy --image-url=us-central1-docker.pkg.dev/gvs-internal/rc-testing-repo/mlflow-imagine:latest




Update the Mlflow docker (only needed for CloudRun implementation)
docker build -t mlflow-gcp .
docker tag mlflow-gcp us-central1-docker.pkg.dev/gvs-internal/rc-testing-repo/mlflow-imagine:latest

TODO add bit about proxying to be able to see this
