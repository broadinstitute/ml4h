ML4H has a GAE url for MLflow
 

The UI needs a username and password to access
These are stored in the GCP Secret Manager


You can use this from your local machine or from your GCP VM

To start, you will want to use the .env.template to make your own .env file
and then update your python script with this path

you will also need to log in using 
gcloud auth application-default login 

note the path that your auth json is stored at and put that into your .env file as well




Setup and Maintanence

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
