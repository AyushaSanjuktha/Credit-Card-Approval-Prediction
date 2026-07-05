"""
IBM Watson Machine Learning Deployment Pipeline
--------------------------------------------------
Deploys the trained fraud-detection model (best_model.pkl) to IBM Watson
Machine Learning so it can be scored via a hosted, scalable REST endpoint.

Prerequisites:
  pip install ibm-watson-machine-learning
  An IBM Cloud account + Watson Machine Learning service instance
  Your API key and deployment space ID from IBM Cloud

Fill in the placeholders below with your own credentials before running.
"""

from ibm_watson_machine_learning import APIClient
import joblib

# --- 1. Credentials (replace with your own IBM Cloud values) ---
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",   # your region's endpoint
    "apikey": "<YOUR_IBM_CLOUD_API_KEY>",
}
SPACE_ID = "<YOUR_DEPLOYMENT_SPACE_ID>"

client = APIClient(wml_credentials)
client.set.default_space(SPACE_ID)

# --- 2. Load the locally trained model ---
model = joblib.load("best_model.pkl")

# --- 3. Define software specification (matches training environment) ---
sofware_spec_uid = client.software_specifications.get_uid_by_name("runtime-23.1-py3.10")

meta_props = {
    client.repository.ModelMetaNames.NAME: "credit-card-fraud-detection-model",
    client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid,
}

# --- 4. Store the model in the Watson repository ---
published_model = client.repository.store_model(model=model, meta_props=meta_props)
published_model_uid = client.repository.get_model_id(published_model)
print("Stored model UID:", published_model_uid)

# --- 5. Deploy it as an online (real-time) scoring endpoint ---
deployment_meta = {
    client.deployments.ConfigurationMetaNames.NAME: "credit-card-fraud-detection-deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
}
deployment = client.deployments.create(published_model_uid, meta_props=deployment_meta)
deployment_uid = client.deployments.get_uid(deployment)
scoring_endpoint = client.deployments.get_scoring_href(deployment)
print("Deployment UID:", deployment_uid)
print("Scoring endpoint:", scoring_endpoint)

# --- 6. Example: score a single transaction against the deployed endpoint ---
# payload = {
#     "input_data": [{
#         "fields": ["Time", "V1", "V2", "...", "V28", "Amount"],
#         "values": [[0.0, -1.36, -0.07, "...", 0.13, 149.62]]
#     }]
# }
# result = client.deployments.score(deployment_uid, payload)
# print(result)
