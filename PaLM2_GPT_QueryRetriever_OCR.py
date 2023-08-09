import json
import streamlit as st


type = st.secrets['type']
project_id = st.secrets['project_id']
private_key_id = st.secrets['private_key_id']
private_key = st.secrets['private_key']
client_email = st.secrets['client_email']
client_id = st.secrets['client_id']
auth_uri = st.secrets['auth_uri']
token_uri = st.secrets['token_uri']
auth_provider_x509_cert_url = st.secrets['auth_provider_x509_cert_url']
client_x509_cert_url = st.secrets['client_x509_cert_url']
universe_domain = st.secrets['universe_domain']


toml_data = {
    'type': type,
    'project_id': project_id,
    'private_key_id': private_key_id,
    'private_key': private_key,
    'client_email': client_email,
    'client_id': client_id,
    'auth_uri': auth_uri,
    'token_uri': token_uri,
    'auth_provider_x509_cert_url': auth_provider_x509_cert_url,
    'client_x509_cert_url': client_x509_cert_url,
    'universe_domain': universe_domain
}


json_data = json.dumps(toml_data, indent=4)

st.write(json_data)
