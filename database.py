from google.cloud import storage
from google.oauth2 import service_account
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

credentials = service_account.Credentials.from_service_account_file(
    os.path.join(current_dir, 'database_credentials.json')
)
client = storage.Client(credentials=credentials)

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"Uploaded {source_file_path} to gs://{bucket_name}/{destination_blob_name}")

def download_from_gcs(bucket_name, source_blob_name, destination_file_path):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)
    print(f"Downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_path}")

# Example: list buckets
buckets = list(client.list_buckets())
for bucket in buckets:
    print(bucket.name)

upload_to_gcs("training_data_sample_bucket", "database_credentials.json", "database")