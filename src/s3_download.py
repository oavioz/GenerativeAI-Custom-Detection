import boto3
import os
from botocore.exceptions import NoCredentialsError
from urllib.parse import urlparse
from pathlib import Path
import secrets
import random


# Replace these with your AWS credentials and S3 bucket information
aws_access_key_id = 'AKIAUIWGZP2QT3F4LMMY'
aws_secret_access_key = 'YHiewi4nxlO2EQBEeUwRb9toVwyA5blp+kLFlBoR'


s3_uri = 's3://your-s3-bucket/your-object-key'


# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


def download_file_from_s3(s3_uri, download_path):
    # Parse the S3 URI to extract bucket and object key
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')

    directory_path = Path(download_path)
    
    # Check if the directory exists, and if not, create it
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)

    image_name = object_key.split("/")[-1];
    # Generate a random number and append it to the file name
    random_number = random.randint(1, 100000)  # Adjust the range as needed
    file_name_with_random = f"{os.path.splitext(image_name)[0]}_{random_number}{os.path.splitext(image_name)[1]}"
    download_path = os.path.join(directory_path, file_name_with_random)

    print(f"Downloading s3://{bucket_name}/{object_key} to {download_path}...")
    
    # Download the file
    try:
        # Download the file
        s3.download_file(bucket_name, object_key, download_path)
        print(f"Downloaded '{s3_uri}' to '{download_path}'")
        return download_path;
    except NoCredentialsError:
        print("No AWS credentials found. Please provide valid credentials.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None



async def getPresingedURL(s3_uri):
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')
    # Generate a presigned URL for the S3 object
    url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': object_key}, ExpiresIn=3600)  # URL valid for 1 hour (3600 seconds)
    print(f"Generated presigned URL: {url} for {s3_uri}")
    return url;