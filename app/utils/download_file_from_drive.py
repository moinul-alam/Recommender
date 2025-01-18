import requests

def download_file_from_drive(url: str, local_path: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")
