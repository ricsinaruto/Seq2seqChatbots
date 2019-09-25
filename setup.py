import os


print('Installing requirements...')
os.system('pip install -r requirements.txt')

import requests
import zipfile
from clint.textui import progress


def download_data(url, zipped_path, extract):
  # Open the url and download the data with progress bars.
  data_stream = requests.get(url, stream=True)

  with open(zipped_path, 'wb') as file:
    total_length = int(data_stream.headers.get('content-length'))
    for chunk in progress.bar(data_stream.iter_content(chunk_size=1024),
                              expected_size=total_length / 1024 + 1):
      if chunk:
        file.write(chunk)
        file.flush()

  # Extract file.
  zip_file = zipfile.ZipFile(zipped_path, 'r')
  zip_file.extractall(extract)
  zip_file.close()


print('Do you want to download all datasets used in the paper (116 MB)? (y/n)')
if input() == 'y':
  if not os.path.exists('data'):
    os.mkdir('data')
  download_data('https://ricsinaruto.github.io/website/docs/Twitter.zip', 'data/Twitter.zip', 'data')
  download_data('https://ricsinaruto.github.io/website/docs/Cornell.zip', 'data/Cornell.zip', 'data')
  download_data('https://ricsinaruto.github.io/website/docs/DailyDialog.zip', 'data/DailyDialog.zip', 'data')

print('Do you want to download all generated responses on the test set by the different models (7 MB)? (y/n)')
if input() == 'y':
  download_data('https://ricsinaruto.github.io/website/docs/responses.zip', 'responses.zip', '')
