# Download files from drive
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# choose a local (colab) directory to store the data.
# TODO: 1. replace /inputs with your own desired path
local_download_path = os.path.expanduser('/inputs')
try:
  os.makedirs(local_download_path)
except: pass

file_list = drive.ListFile(
    # TODO: 2. Replace 1AraDPVMwDecHqCFNDR9OfelCic0mJLNI with your own folder path
    # You can get this number from an URL such as https://drive.google.com/drive/u/1/folders/1AraDPVMwDecHqCFNDR9OfelCic0mJLNI
    {'q': "'1AraDPVMwDecHqCFNDR9OfelCic0mJLNI' in parents"}).GetList()

for f in file_list:
  # 3. Create & download by id.
  print('title: %s, id: %s' % (f['title'], f['id']))
  fname = os.path.join(local_download_path, f['title'])
  print('downloading to {}'.format(fname))
  f_ = drive.CreateFile({'id': f['id']})
  f_.GetContentFile(fname)
