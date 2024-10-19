"""
from snippets.cv_model_shipper import CVModelsShipper
_obj = CVModelsShipper(['upload', 'preprod', '/home/tushar/Desktop/Important/gcp_keys/Models/ssdlitefpn_mobilenet.pbtxt'])
_obj.initiate()


from snippets.cv_model_shipper import CVModelsShipper
_obj = CVModelsShipper(['download', 'preprod', 'ssdlitefpn_mobilenet.pbtxt'])
_obj.initiate()
"""


import os
import sys
import logging
import json
from google.cloud import storage
from tqdm import tqdm
import time

logging.basicConfig(filename='/tmp/cv_model_shipper.log', level=logging.INFO, format='%(asctime)s %(message)s')
DATASET_NAME = 'benchmark_dataset_13_1_20'
GS_CREDENTIALS_FILE_LOCATION = "verificientcvmodels.json"
DOWNLOAD_LOCATION = "/home/ajeet/codework/ujjawal_github/"
print('[Info] GS_CREDENTIALS_FILE_LOCATION : ', os.path.exists(GS_CREDENTIALS_FILE_LOCATION))
BUCKET_NAME = 'verificientcvmodels'
FOLDER_NAME_DEV = 'Prestaging'
FOLDER_NAME_PREPROD = 'Preproduction'
FOLDER_NAME_PROD = 'Production'
FOLDER_NAME_RD_PROD = 'RD_Production'
FOLDER_NAME_DATASET = 'Dataset'


class CVModelsShipper(object):
    
    def __init__(self, args):
        self.op_type = args[1]
        self.tenv = args[2]
        self.cv_model_file_path = args[3] if len(args) > 3 else ''
        self.storage_client = storage.Client.from_service_account_json(GS_CREDENTIALS_FILE_LOCATION)
        self.data_for = args[4] if len(args) > 4 else ''
        self.month = args[5] if len(args) > 5 else ''
        self.user_id = args[6] if len(args) > 6 else ''
        self.test_session_id = args[7] if len(args) > 7 else ''
        self.data_type = args[8] if len(args) > 8 else ''
        
    def _get_folder_name(self):
        if self.tenv == 'prod':
            folder_name = FOLDER_NAME_PROD
        elif self.tenv == 'rdprod':
            folder_name = FOLDER_NAME_RD_PROD
        elif self.tenv == 'preprod':
            folder_name = FOLDER_NAME_PREPROD
        elif self.tenv == 'dataset':
            folder_name = FOLDER_NAME_DATASET
        else:
            folder_name = FOLDER_NAME_DEV
        return folder_name


    def _generate_file_upload_bucket_path(self, cv_model_file_path):
        cv_model_file_name = os.path.basename(cv_model_file_path)

        folder_name = self._get_folder_name()

        bucket_path = "{fn}/{cmfn}".format(fn=folder_name, cmfn=cv_model_file_name)
        logging.info("Bucket and file path for Stream : {bp}".format(bp=bucket_path))
        return bucket_path


    def upload_cv_model(self, local_path):
        bucket = self.storage_client.get_bucket(BUCKET_NAME)
        
        object_name = self._generate_file_upload_bucket_path(local_path)
        key = bucket.blob(object_name)

        logging.info("Uplaoding CV Model From: {lp} to verificientcvmodels/{on}".format(lp=local_path, on=object_name))
        
        key.upload_from_filename(local_path)

        msg = "File Uploaded: {status}".format(status=True if bucket.get_blob(object_name) else False )
        logging.info(msg)


    def download_cv_models(self):
        bucket = self.storage_client.get_bucket(BUCKET_NAME)

        file_name = os.path.basename(self.cv_model_file_path)
        object_name = os.path.join(self._get_folder_name(), file_name)
        
        if not bucket.get_blob(object_name):
            msg = "Object {on} does not exists in bucket {bn}.".format(on=object_name, bn=BUCKET_NAME)
            logging.exception(msg)
            print(msg)
            sys.exit(1)

        key = bucket.blob(object_name)
        
        local_path = os.path.join(DOWNLOAD_LOCATION, os.path.basename(key.name))
        print(local_path)
        
        
        logging.info("Downloading CV Model to {lp} from verificientcvmodels/{bn}".format(lp=local_path, bn=key.name))

        key.download_to_filename(local_path)

        download_status = os.path.exists(local_path)
        
        msg = "File {file_name} downloaded at {local_path} with status {status}".format(file_name=file_name,
                                                                                        local_path=local_path,
                                                                                        status=download_status)
        logging.info(msg)

    def download_all_cv_models(self):
        bucket = self.storage_client.get_bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix= "{0}/{1}".format(self._get_folder_name(), DATASET_NAME))
        ########################
        filtered_session=[2591822,2602597,2603060,2619480,2598336,2582196,2578685,2578378,2575985,
        2573678,2572904,2562989,538482,2529909,2565397,2625050,2648356,2648356,2655744,2649876]
        # little hack to ignore the first folder object 
        count = 1
        files_ = []
        pbar = tqdm(total=2359)  # Init pbar
        for blob in blobs:
            session_id = blob.name.split('/')[2] if len(blob.name.split('/')) > 2 else None
            print('[Blob] ', blob.name, session_id)
            print(type(session_id))
            local_path = os.path.join(DOWNLOAD_LOCATION, blob.name)
            pbar.update(n=1)
            if not os.path.exists(local_path):
                print("local_path", local_path)
                print('[Info] ', os.path.dirname(local_path))
                # x=os.path.dirname(local_path).replace('/','\\')
                x = os.path.dirname(local_path)
                print("x", x)
                cmd="mkdir -p "+x
                print("cmd]",cmd)

                os.system(cmd)
                #os.system('md  {}'.format(os.path.dirname(local_path)))

            count += 1
            files_.append(blob.name)


            if os.path.exists(local_path):
                # print('file already exists '+local_path)
                continue
            if not os.path.basename(blob.name):
                continue

            # logging.info("Downloading Data to {lp} from {bn}".format(lp=local_path, bn=blob.name))
            # print('[download_all_cv_models] ', blob.name, local_path)
        #########
            blob.download_to_filename(local_path)
            print('[Info] Count: ', count)
        # with open('test_data.json', 'w+') as f:
        #     json.dump(files_, f)




                # if int(session_id) in filtered_session:
                #     local_path = os.path.join(DOWNLOAD_LOCATION, blob.name)
                #     pbar.update(n=1)
                #     if not os.path.exists(local_path):
                #         print("local_path", local_path)
                #         print('[Info] ', os.path.dirname(local_path))
                #         # x=os.path.dirname(local_path).replace('/','\\')
                #         x = os.path.dirname(local_path)
                #         print("x", x)
                #         cmd="mkdir -p "+x
                #         print("cmd]",cmd)

                #         os.system(cmd)
                #         #os.system('md  {}'.format(os.path.dirname(local_path)))

                #     count += 1
                #     files_.append(blob.name)


                #     if os.path.exists(local_path):
                #         # print('file already exists '+local_path)
                #         continue
                #     if not os.path.basename(blob.name):
                #         continue

                #     # logging.info("Downloading Data to {lp} from {bn}".format(lp=local_path, bn=blob.name))
                #     # print('[download_all_cv_models] ', blob.name, local_path)
                # #########
                #     blob.download_to_filename(local_path)
                #     print('[Info] Count: ', count)
                # # with open('test_data.json', 'w+') as f:
                # #     json.dump(files_, f)

   
    def _generate_od_file_upload_bucket_path(self, cv_model_file_path):
        cv_model_file_name = os.path.basename(cv_model_file_path)

        folder_name = self._get_folder_name()

        bucket_path = "{fn}/Object_detection/{data_for}/{month}/{user_id}/{test_session_id}/{data_type}/{cmfn}".format(fn=folder_name, data_for=self.data_for, month=self.month, user_id=self.user_id, test_session_id=self.test_session_id, data_type=self.data_type, cmfn=cv_model_file_name)
        logging.info("Bucket and file path for Stream : {bp}".format(bp=bucket_path))
        return bucket_path
        
    def upload_od_files(self, local_path):
        bucket = self.storage_client.get_bucket(BUCKET_NAME)
        
        object_name = self._generate_od_file_upload_bucket_path(local_path)
        key = bucket.blob(object_name)

        logging.info("Uplaoding CV Model From: {lp} to verificientcvmodels/{on}".format(lp=local_path, on=object_name))
        
        key.upload_from_filename(local_path)

        msg = "File Uploaded: {status}".format(status=True if bucket.get_blob(object_name) else False )
        logging.info(msg)
    
    def upload_od_data(self,):
        if not os.path.exists(self.cv_model_file_path):
            msg = "File {file_path} does not exists.".format(file_path=self.cv_model_file_path)
            print(msg)
            logging.info(msg)
            sys.exit(1)

        cv_models_files_list = os.listdir(self.cv_model_file_path)
        cv_models_files_list = [f for f in cv_models_files_list if os.path.splitext(f.lower())[1] in ['.jpg', '.jpeg', '.png', '.mp4']]
        for cv_model_file_path in cv_models_files_list:
            logging.info("Starting uploading of CV Model: {cmfp}".format(cmfp=cv_model_file_path))
            print(os.path.join(self.cv_model_file_path, cv_model_file_path))
            self.upload_od_files(os.path.join(self.cv_model_file_path, cv_model_file_path))
            
    def initiate(self, op_type=None):
        if not op_type:
            sys.exit(1)

        if self.op_type=='upload':
            if not os.path.exists(self.cv_model_file_path):
                msg = "File {file_path} does not exists.".format(file_path=self.cv_model_file_path)
                print(msg)
                logging.info(msg)
                sys.exit(1)

            cv_models_files_list = [self.cv_model_file_path]
            for cv_model_file_path in cv_models_files_list:
                logging.info("Starting uploading of CV Model: {cmfp}".format(cmfp=cv_model_file_path))
                self.upload_cv_model(cv_model_file_path)
        elif self.op_type=="download":
            self.download_all_cv_models()
        elif self.op_type == 'upload_od':
            self.upload_od_data()
                
        else:
            sys.exit(1)

    

if __name__ == "__main__":
    if sys.argv[1] == 'upload':
        if len(sys.argv) != 4:
            print ("Usage: python cv_model_shipper.py <upload> <dev|preprod|prod|rdprod> cv_model_file_path")
            sys.exit(1)
    elif sys.argv[1] == 'download':
        if len(sys.argv) != 3:
            print ("Usage: python od_cv_model_shipper.py download <dev|preprod|prod|rdprod>")  # it will download all the models
            sys.exit(1)
    elif sys.argv[1] == 'upload_od':
        if len(sys.argv) != 9:
            print ("Usage: python cv_model_shipper.py <upload_od> <dev|preprod|prod|rdprod|dataset> <data_folder_path> <client/staff> <month> <user_id> <test_session_id> <images/videos>")
            sys.exit(1)
    else:
        print ("Usage: python cv_model_shipper.py <upload|downlaod> <dev|preprod|prod|rdprod> cv_model_file_path")
        sys.exit(1)

    ss = CVModelsShipper(sys.argv)
    ss.initiate(sys.argv[1])
    
    
# python od_cv_model_shipper.py download dataset
