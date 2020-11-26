# new depp ML python cloud function

def faceRecognition(pathToTestImage):
    faceScore = 1

    import face_recognition
    #image = face_recognition.load_image_file("your_file.jpg")
    #face_locations = face_recognition.face_locations(image)
    print("it worked")

    pathTest = r"C:\Users\mattf\OneDrive\Documents\deppLearning\johnny\img_7.jpeg"
    pathTrue = r"C:\Users\mattf\OneDrive\Documents\deppLearning\static\20170314_114950-min.jpg"

    known_image = face_recognition.load_image_file("jon1.jpg")
    unknown_image = face_recognition.load_image_file("notJon.jpg")

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    print(results)
    
    return faceScore

def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    """
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': ['GET', 'POST', 'OPTIONS'],
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': ['GET', 'POST', 'OPTIONS'],
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    """
    print(request)
    print("now here")
    
    containsImage = False
    request_json = request.get_json()
    print(request_json)
    print("now here 2")
    
    #import numpy as np
    # pulse pal download a vid for ML on cloud function
    from google.cloud import storage
    storageClient = storage.Client()
    import os
    import json

    BUCKET_NAME = "depplearning"
    #SAVE_LOCATION = "/tmp"#r"C:\Users\mattf\OneDrive\Documents\pulseMate"#"/tmp"
    SAVE_LOCATION = r"C:\Users\mattf\OneDrive\Documents\deppLearning\images"#"/tmp"
    #source_blob_name = "VID_20200124_153407_2_Trim.mp4"

    bucket = storageClient.get_bucket(BUCKET_NAME)
    # List all the file names in the bucket
    blobs = bucket.list_blobs()
    print("here we are downloading flacs")
    source_blob_name = request.get_json()["filename"]
    print("source_blob_name",source_blob_name)
    #print("fileIDs",list(fileIDs),type(fileIDs))
    #print("new commit")
    print(blobs)
    #source_blob_name = blobs[0]
    #print(source_blob_name)
    #l=lp
    blob = bucket.blob(source_blob_name)
    print("here",blob)
    blob.download_to_filename(os.path.join(SAVE_LOCATION,"test.jpg"))
    print('Blob {} downloaded to {}.'.format(source_blob_name, SAVE_LOCATION))

    # this is where we need to do faceRecognition stuff
    faceScore = faceRecognition(SAVE_LOCATION)
    
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return faceScore

class PretendRequest():
    def __init__(self):
        self.args = {"messag6":"ok"}
        a=1
    def get_json(self):

        return {"filename": "choppingBoard.jpg"}#d9d716c1-e54d-49b0-8483-3a85df9a4ffb#5559491f-981e-4374-b524-f22da1709d33
        
if __name__ == "__main__":

    
    pretendRequest = PretendRequest()
    hello_world(pretendRequest)