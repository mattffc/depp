# depp

To get this to run:

1. Create a GCP Deep learning instance, doesn't need any gpu cores
2. Install Dlib (face_recognition) by running these commands one line at a time:
2a.sudo apt-get install build-essential cmake pkg-config
2b.sudo apt-get install libx11-dev libatlas-base-dev
2c.sudo apt-get install libgtk-3-dev libboost-python-dev
3. sudo pip3 install face_recognition 
Note this one takes ages (5 mins)
4. Then run the webserver with:
4a. Sudo python3 webserver.py
Note in order for this to run you need the templates folder, empty uploads folder
