//web app minimal for depp
const express = require('express')
const app = express()
const port = 3000
const fileupload = require('express-fileupload')
const fs = require('fs');
var request = require('request');

//__dirname = "public"
app.get('/', function(req, res){
res.sendFile(__dirname + '/public/index.html');


console.log("in opening func")
console.log("here before http to depp arrrrr")
var filepath = 'C:\Users\mattf\Downloads\98975842260E89A600EA4AF0A6F18713.jpg'
//data = {"filepath",filepath}
app.get('https://us-central1-ml-bp-project.cloudfunctions.net/faceRecog1', function (req, res) {
    res.sendFile(filepath);
    //res.json(data);
})

console.log("here 5")
request.post(
        'https://us-central1-ml-bp-project.cloudfunctions.net/faceRecog1',
        { json: { "testKey": "testItem" } },
        function (error, response, body) {
            if (!error && response.statusCode == 200) {
                console.log(body);
            }
            console.log(response.statusCode)
        }
    );


});



app.use(
		fileupload(),
        )
app.use(express.static('public'));
        
app.post('/saveImage', (req, res) => {
  const fileName = req.files.myFile.name
  const path = __dirname + '/images/' + fileName
    console.log("fileNameloool",fileName)
    console.log(req.files.myFile)
    console.log("path",path);
    fs.writeFile(path, req.files.myFile.data, function (err) {
         if (err) {
            console.log("err",err)
         }

         console.log("done it")
      });
    
    console.log("about to start upload to bucket code")
    // first need to save image locally
    filename = fileName
    
    
    // upload image to bucket here
    bucketName = "depplearning"
    const {Storage} = require('@google-cloud/storage');
    console.log(filename,"filenameloool")
    //filename = 'C:\\Users\\mattf\\Downloads\\8ef4ce55-2507-40b7-9926-7626fa0b8e4f.flac'
    console.log(path,"pathloool")
  // Creates a client
  const storage = new Storage();
    console.log("before async 777")
    console.log("process.argv.slice(2)",process.argv.slice(2))
  async function uploadFile() {
      console.log("start of function upload 324")
    // Uploads a local file to the bucket
    await storage.bucket(bucketName).upload(path, {
      // Support for HTTP requests made with `Accept-Encoding: gzip`
      gzip: false,
      // By setting the option `destination`, you can change the name of the
      // object you are uploading to a bucket.
      metadata: {
        // Enable long-lived HTTP caching headers
        // Use only if the contents of the file will never change
        // (If the contents will change, use cacheControl: 'no-cache')
        cacheControl: 'public, max-age=31536000',
      },
    });
    console.log("done async")
    request.post(
        'https://us-central1-ml-bp-project.cloudfunctions.net/faceRecog1',
        { json: { filename: filename } },
        function (error, response, body) {
            if (!error && response.statusCode == 200) {
                console.log(body);
            }
            console.log(response.statusCode)
        }
    );
    console.log("done hope 200")
    
  }
  uploadFile()
    
    //main("bucketnameazhir",path,...process.argv.slice(2));
    console.log("try to save image here")
    
    /*
  image.mv(path, (error) => {
    if (error) {
      console.error(error)
      res.writeHead(500, {
        'Content-Type': 'application/json'
      })
      res.end(JSON.stringify({ status: 'error', message: error }))
      return
    }

    res.writeHead(200, {
      'Content-Type': 'application/json'
    })
    res.end(JSON.stringify({ status: 'success', path: '/img/houses/' + fileName }))
  }
  
  
  )*/
})

//app.listen(port, () => {
//  console.log(`Example app listening at http://localhost:${port}`)
//})

//////////////////////////////////////////////////////////////////////////////////// GOOOGLE FROM HERE
// Imports the Google Cloud client library
/*
const {Storage} = require('@google-cloud/storage');

// Creates a client
const storage = new Storage();
// Creates a client from a Google service account key.
//const storage = new Storage({keyFilename: "key.json"});


//TODO(developer): Uncomment these variables before running the sample.

const bucketName = 'bucketnameazhir';

async function createBucket() {
  // Creates the new bucket
  await storage.createBucket(bucketName);
  console.log(`Bucket ${bucketName} created.`);
}

createBucket().catch(console.error);
*/
////////////////////////////////////////////////////////////////////////////////////////////
/*
function main(bucketName = 'bucketnameazhir', filename = 'C:\Users\mattf\Downloads\8ef4ce55-2507-40b7-9926-7626fa0b8e4f.flac') {
  // [START storage_upload_file]
  console.log("bucketName",bucketName)
  console.log("doing upload to bucket in here",filename)
   //const bucketName = bucketName//'Name of a bucket, e.g. my-bucket';
   //const filename = filename//'Local file to upload, e.g. ./local/path/to/file.txt';

  // Imports the Google Cloud client library
  const {Storage} = require('@google-cloud/storage');
    console.log(filename,"loool")
    //filename = 'C:\\Users\\mattf\\Downloads\\8ef4ce55-2507-40b7-9926-7626fa0b8e4f.flac'
    console.log(filename,"loool")
  // Creates a client
  const storage = new Storage();

  async function uploadFile() {
    // Uploads a local file to the bucket
    await storage.bucket(bucketName).upload(filename, {
      // Support for HTTP requests made with `Accept-Encoding: gzip`
      gzip: true,
      // By setting the option `destination`, you can change the name of the
      // object you are uploading to a bucket.
      metadata: {
        // Enable long-lived HTTP caching headers
        // Use only if the contents of the file will never change
        // (If the contents will change, use cacheControl: 'no-cache')
        cacheControl: 'public, max-age=31536000',
      },
    });

    console.log(`${filename} uploaded to ${bucketName}.`);
  }

  uploadFile().catch(console.error);
  // [END storage_upload_file]
}
*/

//main(...process.argv.slice(2));

//app.get('/', function(req, res){
//res.sendFile(__dirname + '/public/upload.html');
//});

//console.log("done sending http to ML function on cloud")

// Start the server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});
// [END gae_node_request_example]



module.exports = app;
