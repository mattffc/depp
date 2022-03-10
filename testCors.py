#test CORS
from flask import Flask
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/apitest", methods=["POST"])
#@cross_origin()
def post_example():
    """POST in server"""
    return "yo there"#jsonify(message="POST request returned, cors done success")
    
if __name__ == "__main__":
    print("hello")
    app.run(host='0.0.0.0', port=80, debug=True,threaded=False)