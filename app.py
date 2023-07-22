from flask import Flask
from flask import Flask, jsonify,request
import garden_care as gc


app = Flask(__name__)

@app.route('/')
def root_path():
    res={
        "message":"invalid route",
        "data":None
    }
    return jsonify(res)

@app.route('/gardencare', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        res={
            "message":"no file found",
            "data":None
        }
        return jsonify(res)

    file = request.files['file']
    if file.filename == '':
         res={
            "message":"no file found",
            "data":None
        }
         return jsonify(res)
     
    predicted_class_label, confidence = gc.recognize(file)
    res={"message":"data retrieved successfully",
            "data":{
                "disease_name": predicted_class_label,
                "confidence": confidence
                }
        }
    return res


if __name__ == '__main__':
    app.run()
