import os
import tempfile
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

@app.route('/verify', methods=['POST'])
def verify_faces():
    try:
        img1_base64 = request.json.get('img1')
        img2_base64 = request.json.get('img2')

        if not img1_base64 or not img2_base64:
            return jsonify({"error": "Both img1 and img2 are required."}), 400

        img1_data = base64.b64decode(img1_base64)
        img2_data = base64.b64decode(img2_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img1, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img2:

            img1_path = tmp_img1.name
            img2_path = tmp_img2.name

            with open(img1_path, 'wb') as f1:
                f1.write(img1_data)
            with open(img2_path, 'wb') as f2:
                f2.write(img2_data)

            # Run face verification
            try:
                result = DeepFace.verify(img1_path, img2_path)
            except ValueError as e:
                # Common error when face is not detected
                return jsonify({"error": "Face not detected in one or both images."}), 400

            return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up
        try:
            if os.path.exists(img1_path):
                os.remove(img1_path)
            if os.path.exists(img2_path):
                os.remove(img2_path)
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
