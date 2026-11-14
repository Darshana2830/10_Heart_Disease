import os
import json
import boto3
from flask import Flask, render_template, request

app = Flask(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "heart-disease-endpoint-10-v142")
runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Numeric categories are taken from heart_disease_dataset_02.csv.
CATEGORICAL = {
    "sex": [0, 1],
    "cp": [0, 1, 2, 3],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [0, 1, 2],
    "ca": [0, 1, 2, 3],
    "thal": [0, 1, 2, 3],
}

FIELDS = [
    {"key": "age", "label": "Age (years)", "kind": "number", "min": 1, "max": 120, "step": "1"},
    {"key": "sex", "label": "Sex", "kind": "select", "options": CATEGORICAL["sex"]},
    {"key": "cp", "label": "Chest pain type (cp)", "kind": "select", "options": CATEGORICAL["cp"]},
    {"key": "trestbps", "label": "Resting blood pressure (mm Hg)", "kind": "number", "min": 50, "max": 250, "step": "1"},
    {"key": "chol", "label": "Serum cholesterol (mg/dl)", "kind": "number", "min": 50, "max": 700, "step": "1"},
    {"key": "fbs", "label": "Fasting blood sugar (fbs)", "kind": "select", "options": CATEGORICAL["fbs"]},
    {"key": "restecg", "label": "Resting ECG result (restecg)", "kind": "select", "options": CATEGORICAL["restecg"]},
    {"key": "thalach", "label": "Maximum heart rate (thalach)", "kind": "number", "min": 50, "max": 250, "step": "1"},
    {"key": "exang", "label": "Exercise-induced angina (exang)", "kind": "select", "options": CATEGORICAL["exang"]},
    {"key": "oldpeak", "label": "ST depression (oldpeak)", "kind": "number", "min": 0, "max": 10, "step": "any"},
    {"key": "slope", "label": "Slope (slope)", "kind": "select", "options": CATEGORICAL["slope"]},
    {"key": "ca", "label": "Number of major vessels (ca)", "kind": "select", "options": CATEGORICAL["ca"]},
    {"key": "thal", "label": "Thalassemia (thal)", "kind": "select", "options": CATEGORICAL["thal"]},
]

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", fields=FIELDS, values={})

    try:
        values_by_key = {}
        for field in FIELDS:
            key = field["key"]
            raw = request.form.get(key, "").strip()
            if raw == "":
                raise ValueError(f"Please provide {field['label']}.")

            value = float(raw)
            if field["kind"] == "select":
                allowed = CATEGORICAL[key]
                if not value.is_integer() or int(value) not in allowed:
                    raise ValueError(
                        f"{field['label']} must be one of: {', '.join(map(str, allowed))}."
                    )
                value = int(value)
            else:
                if value < field["min"] or value > field["max"]:
                    raise ValueError(
                        f"{field['label']} must be between {field['min']} and {field['max']}."
                    )
            values_by_key[key] = value

        instance = [values_by_key[key] for key in FEATURE_ORDER]
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"instances": [instance]})
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        prediction = int(result["predictions"][0])
        probability = float(result["disease_probabilities"][0])

        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability * 100
        )

    except ValueError as exc:
        return render_template(
            "index.html", fields=FIELDS,
            values=request.form, error=str(exc)
        ), 400
    except Exception:
        app.logger.exception("SageMaker prediction request failed")
        return render_template(
            "index.html", fields=FIELDS,
            values=request.form,
            error="Prediction request failed. Check that the SageMaker endpoint is InService and AWS credentials allow sagemaker:InvokeEndpoint."
        ), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
