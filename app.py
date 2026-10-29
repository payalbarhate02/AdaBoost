import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained AdaBoost model
try:
    with open('adaboost.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Attractive HTML Template using Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-50 via-indigo-50 to-blue-100 min-h-screen py-10 px-4">
    <div class="max-w-4xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-100">
        
        <!-- Header -->
        <div class="bg-gradient-to-r from-indigo-600 to-violet-700 py-8 px-8 text-white">
            <h1 class="text-3xl font-extrabold tracking-tight">Student Performance Predictor</h1>
            <p class="mt-2 text-indigo-100 text-sm">Enter student details below to generate predictions using your deployed AdaBoost model.</p>
        </div>

        <!-- Form -->
        <form method="POST" class="p-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Age</label>
                <input type="number" name="age" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 18">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Gender (Encoded)</label>
                <input type="number" name="gender" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 0 or 1">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">State (Encoded)</label>
                <input type="number" name="state" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 5">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Family Income ($)</label>
                <input type="number" step="any" name="family_income" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 75000">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">High School GPA</label>
                <input type="number" step="0.01" name="high_school_gpa" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 3.8">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">SAT Score</label>
                <input type="number" name="sat_score" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 1350">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">ACT Score</label>
                <input type="number" name="act_score" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 28">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Attendance Rate (0.0 - 1.0)</label>
                <input type="number" step="0.01" name="attendance_rate" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 0.95">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">AP Courses Count</label>
                <input type="number" name="ap_courses" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 4">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Extracurricular Count</label>
                <input type="number" name="extracurricular_count" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 3">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Volunteer Hours</label>
                <input type="number" name="volunteer_hours" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 50">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Leadership Positions</label>
                <input type="number" name="leadership_positions" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 2">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Coding Projects</label>
                <input type="number" name="coding_projects" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 5">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Social Media Hours / Day</label>
                <input type="number" step="0.1" name="social_media_hours" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 2.5">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Online Certifications</label>
                <input type="number" name="online_certifications" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 3">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Essay Score (0 - 100)</label>
                <input type="number" step="0.1" name="essay_score" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 88.5">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Recommendation Score (0 - 100)</label>
                <input type="number" step="0.1" name="recommendation_score" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 90.0">
            </div>

            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Interview Score (0 - 100)</label>
                <input type="number" step="0.1" name="interview_score" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" placeholder="e.g., 85.0">
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 shadow-lg transform hover:-translate-y-0.5">
                    Run Prediction
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="bg-indigo-50 border-t border-indigo-100 p-6 text-center">
            <h2 class="text-lg font-semibold text-gray-600 uppercase tracking-wider">Prediction Result</h2>
            <p class="mt-2 text-3xl font-extrabold text-indigo-600">{{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Features mapped precisely to the model's required input schema[cite: 2]
            features = [
                float(request.form['age']),
                float(request.form['gender']),
                float(request.form['state']),
                float(request.form['family_income']),
                float(request.form['high_school_gpa']),
                float(request.form['sat_score']),
                float(request.form['act_score']),
                float(request.form['attendance_rate']),
                float(request.form['ap_courses']),
                float(request.form['extracurricular_count']),
                float(request.form['volunteer_hours']),
                float(request.form['leadership_positions']),
                float(request.form['coding_projects']),
                float(request.form['social_media_hours']),
                float(request.form['online_certifications']),
                float(request.form['essay_score']),
                float(request.form['recommendation_score']),
                float(request.form['interview_score'])
            ]
            
            final_features = np.array([features])
            prediction_result = model.predict(final_features)[0]
            prediction = f"{prediction_result}"
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
