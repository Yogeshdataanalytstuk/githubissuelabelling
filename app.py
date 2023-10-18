import secrets
import csv
from flask import Flask, render_template, request, redirect, url_for
import joblib
app = Flask(__name__)
lst=list()
model = joblib.load(open('model/pred.sav', 'rb'))

GITHUB_TOKEN = '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'
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

@app.route('/')
def index():
    print(HEADERS)
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    issue_number = generate_issue_number()
    heading = [(request.form['heading'])]
    comment = [(request.form['comment'])]
    combined_text = f"{heading}_{comment}"
    prediction=model.predict([combined_text])
    print(prediction)
    with open('predicted_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['full_text', 'issue_label', 'issue_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'full_text': combined_text, 'issue_label': prediction, 'issue_number': lst[-1]})

    # Process the data as needed (e.g., store it in a database)
    print(f'Issue Number: {issue_number[-1]}, Heading: {heading}, Comment: {comment}')
    
    feedback_url = url_for('feedback', reference=issue_number)
    return render_template('feedback.html', feedback_url=feedback_url)

@app.route('/submit/feedback', methods=['POST'])
def feedback():
    feedback = request.form['feedback']
    reference_number = request.args.get('reference')
    with open('feedback.csv', 'a', newline='') as csvfile:
        fieldnames = ['issue_number', 'issue_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'issue_number': lst[-1], 'issue_label': feedback})

    
    # Process the feedback data along with the reference number
    print(f'Issue Number: {lst[-1]}, Feedback: {feedback}')
    return redirect(url_for('index'))

def generate_issue_number():
    # Generate a random 64-character hexadecimal token
    issue_number = secrets.token_hex(32)
    lst.append(issue_number)
    return lst

if __name__ == '__main__':
    app.run()
