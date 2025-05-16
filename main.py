
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the pre-trained SVM model
model = joblib.load('email-spam/svm_email_classifier.pkl')

# Database setup
DATABASE = 'emails.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                receiver TEXT,
                subject TEXT,
                email_text TEXT,
                category TEXT
            )
        """)
        conn.commit()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    session['user'] = username
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/send_email', methods=['POST'])
def send_email():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    sender = session['user']
    receiver = request.form['receiver']
    subject=request.form['subject']
    email_text = request.form['email_text']
    
    # Classify the email
    category = "Spam" if model.predict([email_text])[0] == 1 else "Ham"

    # Store email in database
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO emails (sender, receiver,subject, email_text, category) VALUES (?,?, ?, ?, ?)",
                       (sender, receiver,subject, email_text, category))
        conn.commit()
    
    return redirect(url_for('dashboard'))

from flask import jsonify

@app.route('/view_emails/<email_type>')
def view_emails_by_type(email_type):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user = session['user']
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT subject,sender, email_text FROM emails WHERE receiver=? AND category=?",
            (user, email_type.capitalize())
        )
        emails = cursor.fetchall()
    
    # Format response as a JSON array of dictionaries
    email_list = [{"subject":email[0],"sender": email[1], "content": email[2]} for email in emails]
    return jsonify(email_list)

    

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

