from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from ml_model.preprocess import preprocess_input
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_bcrypt import Bcrypt
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

bcrypt = Bcrypt(app)

# In-memory user store for demo (replace with DB in production)
users = {}

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# Load model
model = tf.saved_model.load('ml_model/fixed_model')
predict_fn = model.signatures['serving_default']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        processed = preprocess_input(input_data).reshape(1, 1, -1).astype(np.float32)
        output = predict_fn(tf.constant(processed))
        pred = output['output_0'].numpy()
        result = ['Non-Diabetic', 'Type 1 Diabetes', 'Type 2 Diabetes'][np.argmax(pred)]
        return render_template('predictions.html', prediction=result)
    except Exception as e:
        return render_template('predictions.html', prediction=f"Error: {str(e)}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/diet')
def diet():
    return render_template('Diet.html')

@app.route('/articles')
def articles():
    # Example: Fetch latest diabetes news from a reputable source
    news = []
    try:
        res = requests.get('https://www.medicalnewstoday.com/categories/diabetes')
        soup = BeautifulSoup(res.text, 'html.parser')
        for card in soup.select('li.card')[:6]:
            title = card.select_one('h3')
            link = card.select_one('a')
            img = card.select_one('img')
            if title and link:
                news.append({
                    'title': title.text.strip(),
                    'url': link['href'],
                    'img': img['src'] if img else None
                })
    except Exception as e:
        news = [{'title': 'Could not fetch live news.', 'url': '#', 'img': None}]
    return render_template('Articles.html', news=news)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if any(u.username == username for u in users.values()):
            return render_template('register.html', error='Username already exists')
        user_id = str(len(users) + 1)
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(user_id, username, password_hash)
        users[user_id] = user
        login_user(user)
        return render_template('home.html', user=current_user)
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = next((u for u in users.values() if u.username == username), None)
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return render_template('home.html', user=current_user)
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return render_template('login.html', message='Logged out successfully')

if __name__ == '__main__':
    app.run(debug=True)