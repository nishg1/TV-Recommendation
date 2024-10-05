# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
from Rec_Algorithm import find_similar_shows

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['tv_show']
    recommendations = find_similar_shows(user_input)
    return render_template('recommendations.html', recommendations=recommendations, tv_show=user_input)
  
if __name__=='__main__': 
   app.run(debug = True)
   

