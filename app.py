from flask import Flask, render_template, url_for, request, flash, redirect, logging, session
from flask_mysqldb import MySQL
from wtforms import Form, StringField, PasswordField, validators
from wtforms.validators import InputRequired, Email, Length
import bcrypt
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model_cnn_hi = keras.models.load_model("model_cnn_hi.h5")
model_cnn_lo = keras.models.load_model("model_cnn_lo.h5")
model_cnngru_hi = keras.models.load_model("model_cnngru_hi.h5")
model_cnngru_lo = keras.models.load_model("model_cnngru_lo.h5")
model_gru_hi = keras.models.load_model("model_gru_hi.h5")
model_gru_lo = keras.models.load_model("model_gru_lo.h5")
db = yaml.load(open('db.yaml'))

app.config['UPLOAD_FOLDER'] = 'static/upload'


app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)


@app.route('/')
def home():
  return render_template('index.html')

 
@app.route('/login',methods=['POST', 'GET'])
def login():
	if request.method == 'POST':
		userDetails = request.form
		username = userDetails['username']
		password1 = userDetails['password1']
		
		cur = mysql.connection.cursor()

		result = cur.execute("SELECT * FROM users where username = %s",[username])
		
		if result > 0:
			data = cur.fetchone()
			password = data['password']

			if sha256_crypt.verify(password1, password):
				session['logged_in'] = True
				session['username'] = username
				return redirect(url_for('dashboard'))
			else:
				error = 'Invalid login'
				return render_template('login.html', error=error)
		else:
			error = 'Username not valid'
			return render_template('login.html', error = error)
	return render_template('login.html')

class RegisterForm(Form):
	name = StringField('name')
	username = StringField('username')
	email = StringField('email')
	password1 = PasswordField('password')

@app.route('/register',methods=['POST', 'GET'])
def register():
	form = RegisterForm(request.form)
	if request.method == 'POST':
		name = form.name.data
		username = form.username.data
		email = form.email.data
		password = request.form['password'].encode('utf-8')
		hash_password = bcrypt.hashpw(password, bcrypt.gensalt())
		cur = mysql.connection.cursor()
		cur.execute("INSERT INTO users(name, email, username, password) VALUES (%s, %s, %s, %s)", (name, email, username, hash_password))
		mysql.connection.commit()
		cur.close()
		#return success
		#ADD ALERT HERE
		return render_template('login.html')
	else:
		return render_template('register.html')
	

@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    return render_template('dashboard.html')
def process_df(df):
  one_seq = np.array(df)  
  to_pad = 19200
  new_seq = []
  len_one_seq = len(one_seq)
  last_val = one_seq[-1]
  n = to_pad - len_one_seq
    
  to_concat = np.repeat(one_seq[-1], n).reshape(8,n).transpose()
  new_one_seq = np.concatenate([one_seq, to_concat])
  new_seq.append(new_one_seq)
  final_seq = np.stack(new_seq)

  #truncate the sequence to length 18000
  from keras.preprocessing import sequence
  seq_len = 18000
  eeg_sequence =sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')
  return eeg_sequence



    
#@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file_hi():
    	#if request.method == 'POST':
        option = request.form.getlist('task_type')
        print(option)
		
        f = request.files['file']  
        f.save(f.filename)
        df = pd.read_csv(f.filename)
        print(df) 
        drop_cols = ['Time','FC5','T7','T8','P8','F4','F8'] 
        df.drop(drop_cols,inplace = True,axis =1)
        df.reset_index(drop = True, inplace = True)
        df = np.array(df)
        print('Shape = ',df.shape)
        eeg_data = process_df(df)
        print(eeg_data.shape)
        n1 = eeg_data.shape
        print(len(n1))
        if len(n1) == 3:
        	eeg_data1 = eeg_data.reshape(1,n1[1]*n1[2])
        stdscaler = StandardScaler()
        scaled = stdscaler.fit_transform(eeg_data1)
        s = scaled.shape
        final_data = scaled.reshape(1,1,s[1])
        print(final_data.shape)
        
        label = model_cnngru_hi.predict(final_data)
        print(label)
        if label[0][0] >0.5:
          print('Medium: ', label[0][0])
          result = 'Moderate Workload level'
          return render_template('dashboard.html',label = result)
        if label[0][1]>0.5:
          print('High: ',label[0][1])
          result = 'High Workload level'
          
          return render_template('dashboard.html',label = result)
    	#else:
        #return render_template('dashboard.html')


#@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file_lo():
    	#if request.method == 'POST':
        option = request.form.getlist('task_type')
        print(option)
        
		
        f = request.files['file']  
        f.save(f.filename)
        df = pd.read_csv(f.filename)
        print(df) 
        drop_cols = ['Time','FC5','T7','T8','P8','F4','F8'] 
        df.drop(drop_cols,inplace = True,axis =1)
        df.reset_index(drop = True, inplace = True)
        df = np.array(df)
        print('Shape = ',df.shape)
        eeg_data = process_df(df)
        print(eeg_data.shape)
        n1 = eeg_data.shape
        print(len(n1))
        if len(n1) == 3:
        	eeg_data1 = eeg_data.reshape(1,n1[1]*n1[2])
        stdscaler = StandardScaler()
        scaled = stdscaler.fit_transform(eeg_data1)
        s = scaled.shape
        final_data = scaled.reshape(1,1,s[1])
        print(final_data.shape)
       
        label = model_cnngru_lo.predict(final_data)
       
        print(label)
        if label[0][0] >0.5:
          print('Low: ', label[0][0])
          result = 'Low Workload level'
          return render_template('dashboard.html',label = result)
        if label[0][1]>0.5:
          print('Medium: ',label[0][1])
          result = 'Moderate workload level'
          return render_template('dashboard.html',label = result)
    	#else:
        #return render_template('dashboard.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
     option = request.form.getlist('task_type')
     print(option)
     if 'simkap' in option:
       print('Going to high')
       return upload_file_hi()
     else:
       print('Going to low')
       return upload_file_lo()
    else:
     return render_template('dashboard.html')

class EditForm(Form):
	id = StringField('uid')
	username = StringField('username')
	name = StringField('name')
	email = StringField('email')
	password = PasswordField('password')

@app.route('/editprofile')
def editprofile():
	form = EditForm(request.form)
	if request.method == 'POST':
		username = form.username.data
		name = form.name.data
		email = form.email.data
		password = request.form['password'].encode('utf-8')
		hash_password = bcrypt.hashpw(password, bcrypt.gensalt())
		cur = mysql.connection.cursor()
		cur.execute("UPDATE users SET name='%s', username='%s', password='%s' where ",(name, username, password))
		mysql.connection.commit()
		cur.close()
		return render_template('editprofile.html')
	else:
		return render_template('editprofile.html')


if __name__ == '__main__':
  app.run(debug=True)