from flask import Flask, render_template, request
import pandas as pd

df = pd.read_csv('D:\TA CODING\SISTEM REKOMENDASI - ELIZA\Recommender-System-for-Skincare-Products-by-Eliza\FlaskApp\data\coba.csv')

app = Flask(__name__)

@app.route('/')
def main():
  df_html = df.values
  return render_template('katalog.html', data=df_html)

@app.route('/katalog', methods=['POST','GET'])
def katalog():
  df_html = df.values
  if request.method == 'POST':
    input = request.form['myInput']
    data = []
    for value in df_html:
      if value[1].lower().find(input.lower()) != -1:
        data.append(value)
    return render_template('katalog.html', data=data)
  else:
    return render_template('katalog.html', data=df_html)

@app.route('/rekomendasi')
def rekomendasi():
  return render_template('rekomendasi.html')

@app.route('/detail/')
def detail():
  Id = request.args.get("id")
  df_html = df.values
  for i in range(0, len(df_html)):
    num = i+1
    if Id == str(num) :
      the_data = df_html[i]
      return render_template('detail.html', data=the_data)

if __name__ == '__main__':
  app.run(debug=True)