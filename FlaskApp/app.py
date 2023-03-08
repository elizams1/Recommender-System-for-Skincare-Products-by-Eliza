from flask import Flask, render_template, request
import pandas as pd
from flask_paginate import Pagination, get_page_parameter, get_page_args


df = pd.read_csv('D:\TA CODING\SISTEM REKOMENDASI - ELIZA\Recommender-System-for-Skincare-Products-by-Eliza\FlaskApp\data\coba.csv')

app = Flask(__name__)

df_html = df.values

def get_datas(offset=0, per_page=10):
  return df_html[offset: offset + per_page]

def get_search(data, offset=0, per_page=10):
  return data[offset: offset + per_page]

@app.route('/')
def main():
  #PAGINATION
  page, per_page, offset = get_page_args(page_parameter='page',                       per_page_parameter='per_page')

  total = len(df_html)
  products = get_datas(offset=offset, per_page=per_page)
  pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')

  return render_template('katalog.html',
                          products=products,
                          pagination=pagination,  page=page,
                          per_page=per_page)

@app.route('/katalog', methods=['POST','GET'])
def katalog():
  page, per_page, offset = get_page_args(page_parameter='page',                       per_page_parameter='per_page')
  
  #SEARCH FUNCTION
  if request.method == 'POST':
    input = request.form['myInput']
    data = []
    for value in df_html:
      if value[1].lower().find(input.lower()) != -1:
        data.append(value)
      elif value[3].lower().find(input.lower()) != -1:
        data.append(value)   

    total = len(data)
    products = get_search(data, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4') 
    return render_template('katalog.html', products=products,
                          pagination=pagination,  page=page,
                          per_page=per_page)
  
  #ALL PRODUCT
  else:
    total = len(df_html)
    products = get_datas(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')

    return render_template('katalog.html', products=products,
                          pagination=pagination,  page=page,
                          per_page=per_page)

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