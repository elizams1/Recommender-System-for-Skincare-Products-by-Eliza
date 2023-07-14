from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from flask_paginate import Pagination, get_page_parameter, get_page_args

import pandas as pd
import numpy as np


# untuk menggunakan library nltk
# pip install nltk
import nltk
import string
import re
from nltk.tokenize import word_tokenize
# Mendownload package untuk menghapus tanda baca
nltk.download('punkt')
nltk.download('stopwords')

# untuk menggunakan library Sastrawi
# pip install PySastrawi

# untuk menggunakan library sklearn
# pip install -U scikit-learn

df_product = pd.read_csv(
    './static/data/DataProdukVers5.csv')

df_user = pd.read_csv(
    './static/data/DataPengguna4.csv')

df_dataset = pd.read_excel(
    './static/data/Dataset Digabungkan.xlsx')

app = Flask(__name__)

df_user["Kandungan"].fillna(" ", inplace=True)

# mengambil value dari dataframe df_product
df_html = df_product.values

# mengambil value dari dataframe df_user
df_user_value = df_user.values

# mendapatkan 10 data per halaman
def get_datas(offset=0, per_page=10):
    return df_html[offset: offset + per_page]

# mencari 10 data per halaman
def get_search(data, offset=0, per_page=10):
    return data[offset: offset + per_page]


# TEXT PREPROCESSING INPUTAN
def text_pre_input_two(user_input):
  # CASE FOLDING
    user_input = user_input.lower()

  # TOKENIZING
    # Menghapus Angka
    user_input = re.sub(r"\d+", "", user_input)
    # Menghapus Tanda Baca
    user_input = user_input.translate(
        str.maketrans("", "", string.punctuation))
    # Menghapus Simbol (@)
    user_input = re.sub(r"@\S+", "", user_input)
    # Menghapus URL
    user_input = re.sub("http[s]?\://\S+", "", user_input)
    # Menghapus tab atau line karakter
    user_input = re.sub(r"\n", "", user_input)
    # Menghapus berlebihan spasi
    user_input = re.sub('\s+', ' ', user_input)
    # dilakukan proses tokenization pada user_input
    tokens_query = [str(tok) for tok in nltk.word_tokenize(user_input)]

  # FILTERING
    # Stopword Bahasa Indonesia
    indo_stopword = set(stopwords.words('indonesian'))

    # Stopword Bahasa Inggris
    eng_stopword = set(stopwords.words('english'))

    # Data Stopword
    data_stopword = str(indo_stopword) + " " + str(eng_stopword)

    # Data tanpa stopword
    tokens_query = [word for word in tokens_query if not word in data_stopword]

  # STEMMING
    the_query = []
    # Membuat stemmer
    Factory = StemmerFactory()
    Stemmer = Factory.create_stemmer()
    for i in range(0, len(tokens_query)):
        tokens_query[i] = Stemmer.stem(tokens_query[i])
        the_query.append(tokens_query[i])

    join_query = ' '.join(tokens_query)
    return (join_query)


def get_recommendations_product_two(tfidf_dataset):
    # Mencari tingkat kemiripan antara user_input dengan dataset
    cos_sim = cosine_similarity(tfidf_dataset, tfidf_dataset)
    # print(cos_sim)
    df_cosim = pd.DataFrame(data=cos_sim)
    # display(df_cosim)

    # Mengambil nilai cosine similarity pada baris yang berisi inputan pengguna
    val_cos_sim = pd.DataFrame(
        data=cos_sim[-1, :], columns=['Cosine Similarity Score'])
    # display(val_cos_sim)

    # Mencari 5 teratas nilai kemiripan tertinggi
    top5 = val_cos_sim.sort_values(
        "Cosine Similarity Score", ascending=False)[1:6]
    # display(top5)

    # Mencari index dari 5 nilai kemiripan teratas
    index = list(top5.index.values)

    return index, top5

# HALAMAN UTAMA


@app.route('/')
def main():
    # PAGINATION
    page, per_page, offset = get_page_args(
        page_parameter='page',
        per_page_parameter='per_page')

    total = len(df_html)
    products = get_datas(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page,
                            total=total, css_framework='bootstrap4')

    return render_template('katalog.html',
                           products=products,
                           pagination=pagination,  page=page,
                           per_page=per_page)

# HALAMAN KATALOG


@app.route('/katalog', methods=['POST', 'GET'])
def katalog():
    page, per_page, offset = get_page_args(
        page_parameter='page',
        per_page_parameter='per_page')

    # SEARCH FUNCTION
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
        pagination = Pagination(
            page=page, per_page=per_page, total=total, css_framework='bootstrap4')
        return render_template('katalog.html', products=products,
                               pagination=pagination,  page=page,
                               per_page=per_page)

    # ALL PRODUCT
    else:
        total = len(df_html)
        products = get_datas(offset=offset, per_page=per_page)
        pagination = Pagination(
            page=page, per_page=per_page, total=total, css_framework='bootstrap4')

        return render_template('katalog.html', products=products,
                               pagination=pagination,  page=page,
                               per_page=per_page)


@app.route('/rekomendasi', methods=['POST', 'GET'])
def rekomendasi():

    # REKOMENDASI SUBMIT
    if request.method == 'POST':
        type_product = request.form.get('inputTypeProduct')
        type_skin = request.form.get('inputTypeSkin')
        problem_skin = request.form.get('inputProblemSkin')

        data_input_user = ""
        iP = ""
        tP = type_product
        tS = type_skin
        pS = problem_skin

        for i in range(0, len(df_user_value)):
            if df_user_value[i][1] == type_product and df_user_value[i][2] == type_skin and df_user_value[i][3] == problem_skin:
                print(df_user_value[i][1] + " " + df_user_value[i][2] +
                      " " + df_user_value[i][3] + " " + df_user_value[i][4])
                data_input_user = df_user_value[i][1] + " " + df_user_value[i][2] + \
                    " " + df_user_value[i][3] + " " + df_user_value[i][4]
                if df_user_value[i][2] == 'Normal' and df_user_value[i][3] == 'Normal':
                    iP = " yang cocok untuk kulit yang normal "
                else:
                    iP = " " + df_user_value[i][4]

        if data_input_user == "":
            the_output = "Isian Belum Lengkap! Isi form diatas untuk mendapatkan rekomendasi produk skincare"
            return render_template('rekomendasi.html', data2=the_output)

        else:
            the_output2 = text_pre_input_two(data_input_user)
            print(the_output2)
            add_data = pd.DataFrame({'data': the_output2}, index=[-1])
            the_data = df_html

            # percabangan agar produk yang direkomendasikan sesuai dengan kategori
            if (data_input_user.find('Facial Wash') != -1):
                print('Facial Wash')
                th_idx = pd.concat(
                    [df_dataset[0:20], add_data]).reset_index(drop=True)
                the_product = the_data[0:20]
            elif (data_input_user.find('Face Toner') != -1):
                print('Face Toner')
                th_idx = pd.concat(
                    [df_dataset[20:40], add_data]).reset_index(drop=True)
                the_product = the_data[20:40]
            elif (data_input_user.find('Face Serum') != -1):
                print('Face Serum')
                th_idx = pd.concat(
                    [df_dataset[40:60], add_data]).reset_index(drop=True)
                the_product = the_data[40:60]
            elif (data_input_user.find('Moisturizer') != -1):
                print('Moisturizer')
                th_idx = pd.concat(
                    [df_dataset[60:80], add_data]).reset_index(drop=True)
                the_product = the_data[60:80]
            else:
                print('Sunscreen')
                th_idx = pd.concat(
                    [df_dataset[80:100], add_data]).reset_index(drop=True)
                the_product = the_data[80:100]

            # Stopword Bahasa Indonesia
            indo_stopword = set(stopwords.words('indonesian'))

            # Stopword Bahasa Inggris
            eng_stopword = set(stopwords.words('english'))

            # Data Stopword
            data_stopword = str(indo_stopword) + " " + str(eng_stopword)

            # TFIDF vectorizer
            vectorizer = TfidfVectorizer(
                analyzer='word', stop_words=list(data_stopword))
            tfidf_dataset1 = vectorizer.fit_transform(th_idx['data'])
            index, top5 = get_recommendations_product_two(tfidf_dataset1)
            print(index)
            
            data_res = []
            for i in index:
                for j in range(0,len(the_product)):
                    if i == j:
                        data_res.append(the_product[j])

            the_output = data_input_user + " " + str(the_output2)
            print(the_output)

            data2 = "Hallo Besti, Min Glowink akan rekomendasiin " + \
                str(tP) + " untuk jenis kulit " + str(tS) + \
                " yang membantu kamu merawat permasalahan kulit " + \
                pS + " karena produk ini memiliki kandungan " + iP
            return render_template('rekomendasi.html', data=data_res, data2=data2)

    # REKOMENDASI START
    else:
        the_output = "Isi form diatas untuk mendapatkan rekomendasi produk skincare"
        return render_template('rekomendasi.html', data2=the_output)

# DETAIL PAGE


@app.route('/detail/')
def detail():
    Id = request.args.get("id")
    df_html = df_product.values
    for i in range(0, len(df_html)):
        num = i+1
        if Id == str(num):
            the_data = df_html[i]
            return render_template('detail.html', data=the_data)


if __name__ == '__main__':
    app.run(debug=True)

#new