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
    'D:\TA CODING\SISTEM REKOMENDASI - ELIZA\Recommender-System-for-Skincare-Products-by-Eliza\FlaskApp\data\DataProduk3.csv')

df_user = pd.read_csv(
    'D:\TA CODING\SISTEM REKOMENDASI - ELIZA\Recommender-System-for-Skincare-Products-by-Eliza\FlaskApp\data\datasetPengguna.csv')

df_dataset = pd.read_csv(
    'D:\TA CODING\SISTEM REKOMENDASI - ELIZA\Recommender-System-for-Skincare-Products-by-Eliza\FlaskApp\data\datasetProduct.csv')

app = Flask(__name__)

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


def get_recommendations_product2(user_input):

    # Stopword Bahasa Indonesia
    indo_stopword = set(stopwords.words('indonesian'))

    # Stopword Bahasa Inggris
    eng_stopword = set(stopwords.words('english'))

    # Data Stopword
    data_stopword = str(indo_stopword) + " " + str(eng_stopword)

    # TFIDF
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=list(data_stopword))
    tfidf_dataset = vectorizer.fit_transform(df_dataset['data'])

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

    # Cetak query yang telah dilakukan stemming
    print(the_query)
    # Melakukan perhitungan TFIDF terhadap inputan pengguna kepada keseluruhan data kata
    embed_query = vectorizer.transform(the_query)

    # besar matriks TFIDF inputan pengguna
    print(embed_query.shape)

    # Mencari tingkat kemiripan antara user_input dengan dataset
    cos_sim = cosine_similarity(embed_query, tfidf_dataset)

    # Membuat dataframe untuk hasil similaritas
    df_cosim = pd.DataFrame(data=cos_sim)
    # Menampilkan dataframe
    # display(df_cosim)

    # Mencari rata rata dari nilai kemiripan antara inputan pengguna dengan data produk
    cos_sim_a = np.mean(cos_sim, axis=0)
    # Membuat dataframe hasil rata rata
    df_cosim_a = pd.DataFrame(data=cos_sim_a, columns=[
                              "cosine similarity score"])
    # Menampilkan hasil nilai kemiripan
    # display(df_cosim_a)

    # Mencari 5 teratas nilai kemiripan tertinggi
    top5 = df_cosim_a.sort_values(
        "cosine similarity score", ascending=False).head(5)
    # Menampilkan 5 nilai kemiripan teratas
    # display(top5)
    # Mencari index dari 5 nilai kemiripan teratas
    index = list(top5.index.values)

    # Mencetak index
    print(index)

    return index


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
        for i in range(0, len(df_user_value)):
            if df_user_value[i][0] == type_product and df_user_value[i][1] == type_skin and df_user_value[i][2] == problem_skin:
                print(df_user_value[i][0] + " " + df_user_value[i][1] +
                      " " + df_user_value[i][2] + " " + df_user_value[i][3])
                data_input_user = df_user_value[i][0] + " " + df_user_value[i][1] + \
                    " " + df_user_value[i][2] + " " + df_user_value[i][3]

        if data_input_user == "":
            the_output = "Isi form diatas untuk mendapatkan rekomendasi produk skincare"
            return render_template('rekomendasi.html', data2=the_output)

        else:
            the_output2 = get_recommendations_product2(data_input_user)

            data_res = []
            for value in df_html:
                for i in the_output2:
                    if value[0] == i:
                        data_res.append(value)

            the_output = data_input_user + " " + str(the_output2)
            return render_template('rekomendasi.html', data=data_res, data2=data_input_user)

    # REKOMENDASI START
    else:
        the_output = "Isi form diatas untuk mendapatkan rekomendasi produk skincare"
        return render_template('rekomendasi.html', data2=the_output)


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
