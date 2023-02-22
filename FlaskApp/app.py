from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def main():
  return render_template('katalog.html')

@app.route('/rekomendasi')
def rekomendasi():
  return render_template('rekomendasi.html')

@app.route('/detail')
def katalog():
  return render_template('detail.html')

if __name__ == '__main__':
  app.run(debug=True)