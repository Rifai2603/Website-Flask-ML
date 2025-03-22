from flask import Flask, render_template, request, url_for, send_file, session, redirect
from flask_session import Session
import pandas as pd
import joblib
import os
import secrets


app = Flask(__name__)
# Tambahkan secret key untuk session
app.secret_key = '260303'

# Memuat semua model dan preprocessing
scaler = joblib.load('models/scaler.pkl')
kmeans = joblib.load('models/kmeans.pkl')
rf_model_c = joblib.load('models/rf_model_c.pkl')
encoder = joblib.load('models/encoder.pkl')
rf_model_r = joblib.load('models/rf_model_r.pkl')

# Konfigurasi untuk session server-side
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "flask_session"  # Direktori untuk menyimpan file session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)  # Inisialisasi session server-side

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"  # Folder baru untuk menyimpan hasil analisis

# Buat direktori jika belum ada
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET","POST"])
def upload_file():
    # Jika metode GET
    if request.method == "GET":
        # Periksa apakah ada hasil analisis di session
        if session.get('analysis_id') and not session.get('clear_results', False):
            # Jika ada hasil dan belum di-clear, tampilkan halaman hasil
            site_id = session.get('site_id', '')
            alamat_site = session.get('alamat_site', '')
            ringkasan = session.get('ringkasan', '')
            
            # Membaca hasil dari file
            result_filepath = os.path.join(app.config["RESULT_FOLDER"], f"{session['analysis_id']}.csv")
            if os.path.exists(result_filepath):
                df = pd.read_csv(result_filepath)
                data_table = df[["Longitude", "Latitude", "RSRP(dBm)", "SINR(dB)", 
                               "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)", 
                               "Jenis Rekomendasi Optimasi"]].to_html(classes="table", index=False)
            else:
                data_table = "<p>Data tidak tersedia</p>"
            
            return render_template('result.html', 
                                  site_id=site_id, 
                                  alamat_site=alamat_site,
                                  ringkasan=ringkasan, 
                                  data=data_table)
        else:
            # Jika tidak ada hasil atau sudah di-clear, tampilkan form upload
            session['clear_results'] = False
            return render_template('upload.html')

    if request.method == "POST":
        site_id = request.form.get("siteId", "")
        alamat_site = request.form.get("alamatSite", "")

        errors = {}
        if not site_id:
            errors["siteId"] = "Site ID wajib diisi"
        if not alamat_site:
            errors["alamatSite"] = "Alamat Site wajib diisi"
        
        if errors:
            return render_template("upload.html", errors=errors, 
                                  site_id=site_id, 
                                  alamat_site=alamat_site)
        
        file = request.files["file"]

        if file.filename == "":
            return render_template("upload.html", error="Tidak ada file yang dipilih!")
        
        if file:
            if not file.filename.endswith('.csv'):
                return render_template("upload.html", error="File harus berformat CSV")

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Load data dari file CSV yang diunggah
            df = pd.read_csv(filepath)

            # **Preprocessing Data**
            df = df[['Longitude', 'Latitude', 'LTE PCC Serving RSRP(dBm)', 'LTE PCC Serving SINR(dB)', 'LTE DL App Throughput(kbps)', 'LTE UL App Throughput(kbps)']]
            df = df.rename(columns={
                'LTE PCC Serving RSRP(dBm)': 'RSRP(dBm)',
                'LTE PCC Serving SINR(dB)': 'SINR(dB)',
                'LTE DL App Throughput(kbps)': 'Throughput Downlink(Kbps)',
                'LTE UL App Throughput(kbps)': 'Throughput Uplink(Kbps)'
            })

            # Hapus baris yang memiliki Longitude atau Latitude kosong
            df = df.dropna(subset=["Longitude", "Latitude"])

            # Menghitung median setiap kolom parameter
            rsrp_median = df['RSRP(dBm)'].median(skipna=True)
            sinr_median = df['SINR(dB)'].median(skipna=True)
            dl_throughput_median = df['Throughput Downlink(Kbps)'].median(skipna=True)
            ul_throughput_median = df['Throughput Uplink(Kbps)'].median(skipna=True)

            # Isi missing values pada RSRP, SINR, Throughput dengan median
            df['RSRP(dBm)'] = df['RSRP(dBm)'].fillna(rsrp_median)
            df['SINR(dB)'] = df['SINR(dB)'].fillna(sinr_median)
            df['Throughput Downlink(Kbps)'] = df['Throughput Downlink(Kbps)'].fillna(dl_throughput_median)
            df['Throughput Uplink(Kbps)'] = df['Throughput Uplink(Kbps)'].fillna(ul_throughput_median)

            # **Normalisasi Data**
            df[["RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)"]] = scaler.transform(
                df[["RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)"]]
            )

            # **Prediksi Jenis Rekomendasi Optimasi**
            df["Jenis Rekomendasi Optimasi"] = rf_model_c.predict(df[["Longitude", "Latitude", "RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)"]])

            # **One-Hot Encoding**
            encoded = encoder.transform(df[["Jenis Rekomendasi Optimasi"]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Jenis Rekomendasi Optimasi"]))
            df = df.join(encoded_df)

            # **Clustering**
            df["cluster"] = kmeans.predict(df[["Longitude", "Latitude", "RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)"]])

            # **Prediksi Parameter Setelah Optimasi**
            fitur_regresi = df[["Longitude", "Latitude", "RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)", "cluster"] + list(encoded_df.columns)]
            df_prediksi = rf_model_r.predict(fitur_regresi)

            # Simpan hasil prediksi ke dataframe
            df["RSRP(dBm)_after"] = df_prediksi[:, 0]
            df["SINR(dB)_after"] = df_prediksi[:, 1]
            df["Throughput Downlink(Kbps)_after"] = df_prediksi[:, 2]
            df["Throughput Uplink(Kbps)_after"] = df_prediksi[:, 3]

            # **Inverse Transform (Kembali ke Skala Asli)**
            df[["RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)"]] = scaler.inverse_transform(
                df[["RSRP(dBm)", "SINR(dB)", "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)"]]
            )

            df[["RSRP(dBm)_after", "SINR(dB)_after", "Throughput Downlink(Kbps)_after", "Throughput Uplink(Kbps)_after"]] = scaler.inverse_transform(
                df[["RSRP(dBm)_after", "SINR(dB)_after", "Throughput Downlink(Kbps)_after", "Throughput Uplink(Kbps)_after"]]
            )

            # Temukan jenis rekomendasi optimasi yang paling banyak muncul
            rekomendasi_terbanyak = df["Jenis Rekomendasi Optimasi"].value_counts().idxmax()
            
            # Buat kalimat ringkasan
            ringkasan = f"Site ID {site_id} yang beralamat di {alamat_site} diperlukan optimasi {rekomendasi_terbanyak}"

            # Buat ID unik untuk analisis ini
            analysis_id = f"{site_id}_{secrets.token_hex(8)}"

            # Simpan hasil ke CSV untuk ditampilkan di halaman web
            result_filepath = os.path.join(app.config["RESULT_FOLDER"], f"{analysis_id}.csv")
            df.to_csv(result_filepath, index=False)

            # Simpan hanya metadata di session
            session['analysis_id'] = analysis_id
            session['site_id'] = site_id
            session['alamat_site'] = alamat_site
            session['ringkasan'] = ringkasan
            session['clear_results'] = False

            return redirect(url_for('upload_file'))

#Menghapus analisa kembali seperti awal (session jadi kosong)
@app.route("/clear_analysis")
def clear_analysis():
    # Set flag untuk menampilkan form upload
    session['clear_results'] = True

    # Hapus session data yang tidak diperlukan
    for key in ['analysis_id', 'site_id', 'alamat_site', 'ringkasan']:
        session.pop(key, None)

    return redirect(url_for('upload_file'))

@app.route("/download/<site_id>")
def download_result(site_id):
    if 'analysis_id' in session:
        download_filepath = os.path.join(app.config["RESULT_FOLDER"], f"{session['analysis_id']}.csv")
    else:
        return "Tidak ada data untuk diunduh", 404

    if os.path.exists(download_filepath):
        # Baca file CSV original
        df = pd.read_csv(download_filepath)
        
        # Pilih hanya kolom-kolom yang diinginkan
        columns_to_keep = ["Longitude", "Latitude", "RSRP(dBm)", "SINR(dB)", 
                          "Throughput Downlink(Kbps)", "Throughput Uplink(Kbps)", 
                          "Jenis Rekomendasi Optimasi", "RSRP(dBm)_after", "SINR(dB)_after", "Throughput Downlink(Kbps)_after" , "Throughput Uplink(Kbps)_after" ]
        filtered_df = df[columns_to_keep]
        
        # Buat file CSV sementara dengan kolom terpilih
        temp_filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_download_{site_id}.csv")
        filtered_df.to_csv(temp_filepath, index=False)
        
        # Download file
        return_value = send_file(temp_filepath, as_attachment=True, download_name=f"Hasil_Analisis_Site {site_id}.csv")
        
        return return_value
    else:
        return "File tidak ditemukan", 404

@app.route("/rsrp")
def rsrp():
    return render_template("rsrp.html")

@app.route("/sinr")
def sinr():
    return render_template("sinr.html")

@app.route("/throughput_dl")
def throughput_dl():
    return render_template("throughput_dl.html")

@app.route("/throughput_ul")
def throughput_ul():
    return render_template("throughput_ul.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)