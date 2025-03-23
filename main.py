from flask import Flask, render_template, request, url_for, send_file, session, redirect, send_from_directory
from flask_session import Session
import pandas as pd
import joblib
import os
import secrets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import folium
import base64
from io import BytesIO
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

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
    # Jika metode GET
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
            
            # Membuat peta dan gambar untuk sebelum dan sesudah optimasi
            generate_rsrp_maps(df, session['analysis_id'])
            
            # Menghitung statistik RSRP
            rsrp_stats = calculate_rsrp_stats(df)
            
            # Render template hasil RSRP
            return render_template('rsrp_result.html', 
                                  site_id=site_id, 
                                  alamat_site=alamat_site,
                                  ringkasan=ringkasan,
                                  analysis_id=session['analysis_id'],
                                  rsrp_stats=rsrp_stats)
        else:
            return render_template("rsrp.html")
    else:
        # Jika tidak ada hasil atau sudah di-clear, tampilkan halaman RSRP info
        return render_template("rsrp.html")

# Fungsi untuk menghitung statistik RSRP
def calculate_rsrp_stats(df):
    # Statistik untuk nilai RSRP sebelum optimasi
    before = {
        'sangat_bagus': len(df[df['RSRP(dBm)'] >= -85]),
        'bagus': len(df[(df['RSRP(dBm)'] >= -95) & (df['RSRP(dBm)'] < -85)]),
        'normal': len(df[(df['RSRP(dBm)'] >= -100) & (df['RSRP(dBm)'] < -95)]),
        'buruk': len(df[(df['RSRP(dBm)'] >= -105) & (df['RSRP(dBm)'] < -100)]),
        'sangat_buruk': len(df[df['RSRP(dBm)'] < -105])
    }
    
    # Statistik untuk nilai RSRP setelah optimasi
    after = {
        'sangat_bagus': len(df[df['RSRP(dBm)_after'] >= -85]),
        'bagus': len(df[(df['RSRP(dBm)_after'] >= -95) & (df['RSRP(dBm)_after'] < -85)]),
        'normal': len(df[(df['RSRP(dBm)_after'] >= -100) & (df['RSRP(dBm)_after'] < -95)]),
        'buruk': len(df[(df['RSRP(dBm)_after'] >= -105) & (df['RSRP(dBm)_after'] < -100)]),
        'sangat_buruk': len(df[df['RSRP(dBm)_after'] < -105])
    }
    
    return {'before': before, 'after': after}

# Fungsi untuk membuat peta RSRP
def generate_rsrp_maps(df, analysis_id):
    # Pastikan direktori static/maps ada
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    
    # Definisikan warna untuk kelas RSRP
    colors = ['#dc3545', '#FFFF00', '#b6db8f', '#198754', '#0d6efd']  # Merah, Kuning, Hijau Muda, Hijau, Biru
    
    # Definisikan batasan nilai untuk kategori RSRP
    bounds = [-150, -105, -100, -95, -85, -44]
    
    # Hitung rata-rata koordinat untuk center peta
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Fungsi untuk menentukan warna berdasarkan nilai RSRP
    def get_color(rsrp):
        if rsrp >= -85:
            return '#0d6efd'  # Biru (Sangat Bagus)
        elif rsrp >= -95:
            return '#198754'  # Hijau (Bagus)
        elif rsrp >= -100:
            return '#b6db8f'  # Hijau Muda (Normal)
        elif rsrp >= -105:
            return '#FFFF00'  # Kuning (Buruk)
        else:
            return '#dc3545'  # Merah (Sangat Buruk)
    
    # Buat peta untuk RSRP sebelum optimasi
    m_before = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # Tambahkan titik data ke peta
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=get_color(row['RSRP(dBm)']),
            fill=True,
            fill_color=get_color(row['RSRP(dBm)']),
            fill_opacity=0.7,
            popup=f"RSRP: {row['RSRP(dBm)']} dBm"
        ).add_to(m_before)
    
    # Tambahkan legenda
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; background-color: white; 
                border: 2px solid grey; z-index: 9999; padding: 10px;">
        <p><b>RSRP (dBm)</b></p>
        <p><i class="fa fa-circle" style="color:#4e73df"></i> Sangat Bagus (≥ -85)</p>
        <p><i class="fa fa-circle" style="color:#1cc88a"></i> Bagus (≥ -95 s/d -85)</p>
        <p><i class="fa fa-circle" style="color:#f6c23e"></i> Normal (≥ -100 s/d -95)</p>
        <p><i class="fa fa-circle" style="color:#f8a339"></i> Buruk (≥ -105 s/d -100)</p>
        <p><i class="fa fa-circle" style="color:#e74a3b"></i> Sangat Buruk (≥ -150 s/d -105)</p>
    </div>
    '''
    #m_before.get_root().html.add_child(folium.Element(legend_html))
    
    # Simpan peta sebelum optimasi
    m_before.save(f"static/maps/rsrp_before_{analysis_id}.html")
    
    # Buat peta untuk RSRP setelah optimasi
    m_after = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # Tambahkan titik data ke peta
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=get_color(row['RSRP(dBm)_after']),
            fill=True,
            fill_color=get_color(row['RSRP(dBm)_after']),
            fill_opacity=0.7,
            popup=f"RSRP: {row['RSRP(dBm)_after']} dBm"
        ).add_to(m_after)
    
    # Tambahkan legenda
    #m_after.get_root().html.add_child(folium.Element(legend_html))
    
    # Simpan peta setelah optimasi
    m_after.save(f"static/maps/rsrp_after_{analysis_id}.html")

@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

@app.route("/sinr")
def sinr():
    # Jika metode GET
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
            
            # Membuat peta dan gambar untuk sebelum dan sesudah optimasi
            generate_sinr_maps(df, session['analysis_id'])
            
            # Menghitung statistik SINR
            sinr_stats = calculate_sinr_stats(df)
            
            # Render template hasil SINR
            return render_template('sinr_result.html', 
                                  site_id=site_id, 
                                  alamat_site=alamat_site,
                                  ringkasan=ringkasan,
                                  analysis_id=session['analysis_id'],
                                  sinr_stats=sinr_stats)
        else:
            return render_template("sinr.html")
    else:
        # Jika tidak ada hasil atau sudah di-clear, tampilkan halaman RSRP info
        return render_template("sinr.html")

# Fungsi untuk menghitung statistik SINR
def calculate_sinr_stats(df):
    # Statistik untuk nilai SINR sebelum optimasi
    before = {
        'sangat_bagus': len(df[df['SINR(dB)'] >= 10]),
        'bagus': len(df[(df['SINR(dB)'] >= 5) & (df['SINR(dB)'] < 10)]),
        'normal': len(df[(df['SINR(dB)'] >= 0) & (df['SINR(dB)'] < 5)]),
        'buruk': len(df[(df['SINR(dB)'] >= -5) & (df['SINR(dB)'] < 0)]),
        'sangat_buruk': len(df[df['SINR(dB)'] < -5])
    }
    
    # Statistik untuk nilai SINR setelah optimasi
    after = {
        'sangat_bagus': len(df[df['SINR(dB)_after'] >= 10]),
        'bagus': len(df[(df['SINR(dB)_after'] >= 5) & (df['SINR(dB)_after'] < 10)]),
        'normal': len(df[(df['SINR(dB)_after'] >= 0) & (df['SINR(dB)_after'] < 5)]),
        'buruk': len(df[(df['SINR(dB)_after'] >= -5) & (df['SINR(dB)_after'] < 0)]),
        'sangat_buruk': len(df[df['SINR(dB)_after'] < -5])
    }
    
    return {'before': before, 'after': after}

# Fungsi untuk membuat peta SINR
def generate_sinr_maps(df, analysis_id):
    # Pastikan direktori static/maps ada
    maps_dir = os.path.join('static', 'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    
    # Definisikan warna untuk kelas SINR
    colors = ['#dc3545', '#FFFF00', '#b6db8f', '#198754', '#0d6efd']  # Merah, Kuning, Hijau Muda, Hijau, Biru
    
    # Definisikan batasan nilai untuk kategori SINR
    bounds = [-20, -5, 0, 5, 10, 50]
    
    # Hitung rata-rata koordinat untuk center peta
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Fungsi untuk menentukan warna berdasarkan nilai SINR
    def get_color(sinr):
        if sinr >= 10:
            return '#0d6efd'  # Biru (Sangat Bagus)
        elif sinr >= 5:
            return '#198754'  # Hijau (Bagus)
        elif sinr >= 0:
            return '#b6db8f'  # Hijau Muda (Normal)
        elif sinr >= -5:
            return '#FFFF00'  # Kuning (Buruk)
        else:
            return '#dc3545'  # Merah (Sangat Buruk)
    
    # Buat peta untuk SINR sebelum optimasi
    m_before = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # Tambahkan titik data ke peta
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=get_color(row['SINR(dB)']),
            fill=True,
            fill_color=get_color(row['SINR(dB)']),
            fill_opacity=0.7,
            popup=f"SINR: {row['SINR(dB)']} dB"
        ).add_to(m_before)
    
    # Tambahkan legenda
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; background-color: white; 
                border: 2px solid grey; z-index: 9999; padding: 10px;">
        <p><b>RSRP (dBm)</b></p>
        <p><i class="fa fa-circle" style="color:#4e73df"></i> Sangat Bagus (≥ -85)</p>
        <p><i class="fa fa-circle" style="color:#1cc88a"></i> Bagus (≥ -95 s/d -85)</p>
        <p><i class="fa fa-circle" style="color:#f6c23e"></i> Normal (≥ -100 s/d -95)</p>
        <p><i class="fa fa-circle" style="color:#f8a339"></i> Buruk (≥ -105 s/d -100)</p>
        <p><i class="fa fa-circle" style="color:#e74a3b"></i> Sangat Buruk (≥ -150 s/d -105)</p>
    </div>
    '''
    #m_before.get_root().html.add_child(folium.Element(legend_html))
    
    # Simpan peta sebelum optimasi
    m_before.save(f"static/maps/sinr_before_{analysis_id}.html")
    
    # Buat peta untuk RSRP setelah optimasi
    m_after = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # Tambahkan titik data ke peta
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=get_color(row['SINR(dB)_after']),
            fill=True,
            fill_color=get_color(row['SINR(dB)_after']),
            fill_opacity=0.7,
            popup=f"SINR: {row['SINR(dB)_after']} dB"
        ).add_to(m_after)
    
    # Tambahkan legenda
    #m_after.get_root().html.add_child(folium.Element(legend_html))
    
    # Simpan peta setelah optimasi
    m_after.save(f"static/maps/sinr_after_{analysis_id}.html")

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