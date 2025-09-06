# 👁️ Cataract Detection System

Sistem deteksi katarak menggunakan AI untuk menganalisis gambar mata dan memberikan prediksi dengan tingkat kepercayaan.

## ✨ Fitur

- 🔍 **Deteksi Otomatis**: Upload gambar mata dan dapatkan hasil analisis instan
- 📊 **Visualisasi Hasil**: Chart probabilitas dan confidence gauge
- ⚙️ **Pengaturan Fleksibel**: Atur threshold confidence sesuai kebutuhan
- 🔧 **Technical Details**: Informasi detail untuk debugging dan analisis
- 📱 **Responsive UI**: Interface yang user-friendly dan modern

## 🚀 Quick Start

1. **Clone repository**
```bash
git clone https://github.com/your-username/cataract-detection.git
cd cataract-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Siapkan model**
   - Letakkan file model (.h5) di root folder
   - Pastikan nama file sesuai dengan `MODEL_PATHS` di `config.py`

4. **Jalankan aplikasi**
```bash
streamlit run main.py
```

## 📁 Struktur Project

```
cataract_detection/
├── main.py              # Aplikasi utama
├── config.py            # Konfigurasi sistem
├── model_utils.py       # Loading dan prediksi model
├── ui_components.py     # Komponen UI
├── visualization.py     # Chart dan grafik
├── styles.py           # CSS styling
├── utils.py            # Helper functions
├── requirements.txt    # Dependencies
└── README.md          # Dokumentasi
```

## 🔧 Konfigurasi

Edit `config.py` untuk menyesuaikan:
- Path model files
- Default settings
- UI configuration

## 📋 Requirements

- Python 3.7+
- Streamlit
- TensorFlow
- Keras
- NumPy
- Pillow
- Matplotlib
- Plotly

## 🎯 Cara Penggunaan

1. Buka aplikasi di browser
2. Upload gambar mata (JPG/PNG)
3. Tunggu proses analisis
4. Lihat hasil prediksi dan visualisasi

## ⚠️ Important Notes

- **Medical Disclaimer**: Sistem ini hanya alat bantu diagnosis, bukan pengganti konsultasi medis
- **Model Files**: Pastikan model sudah di-train dan kompatibel
- **Image Quality**: Gunakan gambar mata yang jelas dan berkualitas baik

## 🐛 Troubleshooting

### Model Loading Issues
- Pastikan file model ada dan tidak corrupt
- Cek kompatibilitas versi TensorFlow
- Gunakan format SavedModel jika .h5 bermasalah

### Deployment Issues  
- Verifikasi semua dependencies terinstall
- Cek ukuran file model (batasan platform)
- Pastikan environment variables sesuai

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

1. Fork the project
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📞 Support

Jika ada pertanyaan atau issue, silakan buat GitHub Issue atau hubungi maintainer.
