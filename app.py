
from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import tempfile

app = Flask(__name__)

# ✅ Threshold ketat — cosine distance < 0.3 = wajah sama
# Sebelumnya 0.5 — terlalu longgar, wajah orang lain bisa lolos
FACE_THRESHOLD = 0.3

# ✅ Minimum confidence yang diterima = 70%
# Confidence di bawah ini langsung ditolak meski verified=True
MIN_CONFIDENCE = 70.0

# Preload model saat startup agar tidak load ulang setiap request
print("Loading Facenet model...")
DeepFace.build_model("Facenet")
print("Model loaded! Siap menerima request.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'ok',
        'message': 'Face Recognition API berjalan!'
    })

@app.route('/verify', methods=['POST'])
def verify_wajah():
    try:
        if 'foto_selfie' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Foto selfie tidak ditemukan!'
            }), 400

        if 'foto_referensi' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Foto referensi tidak ditemukan!'
            }), 400

        foto_selfie    = request.files['foto_selfie']
        foto_referensi = request.files['foto_referensi']

        tmp_selfie    = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tmp_referensi = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

        foto_selfie.save(tmp_selfie.name)
        foto_referensi.save(tmp_referensi.name)

        tmp_selfie.close()
        tmp_referensi.close()

        path_selfie    = tmp_selfie.name
        path_referensi = tmp_referensi.name

        try:
            result = DeepFace.verify(
                img1_path         = path_selfie,
                img2_path         = path_referensi,
                model_name        = 'Facenet',
                enforce_detection = True,
                distance_metric   = 'cosine',
                threshold         = FACE_THRESHOLD,  # ✅ pakai konstanta
            )

            distance   = result['distance']
            confidence = round((1 - distance) * 100, 2)

            # ✅ Double check — verified dari DeepFace DAN confidence minimum
            verified = result['verified'] and confidence >= MIN_CONFIDENCE

            print(f"Verify result: verified={verified}, confidence={confidence}%, distance={round(distance, 4)}")

            if verified:
                return jsonify({
                    'success'   : True,
                    'verified'  : True,
                    'confidence': confidence,
                    'message'   : f'Wajah cocok! ({confidence}%)'
                })
            else:
                # Beri pesan berbeda tergantung penyebab penolakan
                if confidence < MIN_CONFIDENCE:
                    pesan = f'Wajah tidak cocok! Tingkat kecocokan hanya {confidence}% (minimum {MIN_CONFIDENCE}%)'
                else:
                    pesan = f'Wajah tidak cocok! Distance: {round(distance, 4)}'

                return jsonify({
                    'success'   : False,
                    'verified'  : False,
                    'confidence': confidence,
                    'message'   : pesan
                })

        except ValueError as e:
            error_msg = str(e)
            print(f"ERROR VERIFY (ValueError): {error_msg}")

            # Pesan yang lebih user-friendly
            if 'Face could not be detected' in error_msg:
                pesan = 'Wajah tidak terdeteksi! Pastikan pencahayaan cukup dan wajah terlihat jelas.'
            else:
                pesan = 'Wajah tidak terdeteksi! Coba lagi.'

            return jsonify({
                'success' : False,
                'verified': False,
                'confidence': 0,
                'message' : pesan
            }), 400

        except Exception as e:
            print(f"ERROR VERIFY GENERAL: {str(e)}")
            return jsonify({
                'success' : False,
                'verified': False,
                'confidence': 0,
                'message' : 'Terjadi kesalahan saat verifikasi wajah.'
            }), 500

        finally:
            if os.path.exists(path_selfie):
                os.remove(path_selfie)
            if os.path.exists(path_referensi):
                os.remove(path_referensi)

    except Exception as e:
        print(f"ERROR GLOBAL: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/detect', methods=['POST'])
def detect_wajah():
    try:
        if 'foto' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Foto tidak ditemukan!'
            }), 400

        foto     = request.files['foto']
        tmp_foto = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        foto.save(tmp_foto.name)
        tmp_foto.close()
        path_foto = tmp_foto.name

        try:
            hasil        = DeepFace.extract_faces(
                img_path          = path_foto,
                enforce_detection = True
            )
            jumlah_wajah = len(hasil)

            if jumlah_wajah == 0:
                return jsonify({
                    'success' : False,
                    'detected': False,
                    'message' : 'Wajah tidak terdeteksi!'
                }), 400

            if jumlah_wajah > 1:
                return jsonify({
                    'success' : False,
                    'detected': False,
                    'message' : 'Terdeteksi lebih dari 1 wajah!'
                }), 400

            return jsonify({
                'success' : True,
                'detected': True,
                'message' : 'Wajah berhasil terdeteksi!'
            })

        finally:
            if os.path.exists(path_foto):
                os.remove(path_foto)

    except Exception as e:
        return jsonify({
            'success' : False,
            'detected': False,
            'message' : str(e)
        }), 500


if __name__ == '__main__':
    app.run(
        host         = '0.0.0.0',
        port         = 5000,
        debug        = True,
        use_reloader = False
    )