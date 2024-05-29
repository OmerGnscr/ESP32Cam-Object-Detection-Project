import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import urllib.request
import logging
from datetime import datetime
from playsound import playsound

# CameraWebServer'inin IP adresi
url = '' # Buraya IP adresi yazılacak (Örnek: 'http://192.168.43.59')

# Model dosyalarının bulundugu konum
model_konum1 = "" # Buraya model dosyasının konumu yazılacak (Örnek: "C:\\Users\\Documents\\multi-proje\\yolo_models\\yvm-model.pt")
model1 = YOLO(model_konum1)

model_konum2 = "" # Buraya model dosyasının konumu yazılacak (Örnek: "C:\\Users\\Documents\\multi-proje\\yolo_model\\yolov8l.pt")
model2 = YOLO(model_konum2)

# Ses dosyasi
sound_file = "" # Buraya ses dosyasının konumu yazılacak (Örnek: "C:\\Users\\Documents\\multi-proje\\nesne-taninmadi.mp3

# Log konfigurasyonu
start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dosya = f'object_detection_log_{start_time}.txt'
logging.basicConfig(filename=log_dosya, level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%d.%m.%Y - %H:%M')

def CamAyar():
    # Goruntunun boyutunu belirler. (Default: 640x480)
    # (1600x1200: 13, 1280x720: 11, 800x600: 9, 640:480: 8)
    boyut = 11
    urllib.request.urlopen(url + "/control?var=framesize&val={}".format(boyut))

    # Goruntunu kalitesini belirler. (Default: 34 - Ortalama)
    # (En iyi: 4 - En kotu: 63)
    kalite = 15
    urllib.request.urlopen(url + "/control?var=quality&val={}".format(kalite))

def main():

    CamAyar()

    # Objelerin etrafındaki kutular
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        # ESP2Cam'den görüntüyü alır
        try:
            # IP adresine istek (request) gönderir ve veriyi (goruntuyu) alır.
            response = urllib.request.urlopen(url + "/capture") 
            
            # Alınan yaniti (response) byte array'i olarak depolar.
            image_data = np.array(bytearray(response.read()), dtype=np.uint8)

            # Byte array'i cv2'nin isleyebilecegi sekilde goruntuye donusturur.
            image = cv2.imdecode(image_data, -1) 

        except Exception as e:
            print(f"Goruntu indirilemedi: {e}")
            continue  # görüntüyü alamazsa diğer görüntüye geçer

        # Object Detection modelini calistirir
        result1 = model1(image, agnostic_nms=True)[0]
        result2 = model2(image, agnostic_nms=True)[0]

        # sv (Supervision) kutuphanesinin kullanabilecegi formata donusturur
        detections1 = sv.Detections.from_yolov8(result1)
        detections2 = sv.Detections.from_yolov8(result2)

        # Bazi objelerle cakistigi icin model2'den cikarilacak classlari tutan list
        exclude_classes = ["kite", "refrigerator", "sports ball", "frisbee"]

        detections2_filtered = [
            detection for detection in detections2
            if model2.model.names[detection[2]] not in exclude_classes
        ]
        
        # Kac tane oldugunu saymak icin gerekli
        counts = {}
        play_sound = False

        # Classların sayısını tutmak için
        for _, confidence, class_id, _ in detections1:
            if confidence >= 0.5:  # Confidence %50'den yuksekse
                class_name = model1.model.names[class_id]
                if class_name not in counts:
                    counts[class_name] = 1
                else:
                    counts[class_name] += 1

        for _, confidence, class_id, _ in detections2_filtered:
            if confidence >= 0.5: 
                class_name = "tanimlanamayan nesne"  # Model2'nin tum classlarini "tanimlanamayan nesne" olarak degistir
                if class_name not in counts:
                    counts[class_name] = 1
                else:
                    counts[class_name] += 1
                # Tanimlanamayan nesne algiladiginda sesi oynatmak icin
                play_sound = True

        # Log çıktısı almak için
        if counts:
            log_message = ", ".join([f"{count} {class_name}" for class_name, count in counts.items()])
            logging.info(log_message)

        # Sesi oynatmak için
        if play_sound:
            playsound(sound_file)

        # Her bir class için ekrana text yazdırır
        y_offset = 50
        for i, (class_name, count) in enumerate(counts.items()):
            # Ekrandaki yaziyi dikey yazmak icin (normalde yan yana)
            y = y_offset + i * 50

            # Class isimlerinin labels1 ve labels2 listelerine atar
            # ve model2 icin etiketleri "tanimlanamadi" olarak degistirir
            labels1 = [
                f"{model1.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections1
                if model1.model.names[class_id] == class_name and confidence >= 0.5
            ]
            labels2 = [
                f"tanimlanamadi {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections2_filtered
                if confidence >= 0.5
            ]

            # labels listlerinden classlari alir ve kutucuga yazdirir
            image = box_annotator.annotate(
                scene=image, 
                detections=detections1, 
                labels=labels1
            )
            image = box_annotator.annotate(
                scene=image, 
                detections=detections2_filtered, 
                labels=labels2
            )

            # Confidence %50'den fazla olanlari ekrana yazdir
            if count >= 0.5:
                cv2.putText(image, f"{count} tane {class_name} bulundu!", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Multidisciplinary Project", image) # Pencerenin adi

        if cv2.waitKey(30) == 27: # ESC tusuna basmak programi durdurur
            break

    # Sonlandırır
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
