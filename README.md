# ESP32Cam-Object-Detection-Project

This project is a final year group project involving object detection using an ESP32CAM. The system captures images from the ESP32CAM, processes them with a YOLOv8 object detection models, and displays the detected objects.

### General Overview

In this project, we use a custom dataset for object detection with an ESP32CAM. The objects in our dataset include mavi_kare, mavi_daire, mavi_ucgen, yesil_ucgen, yesil_kare, and yesil_daire. Our goal is to identify only these specific objects, and label any other objects as "tanimlanamayan nesne" (unidentified object). To achieve this, we employ two YOLOv8 models: Model1 and Model2.
Model1: This model is trained on our custom dataset and is responsible for detecting the objects we are specifically interested in.
Model2: This model is a more general object detection model that identifies objects outside our custom dataset. We use it to detect and label any other objects as "tanimlanamayan nesne."

Also, an alert sound and a logging function is added to the project. When the "tanimlanamayan nesne." appears on the screen, an alert sound plays. On the other hand, logging function is used to log every detected object in the image in real-time in a text file on our computer.

![](https://github.com/OmerGnscr/ESP32Cam-Object-Detection-Project/blob/main/images/detections.png)

### Explaination of the code

These are the necessary libraries that are used in project.

```
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import urllib.request
import logging
from datetime import datetime
from playsound import playsound

```

- cv2: OpenCV library for image processing and display.
- YOLO: From the Ultralytics library, used for object detection models.
- supervision: A library for annotating detections on images.
- numpy: A library for numerical operations, used here for image data processing.
- urllib.request: To make HTTP requests to the ESP32CAM web server.
- logging: For logging detected objects with timestamps.
- datetime: For handling date and time in logging.
- playsound: To play sound alerts when an unidentified object is detected.

```python
url = '' # Buraya IP adresi yazılacak (Örnek: '<http://192.168.43.59>')

# Model dosyalarının bulundugu konum
model_konum1 = "" # Buraya model dosyasının konumu yazılacak (Örnek: "C:\\\\Users\\\\Documents\\\\multi-proje\\\\yolo_models\\\\yvm-model.pt")
model1 = YOLO(model_konum1)

model_konum2 = "" # Buraya model dosyasının konumu yazılacak (Örnek: "C:\\\\Users\\\\Documents\\\\multi-proje\\\\yolo_model\\\\yolov8l.pt")
model2 = YOLO(model_konum2)

# Ses dosyasi
sound_file = "" # Buraya ses dosyasının konumu yazılacak (Örnek: "C:\\Users\\Documents\\multi-proje\\nesne-taninmadi.mp3")

# Log konfigurasyonu
start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dosya = f'object_detection_log_{start_time}.txt'
logging.basicConfig(filename=log_dosya, level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%d.%m.%Y - %H:%M')
```

- **url**: IP Address of ESP32. This is where the webserver is located.
- **model1_konum model2_konum**: These variables hold the location of the models in the computer.
- **model1 and model2:** Yolo v8 models.
- **sound_file:** A sound file that runs when the unidentified objects are detected.
- **start_time , log_dosya , basicConfig():** These are used to configuring the logging function.

![](https://github.com/OmerGnscr/ESP32Cam-Object-Detection-Project/blob/main/images/WebServer.png)

```python
def CamAyar():
    # Goruntunun boyutunu belirler. (Default: 640x480)
    # (1600x1200: 13, 1280x720: 11, 800x600: 9, 640:480: 8)
    boyut = 11
    urllib.request.urlopen(url + "/control?var=framesize&val={}".format(boyut))

    # Goruntunu kalitesini belirler. (Default: 34 - Ortalama)
    # (En iyi: 4 - En kotu: 63)
    kalite = 15
    urllib.request.urlopen(url + "/control?var=quality&val={}".format(kalite))

```

CamAyar() function is used to setting up the camera options before the object detection starts. “boyut” variable is used tothe setting up the size of the camera and “kalite” variable is used to  adjust the quality of an image that is taken from the web server on ESP32. In order to adjust this settings, simply sending and request to the certain endpoints on the server get the job done.

```python
def main():

    CamAyar()

    # Objelerin etrafındaki kutular
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
```

The main function starts by calling CamAyar() function to set up the camera. Then it initializes a BoxAnnotator() function from supervision for drawing boxes around detected objects with specified thickness and text scale.

```python
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
```

This loop continuously captures images from the ESP32CAM. It sends a request to /capture endpoint to get the image and formats the image to be able to used by OpenCV. If an error occurs during the operation, it prints out “Goruntu indirilemedi” and continues to next iteration.

```python
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

```

This part of code runs object detection on the captured image using both YOLO models.

“result1” and “result2” variables store the detection results from the two models. Then, the results are converted into a format that is compatible with SuperVision library. An object might be detected by both models at the same time. In order to avoid duplicates, some certain classes from model2 are excluded. And, detections2_filtered filters out the excluded classes from the second model's detections.
The parameter agnostic_nms refers to "class-agnostic non-maximum suppression." This concept is 
used in object detection to refine the set of detected bounding boxes by
 eliminating redundant boxes. When an object detection model makes predictions, it often detects the same object multiple times with slightly different bounding boxes. NMS helps to remove these redundant detections and keep only the most confident ones. NMS algorithm will treat all detected objects as if they belong to the same class. This can be useful in situations where you want to ensure that overlapping detections of different classes are also suppressed.

```python
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
```

This part of code is used to counting the detected objects. For each detection, it, first, checks if the confidence is above 50%. If so, it adds to count for that class.

If an unidentified (”tanimlanamayan nesne”) is detected, it sets “play__sound” variable to True.

```python
        # Log çıktısı almak için
        if counts:
            log_message = ", ".join([f"{count} {class_name}" for class_name, count in counts.items()])
            logging.info(log_message)

        # Sesi oynatmak için
        if play_sound:
            playsound(sound_file)
```

If there are any counts (detected objects), it logs it to a text file. 

If play_sound is True, it plays the sound file.

![](https://github.com/OmerGnscr/ESP32Cam-Object-Detection-Project/blob/main/images/logs.png)

```python
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
                for _, confidence,

```

This part of code is used to display the detected object’s names on the screen. The for loop retrieves each objects’ class name and count. “labels1” and “labels2” are lists that hold the labels for objects detected by Model1 and Model2, respectively. 
The image is annotated with bounding boxes and labels for each detected object using the BoxAnnotator from the supervision library.

```python
	
            cv2.putText(image, f"{count} tane {class_name} bulundu!", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Multidisciplinary Project", image) # Pencerenin adi

        if cv2.waitKey(30) == 27: # ESC tusuna basmak programi durdurur
            break

    # Sonlandırır
    cv2.destroyAllWindows()
```

putText() function displays a text message about the objects that are detected.

imshow() is used to set the window name as "Multidisciplinary Project".

The loop continues to process and display images until the user presses the ESC key.
