import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
from PIL import Image, ImageTk

# Загрузка модели YOLO
model = YOLO("best.pt")


# Переменные для отслеживания предупреждений и столкновений
warnings = 0
collisions = 0

#функцияя предугадывания траектории
def predict_position(track, future_time, fps):
    if len(track) < 2:
        return track[-1]

    N = min(len(track), 25)
    track = np.array(track[-N:])

    times = np.arange(-N + 1, 1)

    A = np.vstack([times, np.ones(len(times))]).T
    k_x, b_x = np.linalg.lstsq(A, track[:, 0], rcond=None)[0]
    k_y, b_y = np.linalg.lstsq(A, track[:, 1], rcond=None)[0]

    future_frames = future_time * fps
    future_x = k_x * future_frames + b_x
    future_y = k_y * future_frames + b_y

    return future_x, future_y

warn_list = []
current_frame=None
def last_fut(x1_l,x2_l,y1_l,y2_l,x1_fut,x2_fut,y1_fut,y2_fut):
    
    global warnings,warn_list
    if x1_l<x1_fut:
        if y1_l<y1_fut:
            if x2_l <x2_fut:
                if y2_l < y2_fut:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_l,x2_fut)])):
                        if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_l,y2_fut)])):
                            warn_list.append(1)
                        else:
                            warn_list.append(0)
                else:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_l,x2_fut)])):
                        if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_fut,y2_l)])):
                            warn_list.append(1)
                        else:
                            warn_list.append(0)
            else:
                if y2_l < y2_fut:
                        if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_fut,x2_l)])):
                            if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_l,y2_fut)])):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
                else:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_fut,x2_l)])):
                            if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_fut,y2_l)])):
                                warn_list.append(1)
                            else:
                                warn_list.append(0)
        else:
            if x2_l <x2_fut:
                if y2_l < y2_fut:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_l,x2_fut)])):
                        if len(set([i for i in range(y1_l,y1_fut)]))&set([i for i in range(y2_l,y2_fut)]):
                            warn_list.append(1)
                        else:
                            warn_list.append(0)
                else:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_l,x2_fut)])):
                        if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_fut,y2_l)])):
                            warn_list.append(1)
                        else:
                            warn_list.append(0)
            else:
                if y2_l < y2_fut:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_fut,x2_l)])):
                        if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_l,y2_fut)])):
                            warn_list.append(1)
                        else:
                            warn_list.append(0)
                else:
                    if len(set([i for i in range(x1_l,x1_fut)])&set([i for i in range(x2_l,x2_fut)])):
                        if len(set([i for i in range(y1_l,y2_fut)])&set([i for i in range(y2_fut,y2_l)])):
                            warn_list.append(1)
                        else:
                            warn_list.append(0)
    else:
        if x1_l<x1_fut:
            if y1_l<y1_fut:
                if x2_l <x2_fut:
                    if y2_l < y2_fut:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_l,x2_fut)])):
                            if len(set([i for i in range(y1_l,y1_fut)]&set([i for i in range(y2_l,y2_fut)]))):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
                    else:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_l,x2_fut)])):
                            if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_fut,y2_l)])):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
                else:
                    if y2_l < y2_fut:
                            if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_fut,x2_l)])):
                                if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_l,y2_fut)])):
                                    warn_list.append(1)
                                else:
                                    warn_list.append(0)
                    else:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_fut,x2_l)])):
                                if len(set([i for i in range(y1_l,y1_fut)])&set([i for i in range(y2_fut,y2_l)])):
                                    warn_list.append(1)
                        else:
                            warn_list.append(0)
            else:
                if x2_l <x2_fut:
                    if y2_l < y2_fut:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_l,x2_fut)])):
                            if len(set([i for i in range(y1_fut,y1_l)])&set([i for i in range(y2_l,y2_fut)])):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
                    else:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_l,x2_fut)])):
                            if len(set([i for i in range(y1_fut,y1_l)])&set([i for i in range(y2_fut,y2_l)])):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
                else:
                    if y2_l < y2_fut:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_fut,x2_l)])):
                            if len(set([i for i in range(y1_l,y2_fut)])&set([i for i in range(y2_l,y2_fut)])):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
                    else:
                        if len(set([i for i in range(x1_fut,x1_l)])&set([i for i in range(x2_l,x2_fut)])):
                            if len(set([i for i in range(y1_fut,y1_l)])&set([i for i in range(y2_fut,y2_l)])):
                                warn_list.append(1)
                        else:
                            warn_list.append(0)
    

def process_video():
    global warnings,collisions,current_frame
    
    # Открытие видео
    cap = cv2.VideoCapture("best_video.mp4")
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return
    
    # Получение характеристик видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Запись выходного видео
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_history = defaultdict(lambda: [])
    future_dict = defaultdict (lambda:[])
    crs_list = []


    # Основной цикл обработки видео
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Обработка кадра с моделью YOLO
        results = model.track(frame)
        if results[0].boxes is not None and results[0].boxes.id is not None :
            # Получение координат боксов и идентификаторов треков
            boxes = results[0].boxes.xywh.cpu()  # получение координат боксов
            track_ids = results[0].boxes.id.int().cpu().tolist()  # получение айдишников

            annotated_frame = results[0].plot()
 
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box  # координаты центра и размеры бокса
                track = track_history[track_id]
                track.append((float(x), float(y)))  # добавление координат в историю
                if len(track) > 30:  # длина истории 30 кадров максимум
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
                future_time = 1.5  # секунд
                future_x, future_y = predict_position(track, future_time, fps)
                future_track = future_dict[track_id]
                future_track.append((float(future_x),float(future_y)))

                if len(track) > 1:
                    last_x, last_y = track[-1]
                    cv2.line(annotated_frame, (int(last_x), int(last_y)), (int(future_x), int(future_y)), (0, 255, 255), 2)
                    # рисование линий трэка
                    
                    cv2.circle(annotated_frame, (int(future_x), int(future_y)), 5, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, 'Predicted', (int(future_x), int(future_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #WARNINGS
            if len(track_ids)>1:
                x1_l = round(float(boxes[0][0]))
                x2_l = round(float(boxes[1][0]))
                y1_l = round(float(boxes[0][1]))
                y2_l = round(float(boxes[1][1]))
                x1_fut = round(float(future_dict[1][0][0]))
                x2_fut = round(float(future_dict[2][0][0]))
                y1_fut = round(float(future_dict[1][0][1]))
                y2_fut = round(float(future_dict[2][0][1]))
                last_fut(x1_l,x2_l,y1_l,y2_l,x1_fut,x2_fut,y1_fut,y2_fut)
                if len(warn_list)>2:
                     warn_list.pop(0)
                if len(warn_list)>1 and warn_list[0]==1 and warn_list[1]==0:
                    warnings+=1  
                
                #CRASHES
                w_x1 = round(float(boxes[0][2]))/2
                w_x2 = round(float(boxes[1][2]))/2
                h_y1 = round(float(boxes[0][3]))/2
                h_y2 = round(float(boxes[1][3]))/2
                # x1_crs = 
                # x2_crs = 
                # y1_crs = 
                # y2_crs = 
                if len(list(set([x for x in range(round(float(boxes[0][0])-w_x1),round(float(boxes[0][0])+w_x1))])&set([x for x in range(round(float(boxes[1][0])-w_x2),round(float(boxes[1][0])+w_x2))])))>0 and len(list(set([y for y in range(round(float(boxes[0][1])-h_y1),round(float(boxes[0][1])+h_y1))])&set([y for y in range(round(float(boxes[1][1])-h_y2),round(float(boxes[1][1])+h_y2))])))>0:
                    crs_list.append(1)
                else:
                    crs_list.append(0)
                if len(crs_list)>2:
                    crs_list.pop(0)
                if crs_list[0]==1 and crs_list[1]==0:
                    collisions+=1


            if annotated_frame is not None:
                # Запись кадра в выходное видео
                out.write(annotated_frame)
                # cv2.imshow("YOLOv11 Tracking", annotated_frame)
                update_interface()
                current_frame = annotated_frame.copy()
            else:
                out.write(frame)# Запись кадра в выходное видео
                # cv2.imshow("YOLOv11 Tracking", frame)
                update_interface()
                current_frame = frame.copy()
                
                
                
        
        if cv2.waitKey(1) == 27:
            print(track[2])
            root.destroy()
            break #ESC чтобы перестало работать
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def update_interface():
    global current_frame
    warnings_label.config(text=f"Warnings: {warnings}")
    collisions_label.config(text=f"Collisions: {collisions}")
    if current_frame is not None:
        # Преобразование кадра OpenCV в формат, который может быть использован в Tkinter
        img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)  # Преобразуем BGR в RGB
        img = Image.fromarray(img)  # Преобразуем в Image
        img = ImageTk.PhotoImage(img)  # Преобразуем в PhotoImage

        # Обновляем метку с изображением
        image_label.config(image=img)
        image_label.image = img  # Сохраняем ссылку на изображение

# Создание основного окна приложения
root = tk.Tk()
root.title("Collision Detection Interface")

# Кнопка для начала обработки видео
start_button = tk.Button(root, text="Start Processing", command=process_video)
start_button.pack(pady=20)

# Метки для отображения количества предупреждений и столкновений
warnings_label = tk.Label(root, text=f"Warnings: {warnings}")
warnings_label.pack(pady=10)

collisions_label = tk.Label(root, text=f"Collisions: {collisions}")
collisions_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)
# Запуск основного цикла приложения
root.mainloop()