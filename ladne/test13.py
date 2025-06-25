import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from math import sqrt
import time
import threading
import queue

from database_function import ParkingDatabaseManager
from centroid_tracker import CentroidTracker

from config import (
    DETECTION_MODEL_PATH, DETECTION_MODEL_PLATES_PATH,
    VIDEO_SOURCE_BOTTOM, VIDEO_SOURCE_TOP, VIDEO_SOURCE_EXIT,
    TARGET_CLASS, PLATE_TARGET_CLASS, CONFIDENCE_THRESHOLD, PLATE_CONFIDENCE_THRESHOLD,
    PARKING_ZONES, TOTAL_PARKING_SPOTS, PARKING_OVERLAP_THRESHOLD,
    ROAD_ZONES, ROAD_OVERLAP_THRESHOLD, PARKED_OVERLAP_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT,
    ENTRYPOINT_ZONE, OVERLAP_THRESHOLD, ENTRY_GATE_LIGHT,
    EXITPOINT_ZONE, EXIT_GATE_LIGHT,
    COLLISION_OVERLAP_THRESHOLD,
    INITIALIZATION_FRAMES, REASSIGNMENT_DISTANCE_THRESHOLD,
    DB_RELOAD_INTERVAL_FRAMES,
    GATE_TIMEOUT
)

# Klasa Wątku dla Kamery OCR
class OCRCameraThread(threading.Thread):
    def __init__(self, thread_id, video_source, plate_detector, ocr_reader, request_queue, results_dict, thread_lock, is_exit_cam=False):
        super().__init__()
        self.thread_id = thread_id
        self.video_source = video_source
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        self.request_queue = request_queue
        self.results_dict = results_dict
        self.lock = thread_lock
        self.is_exit_cam = is_exit_cam
        self._stop_event = threading.Event()

        self.latest_frame = None
        self.latest_ocr_crop = None

    def stop(self):
        self._stop_event.set()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_crop(self):
        with self.lock:
            return self.latest_ocr_crop.copy() if self.latest_ocr_crop is not None else None

    def run(self):
        print(f"Wątek {self.thread_id} startuje...")
        cap = cv2.VideoCapture(self.video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print(f"Błąd: Nie można otworzyć kamery dla wątku {self.thread_id}")
            return

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Wątek {self.thread_id}: Koniec przesyłania obrazu.")
                break

            with self.lock:
                self.latest_frame = frame

            try:
                request_data = self.request_queue.get_nowait()
                tid = request_data['tid']
                expected_plate = request_data.get('expected_plate')

                print(f"Wątek {self.thread_id} otrzymał żądanie OCR dla ID: {tid}")

                results = self.plate_detector(frame, imgsz=640, verbose=False)[0]
                found_plate = None

                for r in results.boxes:
                    if float(r.conf[0]) < PLATE_CONFIDENCE_THRESHOLD: continue
                    if self.plate_detector.names[int(r.cls[0])] != PLATE_TARGET_CLASS: continue

                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    if y1 < y2 and x1 < x2:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            with self.lock:
                                self.latest_ocr_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                            ocr_result = self.ocr_reader.readtext(self.latest_ocr_crop)
                            for (bbox, text, conf) in ocr_result:
                                cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
                                if 4 <= len(cleaned_text) <= 8:
                                    if self.is_exit_cam:
                                        if cleaned_text == expected_plate:
                                            found_plate = cleaned_text
                                            print(f"W wątku {self.thread_id} została potwierdzona tablica: {found_plate}")
                                            break
                                    else:
                                        found_plate = cleaned_text
                                        print(f"W wątku {self.thread_id} została znaleziona tablica: {found_plate}")
                                        break
                    if found_plate:
                        break

                with self.lock:
                    self.results_dict[tid] = {'plate': found_plate, 'source': self.thread_id}

            except queue.Empty:
                time.sleep(0.01)
                continue

        cap.release()
        print(f"Wątek {self.thread_id} zakończony.")


def calculate_overlap(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return intersection_area / box1_area if box1_area > 0 else 0

#Inicjalizacja modeli
detector = YOLO(DETECTION_MODEL_PATH)
plate_detector_model = YOLO(DETECTION_MODEL_PLATES_PATH)
ocr_reader = easyocr.Reader(['pl'])
tracker = CentroidTracker(max_disappeared=50)
db_manager = ParkingDatabaseManager()

# Inicjalizacja i ustawienie wątków
thread_lock = threading.Lock()
entry_ocr_requests = queue.Queue()
exit_ocr_requests = queue.Queue()
ocr_results = {}

entry_thread = OCRCameraThread(
    thread_id='WJAZD',
    video_source=VIDEO_SOURCE_BOTTOM,
    plate_detector=plate_detector_model,
    ocr_reader=ocr_reader,
    request_queue=entry_ocr_requests,
    results_dict=ocr_results,
    thread_lock=thread_lock
)

exit_thread = OCRCameraThread(
    thread_id='WYJAZD',
    video_source=VIDEO_SOURCE_EXIT,
    plate_detector=plate_detector_model,
    ocr_reader=ocr_reader,
    request_queue=exit_ocr_requests,
    results_dict=ocr_results,
    thread_lock=thread_lock,
    is_exit_cam=True
)

entry_thread.start()
exit_thread.start()

# Otwarcie górnej kamery
cap_top = cv2.VideoCapture(VIDEO_SOURCE_TOP)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap_top.isOpened():
    print("Błąd: Nie można otworzyć górnej kamery")
    entry_thread.stop()
    exit_thread.stop()
    entry_thread.join()
    exit_thread.join()
    exit()

# Dozwolone tablice
db_manager.add_allowed_plate("ELW15241")
db_manager.add_allowed_plate("EL3AC61")
db_manager.add_allowed_plate("EL7271N")
db_manager.add_allowed_plate("EL4MF32")
db_manager.add_allowed_plate("EL6HV57")

db_manager.add_allowed_plate("EL8U902")
db_manager.add_allowed_plate("1001")
db_manager.add_allowed_plate("2002")


track_to_plate = {}
track_entered_zone = {}
track_exiting_zone = {}
track_history = {}
frame_num = 0
known_vehicles_from_db = []
is_currently_offending = {}
vehicle_status = {}
gate_opened_time = None

x1_ep, y1_ep, x2_ep, y2_ep = ENTRYPOINT_ZONE
x1_exp, y1_exp, x2_exp, y2_exp = EXITPOINT_ZONE
x1_engl, y1_engl, x2_engl, y2_engl = ENTRY_GATE_LIGHT
x1_exgl, y1_exgl, x2_exgl, y2_exgl = EXIT_GATE_LIGHT

# Główna pętla programu
try:
    while True:
        ret_t, frame_t = cap_top.read()
        frame_b = entry_thread.get_latest_frame()
        frame_e = exit_thread.get_latest_frame()
        entry_crop = entry_thread.get_latest_crop()
        exit_crop = exit_thread.get_latest_crop()


        if not ret_t:
            print("Błąd: Nie można odczytać klatki z kamery górnej. Koniec programu.")
            break
        frame_num += 1

        completed_results = {}
        with thread_lock:
            if ocr_results:
                completed_results = ocr_results.copy()
                ocr_results.clear()

        for tid, result_data in completed_results.items():
            source = result_data['source']
            plate_text = result_data['plate']

            if source == 'WJAZD':
                if plate_text:
                    if db_manager.is_allowed_plate(plate_text):
                        print(f"GŁÓWNY WĄTEK: Potwierdzono dozwoloną tablicę '{plate_text}' dla ID:{tid}.")
                        track_to_plate[tid] = plate_text
                        db_manager.add_entry(plate_text)
                    else:
                        print(f"GŁÓWNY WĄTEK: Wykryta tablica '{plate_text}' nie jest dozwolona.")
                else:
                    print(f"GŁÓWNY WĄTEK: OCR nie powiódł się dla ID:{tid} w strefie wjazdu.")
                if tid in track_entered_zone:
                    del track_entered_zone[tid]

            elif source == 'WYJAZD':
                expected_plate_from_tracker = track_to_plate.get(tid, None)

                if plate_text and plate_text == expected_plate_from_tracker:
                    print(f"GŁÓWNY WĄTEK: Wjechał pojazd '{plate_text}' (ID:{tid}).")
                    db_manager.delete_plate_from_active_parking(plate_text)
                    db_manager.update_exit(plate_text)
                    if tid in track_to_plate: del track_to_plate[tid]
                    if tid in is_currently_offending: del is_currently_offending[tid]
                    if tid in vehicle_status: del vehicle_status[tid]
                    if tid in track_entered_zone: del track_entered_zone[tid]
                    if tid in track_exiting_zone: del track_exiting_zone[tid]
                else:
                    print(f"Główny watek: Błąd tablicy. Oczekiwano: {expected_plate_from_tracker}, Odczytano: {plate_text}.")
                    if tid in track_exiting_zone: del track_exiting_zone[tid]


        # Przeładowywanie makiety z bazy
        if frame_num % DB_RELOAD_INTERVAL_FRAMES == 0:
            print(f"Odświeżanie danych z bazy w klatce {frame_num}")
            known_vehicles_from_db = db_manager.load_vehicles_from_active_parking()

        rects_for_tracker = []
        results_t = detector(frame_t, imgsz=640, verbose=False)[0]
        for r in results_t.boxes:
            if float(r.conf[0]) < CONFIDENCE_THRESHOLD: continue
            if detector.names[int(r.cls[0])] == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                rects_for_tracker.append((x1, y1, x2, y2))

        tracked_objects = tracker.update(rects_for_tracker)
        occupied_parking_zones = set()
        collisions_this_frame = {}

        for tid, box in tracked_objects.items():
            l, t, r_, b = map(int, box)
            cx, cy = (l + r_) // 2, (t + b) // 2
            vehicle_box = (l, t, r_, b)

            if tid not in track_to_plate and known_vehicles_from_db:
                current_centroid = (cx, cy)
                min_dist = float('inf')
                best_match_index = -1
                for i, known_vehicle in enumerate(list(known_vehicles_from_db)):
                    if 'centroid' in known_vehicle and known_vehicle['centroid'] is not None:
                        dist = sqrt((current_centroid[0] - known_vehicle['centroid'][0])**2 +
                                    (current_centroid[1] - known_vehicle['centroid'][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_match_index = i
                if min_dist < REASSIGNMENT_DISTANCE_THRESHOLD and best_match_index != -1:
                    matched_vehicle = known_vehicles_from_db.pop(best_match_index)
                    plate = matched_vehicle['plate']
                    track_to_plate[tid] = plate
                    print(f"Ponowne przypisanie: Obiekt ID:{tid} to tablica '{plate}' - obliczona odległość: {min_dist:.0f}px")

            if tid not in track_history: track_history[tid] = []
            track_history[tid].append((cx, cy))
            track_history[tid] = track_history[tid][-50:]
            
            

            overlap_ratio_entry = calculate_overlap(vehicle_box, ENTRYPOINT_ZONE)
            if overlap_ratio_entry >= OVERLAP_THRESHOLD and tid not in track_to_plate and not track_entered_zone.get(tid):
                track_entered_zone[tid] = True
                request = {'tid': tid}
                entry_ocr_requests.put(request)
                print(f"Pojazd ID:{tid} w strefie wjazdu. OCR do wątku WJAZD.")


            overlap_ratio_exit = calculate_overlap(vehicle_box, EXITPOINT_ZONE)

            if overlap_ratio_exit >= OVERLAP_THRESHOLD:
                plate_to_exit = track_to_plate[tid]
                if db_manager.is_plate_in_active_parking(plate_to_exit):
                    track_exiting_zone[tid] = True
                    request = {'tid': tid, 'expected_plate': plate_to_exit}
                    exit_ocr_requests.put(request)
                    print(f"Pojazd o ID:{tid} ({plate_to_exit}) w strefie wyjazdu. OCR do wątku WYJAZD.")

            if tid in track_to_plate:
                plate = track_to_plate[tid]
                db_manager.add_plate_to_active_parking(plate, l, t, r_, b)

            occupied_zones_by_vehicle = set()
            count = 0
            for zone_name, zone_box in PARKING_ZONES.items():
                if calculate_overlap(vehicle_box, zone_box) >= PARKING_OVERLAP_THRESHOLD:
                    occupied_parking_zones.add(zone_name)
                    occupied_zones_by_vehicle.add(zone_name)
                    count += 1
            
            offense_type_parking = "Zajecie kilku miejsc"
            if count >= 2:
                if tid in track_to_plate:
                    plate = track_to_plate[tid]
                    if tid not in is_currently_offending or is_currently_offending[tid] != offense_type_parking:
                        print(f"Samochód {plate} (ID:{tid}) zajmuje wiele miejsc. Zarejestrowanoo wykroczenie.")
                        db_manager.add_forbidden_move(plate, offense_type_parking)
                        is_currently_offending[tid] = offense_type_parking
                else:
                    print(f"Samochód (ID:{tid}) zajmuje wiele miejsc.")
            else:
                if tid in is_currently_offending and is_currently_offending[tid] == offense_type_parking:
                    print(f"Samochód (ID:{tid}) nie zajmuje już wielu miejsc.")
                    del is_currently_offending[tid]


            offense_type_collision = "Kolizja"
            colliding_with_ids = []
            for other_tid, other_box in tracked_objects.items():
                if tid >= other_tid: continue
                overlap1_2 = calculate_overlap(vehicle_box, other_box)
                overlap2_1 = calculate_overlap(other_box, vehicle_box)
                if overlap1_2 >= COLLISION_OVERLAP_THRESHOLD or overlap2_1 >= COLLISION_OVERLAP_THRESHOLD:
                    colliding_with_ids.append(other_tid)
                    collision_pair = tuple(sorted((tid, other_tid)))
                    collisions_this_frame[collision_pair] = True
            
            if colliding_with_ids:
                if tid in track_to_plate:
                    plate = track_to_plate[tid]
                    if tid not in is_currently_offending or is_currently_offending[tid] != offense_type_collision:
                        print(f"Samochód {plate} (ID:{tid}) rozpoczął kolizję z ID:{colliding_with_ids}. Zarejestrowano wykroczenie")
                        db_manager.add_forbidden_move(plate, offense_type_collision)
                        is_currently_offending[tid] = offense_type_collision
                else:
                    print(f"Samochód (ID:{tid}) rozpoczął kolizję z ID:{colliding_with_ids}.")
            else:
                if tid in is_currently_offending and is_currently_offending[tid] == offense_type_collision:
                    print(f"Samochód (ID:{tid}) skończył kolizję.")
                    del is_currently_offending[tid]

            current_vehicle_status = None
            is_parked = any(calculate_overlap(vehicle_box, zone_box) >= PARKED_OVERLAP_THRESHOLD for zone_box in PARKING_ZONES.values())
            if is_parked:
                current_vehicle_status = "zaparkowany"
            else:
                is_on_road = any(calculate_overlap(vehicle_box, zone_box) >= ROAD_OVERLAP_THRESHOLD for zone_box in ROAD_ZONES.values())
                if is_on_road:
                    current_vehicle_status = "parkuje"
                else:
                    current_vehicle_status = "poza strefami"


            if vehicle_status.get(tid) != current_vehicle_status:
                plate_info = track_to_plate.get(tid, f"ID:{tid}")
                print(f"Samochód {plate_info} zmienił status na: {current_vehicle_status}")
                vehicle_status[tid] = current_vehicle_status


            if tid in track_to_plate:
                plate = track_to_plate[tid]
                db_manager.update_plate_position(plate, l, t, r_, b)

            # Rysowanie obiektów na górnej kamerze
            label_text = track_to_plate.get(tid, f"ID:{tid}")
            status = vehicle_status.get(tid)
            color = (0, 255, 0) if tid in track_to_plate else (255, 0, 0)
            if status == "zaparkowany": color = (255, 255, 0)
            elif status == "parkuje": color = (0, 165, 255)
            
            if tid in is_currently_offending and is_currently_offending[tid] == offense_type_collision:
                color = (0, 0, 255)
                cv2.putText(frame_t, "KOLIZJA", (l, t + (b-t)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.rectangle(frame_t, (l, t), (r_, b), color, 2)
            cv2.putText(frame_t, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if status:
                cv2.putText(frame_t, status, (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if tid in track_history:
                pts = track_history[tid]
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None: continue
                    cv2.line(frame_t, pts[i - 1], pts[i], (0, 255, 255), 2)


        # Rysowanie stref
        for zone_name, (x1, y1, x2, y2) in PARKING_ZONES.items():
            color = (0, 0, 255) if zone_name in occupied_parking_zones else (0, 255, 255)
            cv2.rectangle(frame_t, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_t, zone_name.replace("ZONE_", "Strefa "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for zone_name, (x1, y1, x2, y2) in ROAD_ZONES.items():
            cv2.rectangle(frame_t, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame_t, zone_name.replace("ROAD_", "Droga "), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


        cv2.rectangle(frame_t, (x1_ep, y1_ep), (x2_ep, y2_ep), (255, 0, 0), 2)
        cv2.putText(frame_t, "Strefa Wjazdu", (x1_ep, y1_ep - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(frame_t, (x1_exp, y1_exp), (x2_exp, y2_exp), (255, 0, 0), 2)
        cv2.putText(frame_t, "Strefa Wyjazdu", (x1_exp, y1_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        if frame_num < INITIALIZATION_FRAMES:
            init_text = f"Faza Inicjalizacji: {frame_num}/{INITIALIZATION_FRAMES}"
            cv2.putText(frame_t, init_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

       
        any_allowed_in_entry = False
        for tid_check, box_check in tracked_objects.items():
            if tid_check in track_to_plate and db_manager.is_allowed_plate(track_to_plate[tid_check]):
                if calculate_overlap(box_check, ENTRYPOINT_ZONE) >= OVERLAP_THRESHOLD:
                    any_allowed_in_entry = True
                    break
        
        current_occupied_spots = len(occupied_parking_zones)
        can_enter = any_allowed_in_entry and current_occupied_spots < TOTAL_PARKING_SPOTS


        is_car_in_passage = False
        if not can_enter:
            for tid_check, box_check in tracked_objects.items():
                if calculate_overlap(box_check, ENTRY_GATE_LIGHT) > 0.0:
                    is_car_in_passage = True
                    break


        if can_enter or is_car_in_passage:
            entry_light_color = (0, 255, 0)
        else:
            entry_light_color = (0, 0, 255)

        cv2.rectangle(frame_t, (x1_engl, y1_engl), (x2_engl, y2_engl), entry_light_color, -1)
        cv2.rectangle(frame_t, ENTRY_GATE_LIGHT[:2], ENTRY_GATE_LIGHT[2:], (0, 255, 0), 1)

        
        can_exit = False
        for tid_check, box_check in tracked_objects.items():
            if tid_check in track_to_plate and db_manager.is_plate_in_active_parking(track_to_plate[tid_check]):
                if calculate_overlap(box_check, EXITPOINT_ZONE) >= OVERLAP_THRESHOLD:
                    can_exit = True
                    break
        

        is_car_in_exit_passage = False
        if not can_exit:
            for tid_check, box_check in tracked_objects.items():
                if calculate_overlap(box_check, EXIT_GATE_LIGHT) > 0.0:
                    is_car_in_exit_passage = True
                    break


        if can_exit or is_car_in_exit_passage:
            exit_light_color = (0, 255, 0)
        else:
            exit_light_color = (0, 0, 255)

        cv2.rectangle(frame_t, (x1_exgl, y1_exgl), (x2_exgl, y2_exgl), exit_light_color, -1)
        cv2.rectangle(frame_t, EXIT_GATE_LIGHT[:2], EXIT_GATE_LIGHT[2:], (0, 255, 0), 1)



        if frame_b is not None:
            cv2.imshow("Dolna kamera", cv2.resize(frame_b, None, fx=0.5, fy=0.5))
        if entry_crop is not None:
            cv2.imshow("Tablica wjazdowa", entry_crop)

        cv2.imshow("Górna kamera", frame_t)

        if frame_e is not None:
            cv2.imshow("Kamera wyjazdowa", cv2.resize(frame_e, None, fx=0.5, fy=0.5))
        if exit_crop is not None:
            cv2.imshow("Tablica wyjazdowa", exit_crop)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Zatrzymywanie programu...")
    entry_thread.stop()
    exit_thread.stop()
    cap_top.release()

    print("Kończenie wątków")
    entry_thread.join()
    exit_thread.join()

    db_manager.close()
    cv2.destroyAllWindows()
    print("Koniec programu.")