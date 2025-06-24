import numpy as np
from math import sqrt

class CentroidTracker:
    """
    Prosty tracker obiektów oparty na centroidach (środkach ciężkości).
    """
    def __init__(self, max_disappeared=30):
        """
        Inicjalizuje tracker centroidów.

        Args:
            max_disappeared (int): Maksymalna liczba kolejnych klatek,
                                   przez którą obiekt może być "niewidoczny",
                                   zanim zostanie wyrejestrowany.
        """
        self.next_object_id = 0
        self.objects = {}  # Przechowuje identyfikatory obiektów i ich centroidy
        self.boxes = {}    # Przechowuje identyfikatory obiektów i ich ramki ograniczające
        self.disappeared = {} # Przechowuje identyfikatory obiektów i liczbę klatek od ostatniego zobaczenia
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        """
        Rejestruje nowy obiekt.

        Args:
            centroid (tuple): Centroid (x, y) obiektu.
            box (tuple): Ramka ograniczająca (x1, y1, x2, y2) obiektu.
        """
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Wyrejestrowuje obiekt.

        Args:
            object_id (int): Identyfikator obiektu do wyrejestrowania.
        """
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Aktualizuje tracker o nowe detekcje ramek ograniczających.

        Args:
            rects (list): Lista ramek ograniczających (x1, y1, x2, y2) z bieżącej klatki.

        Returns:
            dict: Słownik aktualnie śledzonych obiektów {object_id: bounding_box}.
        """
        # Jeśli nie podano ramek ograniczających, oznacz wszystkie istniejące obiekty jako zniknięte
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Wyrejestruj obiekty, które zniknęły na zbyt wiele klatek
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.boxes

        # Oblicz nowe centroidy i przechowuj nowe ramki ograniczające
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_boxes = {}
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
            input_boxes[i] = (x1, y1, x2, y2)

        # Jeśli aktualnie nie są śledzone żadne obiekty, zarejestruj wszystkie nowe wejścia
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i])
        # W przeciwnym razie, spróbuj dopasować istniejące obiekty do nowych wejść
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Oblicz odległość euklidesową między każdym istniejącym centroidem obiektu
            # a każdym nowym centroidem wejściowym
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dist = sqrt((object_centroids[i][0] - input_centroids[j][0])**2 + (object_centroids[i][1] - input_centroids[j][1])**2)
                    D[i, j] = dist

            # Znajdź najmniejsze odległości w każdym wierszu, a następnie posortuj je
            # aby dopasować istniejące obiekty do nowych wejść
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Iteruj po dopasowaniach
            for (row, col) in zip(rows, cols):
                # Jeśli wiersz lub kolumna została już użyta, zignoruj
                if row in used_rows or col in used_cols:
                    continue
                
                # Zaktualizuj centroid obiektu, ramkę i zresetuj licznik zniknięć
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = input_boxes[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Oznacz zniknięte obiekty lub zarejestruj nowe
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Jeśli jest więcej istniejących obiektów niż centroidów wejściowych,
            # oznacz niedopasowane istniejące obiekty jako zniknięte
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            # W przeciwnym razie, zarejestruj nowe centroidy wejściowe jako nowe obiekty
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_boxes[col])
        return self.boxes