import os
import sys
import time
import uuid
import queue
import logging
import threading
import subprocess
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, messagebox

# Подавляем системный спам логов OpenCV до загрузки C++ бэкендов
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2
import torch
from PIL import Image, ImageTk, ImageDraw
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PointIdsList
from torchvision.models import resnet50, ResNet50_Weights


# =====================================================================
# ПУТИ
# =====================================================================

if getattr(sys, "frozen", False):
    APPLICATION_PATH = os.path.dirname(os.path.abspath(sys.executable))
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))

# Все изменяемые данные лежат рядом с EXE.
# Это сохраняет совместимость с вашей текущей структурой.
DATA_DIR = APPLICATION_PATH
BASE_DIR = os.path.join(DATA_DIR, "reference_images")
TYPES_FILE = os.path.join(DATA_DIR, "types.txt")

COLLECTION_NAME = "parts_resnet50"
QDRANT_URL = "http://localhost:6333"

os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.info("============================================================")
logging.info("IndustrialVision запускается")
logging.info(f"APPLICATION_PATH: {APPLICATION_PATH}")
logging.info(f"REFERENCE_IMAGES:  {BASE_DIR}")
logging.info(f"QDRANT_URL:        {QDRANT_URL}")
logging.info("============================================================")


# =====================================================================
# ФАЙЛОВЫЕ УТИЛИТЫ
# =====================================================================

def get_path(filename):
    return os.path.join(DATA_DIR, filename)


def safe_write_text(path, lines):
    """Надёжная запись текстового справочника."""
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        logging.exception(f"Не удалось записать файл: {path}")
        return False


def load_list(filename, default):
    filepath = get_path(filename)

    try:
        if not os.path.exists(filepath):
            safe_write_text(filepath, default)
            return list(default)

        with open(filepath, "r", encoding="utf-8") as f:
            result = [line.strip() for line in f if line.strip()]

        return result
    except Exception:
        logging.exception(f"Не удалось прочитать {filepath}")
        return list(default)


def save_list(filename, data_list):
    return safe_write_text(get_path(filename), data_list)


def sanitize_path_part(value):
    """Убирает запрещённые Windows-символы из имени папки."""
    value = str(value).strip()
    forbidden = '<>:"/\\|?*'
    for ch in forbidden:
        value = value.replace(ch, "_")
    value = value.rstrip(". ")
    return value or "Без названия"


def reference_dir(v_type, part):
    return os.path.join(
        BASE_DIR,
        sanitize_path_part(v_type),
        sanitize_path_part(part),
    )


def reference_file(v_type, part, point_id):
    return os.path.join(
        reference_dir(v_type, part),
        f"{point_id}.jpg",
    )


# =====================================================================
# DATABASE MANAGER
# =====================================================================

class DatabaseManager:
    """
    Единая точка работы с:
      1. Qdrant
      2. reference_images

    Правило:
      - сначала физически сохраняем фото;
      - проверяем файл;
      - затем создаём запись в Qdrant;
      - путь к фото сохраняем в payload.
    """

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

    def __init__(self, qdrant_url, collection_name, base_dir):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.base_dir = base_dir
        self.lock = threading.RLock()

        os.makedirs(self.base_dir, exist_ok=True)

        logging.info("Подключение к Qdrant...")
        self.client = QdrantClient(url=self.qdrant_url, timeout=5.0)

        self.ensure_collection()
        self.repair_database()

    def ensure_collection(self):
        with self.lock:
            if not self.client.collection_exists(self.collection_name):
                logging.info(
                    f"Коллекция {self.collection_name} отсутствует. Создаём..."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=2048,
                        distance=Distance.COSINE,
                    ),
                )

            logging.info(
                f"Qdrant готов: collection={self.collection_name}"
            )

    def _normalize_point_id(self, point_id):
        return str(point_id)

    def save_image(self, image_bgr, path):
        """
        Надёжно сохраняет BGR-изображение через PIL.
        Возвращает True только если файл реально существует и имеет размер > 0.
        """
        try:
            if image_bgr is None or image_bgr.size == 0:
                raise ValueError("Пустое изображение")

            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            tmp_path = path + ".tmp.jpg"
            image.save(
                tmp_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 0:
                raise IOError("Временный файл изображения не создан")

            os.replace(tmp_path, path)

            if not os.path.exists(path) or os.path.getsize(path) <= 0:
                raise IOError("Финальный файл изображения не создан")

            logging.info(
                f"📸 Фото сохранено: {path} "
                f"({os.path.getsize(path)} bytes)"
            )
            return True

        except Exception:
            logging.exception(f"❌ Ошибка сохранения изображения: {path}")
            try:
                tmp_path = path + ".tmp.jpg"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    def save_reference(self, image_bgr, vector, v_type, part):
        """
        Полностью атомарная логика регистрации:
        фото -> проверка -> Qdrant.
        """
        point_id = str(uuid.uuid4())
        path = reference_file(v_type, part, point_id)

        with self.lock:
            # 1. Сначала фото
            if not self.save_image(image_bgr, path):
                raise RuntimeError(
                    "Фотография не была сохранена. "
                    "Запись в Qdrant отменена."
                )

            # 2. Затем Qdrant
            payload = {
                "name": str(part),
                "type": str(v_type),
                "image_path": os.path.abspath(path),
                "filename": os.path.basename(path),
                "group_id": f"{v_type}_{part}",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    ],
                    wait=True,
                )
            except Exception:
                # Qdrant не принял запись — удаляем фото, чтобы
                # не оставлять "сироту".
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    logging.exception(
                        "Не удалось удалить фото после ошибки Qdrant"
                    )
                raise

            # 3. Финальная проверка
            try:
                found = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False,
                )
                if not found:
                    raise RuntimeError(
                        "Qdrant не вернул только что сохранённую запись."
                    )
            except Exception:
                logging.exception("Проверка сохранения Qdrant не удалась")

            logging.info(
                f"✅ Эталон сохранён: type={v_type}, part={part}, "
                f"id={point_id}, image={path}"
            )

            return point_id, path

    def search(self, vector, limit=3):
        """Поиск с совместимостью со старыми/новыми qdrant_client."""
        with self.lock:
            if hasattr(self.client, "query_points"):
                return self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                )

            # Старые версии qdrant_client.
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
            return SimpleNamespace(points=points)

    def delete_point(self, point_id):
        with self.lock:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=[self._normalize_point_id(point_id)]
                ),
                wait=True,
            )

    def delete_reference(self, v_type, part, filename):
        """Удаляет конкретную фотографию и соответствующую точку Qdrant."""
        path = os.path.join(reference_dir(v_type, part), filename)
        point_id = os.path.splitext(filename)[0]

        # Сначала удаляем запись Qdrant. Если Qdrant недоступен —
        # локальный файл сохраняем, чтобы не получить рассинхронизацию.
        self.delete_point(point_id)

        if os.path.exists(path):
            os.remove(path)

    def delete_part(self, v_type, part):
        """
        Удаляет деталь полностью.

        Важно: точки Qdrant ищутся по payload, а не только по именам файлов.
        Благодаря этому удаляются и "призрачные" записи, у которых фото уже
        отсутствует на диске.
        """
        points = self.scroll_all()
        point_ids = []

        for point in points:
            payload = point.payload or {}
            p_type = str(payload.get("type", "")).strip()
            p_name = str(payload.get("name", "")).strip()
            if p_type == str(v_type).strip() and p_name == str(part).strip():
                point_ids.append(str(point.id))

        # Сначала Qdrant, затем файловая система.
        if point_ids:
            with self.lock:
                for start in range(0, len(point_ids), 256):
                    chunk = point_ids[start:start + 256]
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(points=chunk),
                        wait=True,
                    )

        p_path = reference_dir(v_type, part)
        if os.path.isdir(p_path):
            for filename in os.listdir(p_path):
                full_path = os.path.join(p_path, filename)
                try:
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                except Exception:
                    logging.exception(f"Не удалось удалить файл: {full_path}")
                    raise

            try:
                os.rmdir(p_path)
            except OSError:
                pass

    def scroll_all(self):
        """
        Получает ВСЕ записи Qdrant, а не первые N.
        """
        records = []
        offset = None

        with self.lock:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                records.extend(points)

                if next_offset is None:
                    break

                offset = next_offset

        return records

    def list_groups_from_qdrant(self):
        """
        Формирует объединённый список из Qdrant + reference_images.

        Поэтому вкладка "Управление базой" не становится пустой только из-за
        рассинхронизации: локальные фотографии всё равно будут показаны.
        """
        groups = {}

        for point in self.scroll_all():
            payload = point.payload or {}
            v_type = str(payload.get("type", "")).strip()
            part = str(payload.get("name", "")).strip()

            if not v_type or not part:
                continue

            key = (v_type, part)
            group = groups.setdefault(
                key,
                {
                    "type": v_type,
                    "part": part,
                    "count": 0,
                    "files": set(),
                },
            )
            group["count"] += 1

            image_path = payload.get("image_path")
            filename = payload.get("filename")

            if image_path and os.path.exists(str(image_path)):
                group["files"].add(os.path.basename(str(image_path)))
            elif filename:
                possible = reference_file(
                    v_type,
                    part,
                    os.path.splitext(str(filename))[0],
                )
                if os.path.exists(possible):
                    group["files"].add(os.path.basename(possible))

        # Добавляем всё, что физически есть на диске, даже если Qdrant пуст.
        for v_type, part, files in scan_reference_groups(self.base_dir):
            key = (v_type, part)
            group = groups.setdefault(
                key,
                {
                    "type": v_type,
                    "part": part,
                    "count": 0,
                    "files": set(),
                },
            )
            group["files"].update(files)

        result = []
        for group in groups.values():
            result.append(
                {
                    "type": group["type"],
                    "part": group["part"],
                    "count": group["count"],
                    "files": sorted(group["files"]),
                }
            )
        return result

    def close(self):
        try:
            close_fn = getattr(self.client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            logging.exception("Ошибка закрытия QdrantClient")

    def repair_database(self):
        """
        Синхронизация после запуска.

        Старые записи, где image_path отсутствует, восстанавливаем
        по структуре reference_images/type/part/UUID.jpg.

        Также обнаруживаем фотографии без Qdrant-записи. Они не удаляются:
        вкладка управления базой всё равно показывает их с диска.
        """
        logging.info("🔧 Проверка/восстановление базы...")

        try:
            points = self.scroll_all()
        except Exception:
            logging.exception("Не удалось прочитать Qdrant при проверке")
            return

        qdrant_ids = set()
        updates = []

        for point in points:
            point_id = str(point.id)
            qdrant_ids.add(point_id)

            payload = dict(point.payload or {})
            v_type = str(payload.get("type", "")).strip()
            part = str(payload.get("name", "")).strip()

            if not v_type or not part:
                continue

            expected = reference_file(v_type, part, point_id)

            changed = False

            if payload.get("image_path") != os.path.abspath(expected):
                payload["image_path"] = os.path.abspath(expected)
                changed = True

            if payload.get("filename") != os.path.basename(expected):
                payload["filename"] = os.path.basename(expected)
                changed = True

            if changed:
                # Не создаём PointStruct(vector=None): некоторые версии
                # qdrant_client валидируют vector как обязательный и падают.
                updates.append((point.id, payload))

        # Меняем только payload, не трогая вектор.
        for point_id, payload in updates:
            try:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=payload,
                    points=[point_id],
                    wait=True,
                )
            except Exception:
                logging.exception(
                    f"Не удалось обновить payload точки {point_id}"
                )

        # Проверяем физические файлы.
        orphan_count = 0

        for root, _, files in os.walk(self.base_dir):
            for filename in files:
                if not filename.lower().endswith(self.IMAGE_EXTENSIONS):
                    continue

                point_id = os.path.splitext(filename)[0]

                if point_id not in qdrant_ids:
                    orphan_count += 1
                    logging.warning(
                        f"⚠️ Найден файл без записи Qdrant: "
                        f"{os.path.join(root, filename)}"
                    )

        logging.info(
            f"🔧 Проверка базы завершена. "
            f"Qdrant записей: {len(points)}, "
            f"файлов без записи: {orphan_count}"
        )


# =====================================================================
# ИНИЦИАЛИЗАЦИЯ ИИ И БД
# =====================================================================

# Тяжёлую инициализацию Qdrant/ResNet выполняем ПОСЛЕ показа окна в
# отдельном потоке. Это принципиально: даже если Qdrant выключен или
# torchvision впервые загружает веса, Tkinter остаётся отзывчивым.


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def scan_reference_groups(base_dir):
    """Возвращает локальные type/part/файлы без обращения к Qdrant."""
    result = []
    if not os.path.isdir(base_dir):
        return result

    try:
        type_dirs = sorted(os.listdir(base_dir))
    except Exception:
        logging.exception(f"Не удалось прочитать {base_dir}")
        return result

    for v_type in type_dirs:
        type_path = os.path.join(base_dir, v_type)
        if not os.path.isdir(type_path):
            continue

        try:
            part_dirs = sorted(os.listdir(type_path))
        except Exception:
            logging.exception(f"Не удалось прочитать {type_path}")
            continue

        for part in part_dirs:
            part_path = os.path.join(type_path, part)
            if not os.path.isdir(part_path):
                continue

            try:
                files = sorted(
                    f for f in os.listdir(part_path)
                    if f.lower().endswith(DatabaseManager.IMAGE_EXTENSIONS)
                    and os.path.isfile(os.path.join(part_path, f))
                )
            except Exception:
                logging.exception(f"Не удалось прочитать {part_path}")
                continue

            if files:
                result.append((v_type, part, files))

    return result


# =====================================================================
# КАМЕРЫ
# =====================================================================

def _camera_backends():
    if os.name == "nt":
        backends = []
        for name in ("CAP_DSHOW", "CAP_MSMF"):
            value = getattr(cv2, name, None)
            if value is not None:
                backends.append(value)
        backends.append(None)  # CAP_ANY / авто
        return backends

    if hasattr(cv2, "CAP_V4L2"):
        return [cv2.CAP_V4L2, None]
    return [None]


def open_camera(index, width=1280, height=720, probe=False):
    """Пробует несколько backend'ов OpenCV и возвращает (cap, first_frame)."""
    last_error = None

    for backend in _camera_backends():
        cap = None
        try:
            if backend is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            # MJPG часто убирает чёрный поток/низкий FPS на USB-камерах Windows.
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

            for prop_name, value in (
                ("CAP_PROP_FRAME_WIDTH", width),
                ("CAP_PROP_FRAME_HEIGHT", height),
                ("CAP_PROP_BUFFERSIZE", 1),
            ):
                prop = getattr(cv2, prop_name, None)
                if prop is not None:
                    try:
                        cap.set(prop, value)
                    except Exception:
                        pass

            # Для некоторых backend'ов поддерживается read timeout.
            read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
            if read_timeout is not None:
                try:
                    cap.set(read_timeout, 1500)
                except Exception:
                    pass

            if not probe:
                return cap, None

            first_frame = None
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    first_frame = frame
                    break
                time.sleep(0.03)

            if first_frame is not None:
                return cap, first_frame

            cap.release()

        except Exception as e:
            last_error = e
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    if last_error:
        logging.debug(f"Камера {index}: {last_error}")
    return None, None


def detect_available_cameras(max_tested=8):
    """Сканирование камер. Вызывать только в фоновом потоке."""
    available = []

    for i in range(max_tested):
        if os.name != "nt" and os.path.exists("/dev"):
            if not os.path.exists(f"/dev/video{i}"):
                continue

        cap, frame = open_camera(i, probe=True)
        if cap is not None and frame is not None:
            available.append(i)
            cap.release()

    logging.info(f"📷 Доступные камеры: {available}")
    return available


# =====================================================================
# GUI
# =====================================================================

class IndustrialVisionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Индустриальный Комплекс Компьютерного Зрения")
        self.root.geometry("1400x850")
        self.root.configure(bg="#2b2b2b")

        self._closing = False
        self.ui_queue = queue.Queue()
        self.inference_queue = queue.Queue(maxsize=1)

        self.state_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.backend_init_lock = threading.Lock()

        self.is_running = threading.Event()
        self.is_running.set()
        self.backend_ready = threading.Event()
        self.backend_error = None

        self.db = None
        self.device = choose_device()
        self.resnet = None
        self.preprocess = None

        self.current_raw_frame = None
        self.latest_crop = None
        self.latest_raw_crop = None

        # Камера 0 показывается сразу, а реальное сканирование камер запускается
        # после создания UI и выполняется в фоне.
        self.available_cameras = [0]
        self.current_camera_index = 0
        self.requested_camera_index = 0
        self.camera_scan_running = threading.Event()
        self.camera_refresh_lock = threading.Lock()

        # Управление камерой. Tkinter-поток только меняет желаемые значения,
        # а VideoCapture настраивается исключительно в camera worker.
        # Так ползунки не могут подвесить интерфейс драйвером камеры.
        self.camera_settings_lock = threading.Lock()
        self.camera_settings_dirty = threading.Event()
        self.autofocus_enabled = True
        self.manual_focus_value = 255

        # Цифровое приближение именно ОБЛАСТИ СКАНИРОВАНИЯ.
        # 1.0x = максимально большая центральная квадратная область,
        # 2.0x = прежний размер (половина меньшей стороны кадра),
        # 4.0x = более тесная центральная область.
        self.scan_zoom = 2.0

        self.scan_results = {
            "part": "Ожидание...",
            "score": "0%",
            "status_text": "СКАНИРОВАНИЕ...",
            "status_bg": "#F59E0B",
            "sim1_name": "---",
            "sim1_score": "0%",
            "path_sim1": "",
            "sim2_name": "---",
            "sim2_score": "0%",
            "path_sim2": "",
            "path_best": "",
        }

        self.frame_counter = 0
        self.scan_line_y = 0
        self.scan_line_dir = 1
        self.is_detected = False
        self.delete_all_confirmed = False
        self.candidate_name = None
        self.confirm_counter = 0

        self.CONFIRM_FRAMES_FAST = 2
        self.CONFIRM_FRAMES_SLOW = 10
        self.MIN_SCORE = 0.50
        self.patience_counter = 0
        self.MAX_PATIENCE = 3
        self.last_spoken_name = None

        # Защита от повторных/устаревших фоновых операций.
        self.gallery_generation = 0
        self.save_in_progress = False

        # Кэш эталонных картинок: раньше они читались с диска ~30 раз/сек.
        self._image_cache = {}
        self._blank_cache = {}
        self._last_scanner_ui_state = None

        # UI
        self.setup_top_bar()
        self.setup_tabs()

        self.blank_image = self.create_blank_image((300, 300), "Нет фото")
        self.blank_sim = self.create_blank_image((200, 200), "Пусто")

        self.setup_scanner_ui(self.tab_scanner)
        self.setup_register_ui(self.tab_register)
        self.setup_gallery_ui(self.tab_gallery)
        self.setup_metadata_ui(self.tab_metadata)
        self.load_registration_lists()

        # Сначала запускаем UI-циклы, затем тяжёлые фоновые задачи.
        self.process_ui_queue()
        self.update_frame()

        threading.Thread(target=self._camera_worker, daemon=True).start()
        threading.Thread(target=self._inference_worker, daemon=True).start()
        threading.Thread(target=self._initialize_backend, daemon=True).start()

        # Сканирование списка камер не блокирует окно.
        self.root.after(300, self.refresh_camera_list)

        self.root.deiconify()
        self.root.lift()
        try:
            self.root.attributes("-topmost", True)
            self.root.after(500, lambda: self._safe_disable_topmost())
        except tk.TclError:
            pass

    def _safe_disable_topmost(self):
        if self._closing:
            return
        try:
            self.root.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _initialize_backend(self):
        if not self.backend_init_lock.acquire(blocking=False):
            return

        try:
            self.backend_ready.clear()
            self.backend_error = None
            self.ui_queue.put(lambda: self._set_backend_status("ИИ/БД: загрузка...", "#F59E0B"))

            logging.info(f"🚀 Вычислительное устройство: {self.device}")

            # ResNet может впервые скачивать веса. Это происходит в фоне.
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            model = model.to(self.device)
            model.eval()
            preprocess = weights.transforms()

            database = DatabaseManager(QDRANT_URL, COLLECTION_NAME, BASE_DIR)

            # Публикуем готовые объекты только после полного успеха.
            self.resnet = model
            self.preprocess = preprocess
            self.db = database
            self.backend_ready.set()

            logging.info("✅ ИИ и Qdrant успешно инициализированы")
            self.ui_queue.put(self._backend_ready_ui)

        except Exception as e:
            self.backend_error = str(e)
            self.backend_ready.clear()
            logging.exception("❌ Ошибка инициализации ИИ/БД")
            self.ui_queue.put(lambda err=str(e): self._backend_error_ui(err))

        finally:
            self.backend_init_lock.release()

    def _set_backend_status(self, text, color):
        if self._closing:
            return
        try:
            self.lbl_backend_status.config(text=text, fg=color)
        except Exception:
            pass

    def _backend_ready_ui(self):
        if self._closing:
            return
        self._set_backend_status("ИИ/БД: готово", "#10B981")
        self.lbl_reg_status.config(text="✅ ИИ и база готовы", fg="#10B981")
        self.load_gallery()

    def _backend_error_ui(self, error_text):
        if self._closing:
            return
        short = error_text if len(error_text) <= 120 else error_text[:117] + "..."
        self._set_backend_status("ИИ/БД: ошибка", "#EF4444")
        self.lbl_reg_status.config(text=f"❌ ИИ/БД: {short}", fg="#EF4444")
        # Локальные фото всё равно показываем.
        self.load_gallery()

    def setup_top_bar(self):
        # Две отдельные строки. В предыдущей версии все элементы находились
        # только в одной строке, поэтому ползунков физически не было в UI.
        header = tk.Frame(self.root, bg="#1f1f1f")
        header.pack(fill=tk.X, side=tk.TOP)

        top_bar = tk.Frame(
            header,
            bg="#1f1f1f",
            pady=7,
            padx=15,
        )
        top_bar.pack(fill=tk.X)

        tk.Label(
            top_bar,
            text="Активная камера:",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#1f1f1f",
        ).pack(side=tk.LEFT, padx=(0, 10))

        camera_options = [f"Камера {idx}" for idx in self.available_cameras]

        self.cb_camera = ttk.Combobox(
            top_bar,
            values=camera_options,
            state="readonly",
            width=15,
            font=("Arial", 10),
        )
        if camera_options:
            self.cb_camera.current(0)
        self.cb_camera.pack(side=tk.LEFT)
        self.cb_camera.bind("<<ComboboxSelected>>", self.on_camera_change)

        self.btn_refresh_cameras = tk.Button(
            top_bar,
            text="Обновить список",
            command=self.refresh_camera_list,
            bg="#374151",
            fg="white",
            font=("Arial", 9, "bold"),
            relief=tk.FLAT,
        )
        self.btn_refresh_cameras.pack(side=tk.LEFT, padx=10)

        self.lbl_camera_status = tk.Label(
            top_bar,
            text="камера: запуск...",
            font=("Arial", 9),
            fg="#F59E0B",
            bg="#1f1f1f",
        )
        self.lbl_camera_status.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_backend_status = tk.Label(
            top_bar,
            text="ИИ/БД: загрузка...",
            font=("Arial", 9, "bold"),
            fg="#F59E0B",
            bg="#1f1f1f",
        )
        self.lbl_backend_status.pack(side=tk.LEFT, padx=10)

        tk.Label(
            top_bar,
            text=f"  База: {BASE_DIR}",
            font=("Arial", 8),
            fg="#9CA3AF",
            bg="#1f1f1f",
        ).pack(side=tk.LEFT, padx=10)

        # -------------------------------------------------------------
        # ВИДИМАЯ ПАНЕЛЬ ФОКУСА И МАСШТАБА
        # -------------------------------------------------------------
        controls_bar = tk.Frame(
            header,
            bg="#292929",
            pady=6,
            padx=15,
        )
        controls_bar.pack(fill=tk.X)

        tk.Label(
            controls_bar,
            text="Камера:",
            font=("Arial", 10, "bold"),
            fg="#D1D5DB",
            bg="#292929",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.var_autofocus = tk.BooleanVar(value=self.autofocus_enabled)
        self.chk_autofocus = tk.Checkbutton(
            controls_bar,
            text="Автофокус",
            variable=self.var_autofocus,
            command=self.on_autofocus_toggle,
            bg="#292929",
            fg="white",
            activebackground="#292929",
            activeforeground="white",
            selectcolor="#1f1f1f",
            font=("Arial", 10, "bold"),
            bd=0,
            highlightthickness=0,
        )
        self.chk_autofocus.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(
            controls_bar,
            text="Ручной фокус:",
            font=("Arial", 9),
            fg="#D1D5DB",
            bg="#292929",
        ).pack(side=tk.LEFT)

        self.scale_focus = tk.Scale(
            controls_bar,
            from_=0,
            to=510,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=220,
            showvalue=False,
            command=self.on_manual_focus_change,
            bg="#292929",
            fg="white",
            troughcolor="#4B5563",
            activebackground="#D1D5DB",
            highlightthickness=0,
            bd=0,
        )
        self.scale_focus.set(self.manual_focus_value)
        self.scale_focus.config(state=tk.DISABLED)
        self.scale_focus.pack(side=tk.LEFT, padx=(6, 4))

        self.lbl_focus_value = tk.Label(
            controls_bar,
            text=str(self.manual_focus_value),
            width=4,
            anchor=tk.W,
            font=("Arial", 9, "bold"),
            fg="#9CA3AF",
            bg="#292929",
        )
        self.lbl_focus_value.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_focus_status = tk.Label(
            controls_bar,
            text="AF: включён",
            width=15,
            anchor=tk.W,
            font=("Arial", 9),
            fg="#10B981",
            bg="#292929",
        )
        self.lbl_focus_status.pack(side=tk.LEFT, padx=(0, 18))

        separator = tk.Frame(controls_bar, width=1, height=28, bg="#4B5563")
        separator.pack(side=tk.LEFT, padx=(0, 18))
        separator.pack_propagate(False)

        tk.Label(
            controls_bar,
            text="Область сканирования:",
            font=("Arial", 10, "bold"),
            fg="#D1D5DB",
            bg="#292929",
        ).pack(side=tk.LEFT)

        self.scale_zoom = tk.Scale(
            controls_bar,
            from_=1.0,
            to=4.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=300,
            showvalue=False,
            command=self.on_scan_zoom_change,
            bg="#292929",
            fg="white",
            troughcolor="#4B5563",
            activebackground="#D1D5DB",
            highlightthickness=0,
            bd=0,
        )
        self.scale_zoom.set(self.scan_zoom)
        self.scale_zoom.pack(side=tk.LEFT, padx=(8, 5))

        self.lbl_zoom_value = tk.Label(
            controls_bar,
            text=f"{self.scan_zoom:.1f}x",
            width=5,
            anchor=tk.W,
            font=("Arial", 10, "bold"),
            fg="#FBBF24",
            bg="#292929",
        )
        self.lbl_zoom_value.pack(side=tk.LEFT)

    def on_autofocus_toggle(self):
        enabled = bool(self.var_autofocus.get())
        with self.camera_settings_lock:
            self.autofocus_enabled = enabled
            self.camera_settings_dirty.set()

        self.scale_focus.config(state=tk.DISABLED if enabled else tk.NORMAL)
        self.lbl_focus_status.config(
            text="AF: включён" if enabled else "AF: выключен",
            fg="#10B981" if enabled else "#FBBF24",
        )

    def on_manual_focus_change(self, value):
        try:
            focus_value = int(round(float(value)))
        except (TypeError, ValueError):
            return

        focus_value = max(0, min(510, focus_value))
        self.lbl_focus_value.config(text=str(focus_value))

        with self.camera_settings_lock:
            self.manual_focus_value = focus_value
            if not self.autofocus_enabled:
                self.camera_settings_dirty.set()

    def on_scan_zoom_change(self, value):
        try:
            zoom = float(value)
        except (TypeError, ValueError):
            return

        zoom = max(1.0, min(4.0, zoom))
        self.scan_zoom = zoom
        self.lbl_zoom_value.config(text=f"{zoom:.1f}x")

        # После изменения ROI старое подтверждение больше не относится к
        # текущей области, поэтому начинаем распознавание заново.
        self.candidate_name = None
        self.confirm_counter = 0
        self.patience_counter = 0
        self.is_detected = False
        self.last_spoken_name = None
        self.scan_line_y = 0
        self._last_scanner_ui_state = None

    def _apply_camera_settings(self, cap):
        """Применяет фокус в camera worker, а не в Tkinter-потоке."""
        if cap is None:
            return

        # Сначала снимаем dirty. Если пользователь изменит ползунок во время
        # применения, callback снова выставит Event и изменение не потеряется.
        self.camera_settings_dirty.clear()
        with self.camera_settings_lock:
            autofocus = bool(self.autofocus_enabled)
            focus_value = int(self.manual_focus_value)

        auto_prop = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
        focus_prop = getattr(cv2, "CAP_PROP_FOCUS", None)
        auto_ok = None
        focus_ok = None
        actual_focus = None

        try:
            if auto_prop is not None:
                auto_ok = bool(cap.set(auto_prop, 1 if autofocus else 0))
        except Exception:
            auto_ok = False
            logging.exception("Не удалось изменить автофокус")

        # При ручном режиме сначала выключаем AF, затем задаём фокус.
        if not autofocus and focus_prop is not None:
            try:
                focus_ok = bool(cap.set(focus_prop, focus_value))
                actual_focus = cap.get(focus_prop)
            except Exception:
                focus_ok = False
                logging.exception("Не удалось изменить ручной фокус")

        if autofocus:
            if auto_prop is None or auto_ok is False:
                status_text = "AF: не поддерживается"
                status_color = "#EF4444"
            else:
                status_text = "AF: включён"
                status_color = "#10B981"
        else:
            if focus_prop is None or focus_ok is False:
                status_text = "Фокус: не поддерживается"
                status_color = "#EF4444"
            else:
                if actual_focus is not None and actual_focus >= 0:
                    status_text = f"Фокус: {int(round(actual_focus))}"
                else:
                    status_text = f"Фокус: {focus_value}"
                status_color = "#FBBF24"

        if self.is_running.is_set():
            self.ui_queue.put(
                lambda text=status_text, color=status_color:
                self.lbl_focus_status.config(text=text, fg=color)
            )

    def setup_tabs(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "TNotebook",
            background="#2b2b2b",
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#374151",
            foreground="white",
            padding=[20, 10],
            font=("Arial", 12, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#D32F2F")],
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH)

        self.tab_scanner = tk.Frame(
            self.notebook,
            bg="#2b2b2b",
        )
        self.tab_register = tk.Frame(
            self.notebook,
            bg="#2b2b2b",
        )
        self.tab_gallery = tk.Frame(
            self.notebook,
            bg="#E5E7EB",
        )
        self.tab_metadata = tk.Frame(
            self.notebook,
            bg="#2b2b2b",
        )

        self.notebook.add(
            self.tab_scanner,
            text="Сканер",
        )
        self.notebook.add(
            self.tab_register,
            text="Регистрация",
        )
        self.notebook.add(
            self.tab_gallery,
            text="Управление базой",
        )
        self.notebook.add(
            self.tab_metadata,
            text="⚙️ Справочники",
        )

        self.notebook.bind(
            "<<NotebookTabChanged>>",
            self.on_tab_changed,
        )

    def setup_scanner_ui(self, parent):
        left_panel = tk.Frame(
            parent,
            width=350,
            bg="#E5E7EB",
            padx=20,
            pady=20,
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            left_panel,
            text="🔍 РЕЗУЛЬТАТ",
            font=("Arial", 14, "bold"),
            bg="#E5E7EB",
            fg="#333",
        ).pack(pady=(0, 10))

        self.lbl_best_img = tk.Label(
            left_panel,
            image=self.blank_image,
            bg="#E5E7EB",
        )
        self.lbl_best_img.pack(pady=10)

        self.lbl_part = tk.Label(
            left_panel,
            text="Деталь: ...",
            font=("Arial", 16, "bold"),
            bg="#E5E7EB",
            fg="#1D4ED8",
            wraplength=300,
        )
        self.lbl_part.pack(anchor=tk.W, pady=10)

        self.lbl_score = tk.Label(
            left_panel,
            text="Точность: 0%",
            font=("Arial", 14, "bold"),
            bg="#E5E7EB",
            fg="#047857",
        )
        self.lbl_score.pack(anchor=tk.W, pady=10)

        self.status_box = tk.Label(
            left_panel,
            text="СКАНИРОВАНИЕ...",
            font=("Arial", 14, "bold"),
            bg="#F59E0B",
            fg="white",
            pady=10,
        )
        self.status_box.pack(
            fill=tk.X,
            side=tk.BOTTOM,
            pady=20,
        )

        self.video_label_scanner = tk.Label(
            parent,
            bg="#2b2b2b",
        )
        self.video_label_scanner.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        right_panel = tk.Frame(
            parent,
            width=300,
            bg="#374151",
            padx=15,
            pady=20,
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            right_panel,
            text="БЛИЖАЙШИЕ СОВПАДЕНИЯ",
            font=("Arial", 12, "bold"),
            bg="#374151",
            fg="#D1D5DB",
        ).pack(pady=(0, 20))

        self.lbl_sim1_img = tk.Label(
            right_panel,
            image=self.blank_sim,
            bg="#374151",
        )
        self.lbl_sim1_img.pack()

        self.lbl_sim1_name = tk.Label(
            right_panel,
            text="---",
            font=("Arial", 11, "bold"),
            bg="#374151",
            fg="#9CA3AF",
        )
        self.lbl_sim1_name.pack(pady=2)

        self.lbl_sim2_img = tk.Label(
            right_panel,
            image=self.blank_sim,
            bg="#374151",
        )
        self.lbl_sim2_img.pack(pady=(15, 0))

        self.lbl_sim2_name = tk.Label(
            right_panel,
            text="---",
            font=("Arial", 11, "bold"),
            bg="#374151",
            fg="#9CA3AF",
        )
        self.lbl_sim2_name.pack(pady=2)

    def setup_register_ui(self, parent):
        self.video_label_register = tk.Label(
            parent,
            bg="#2b2b2b",
        )
        self.video_label_register.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        right_panel = tk.Frame(
            parent,
            width=400,
            bg="#E5E7EB",
            padx=30,
            pady=30,
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            right_panel,
            text="⚙️ РЕГИСТРАЦИЯ ЭТАЛОНА",
            font=("Arial", 16, "bold"),
            bg="#E5E7EB",
            fg="#111827",
        ).pack(pady=(0, 20))

        tk.Label(
            right_panel,
            text="Тип сборки:",
            bg="#E5E7EB",
            fg="#4B5563",
            font=("Arial", 12, "bold"),
        ).pack(anchor=tk.W)

        self.cb_type = ttk.Combobox(
            right_panel,
            font=("Arial", 14),
            state="readonly",
        )
        self.cb_type.pack(
            fill=tk.X,
            pady=(5, 15),
        )
        self.cb_type.bind(
            "<<ComboboxSelected>>",
            self.update_part_list,
        )

        tk.Label(
            right_panel,
            text="Название детали:",
            bg="#E5E7EB",
            fg="#4B5563",
            font=("Arial", 12, "bold"),
        ).pack(anchor=tk.W)

        self.cb_part = ttk.Combobox(
            right_panel,
            font=("Arial", 14),
        )
        self.cb_part.pack(
            fill=tk.X,
            pady=(5, 25),
        )

        self.btn_save_reference = tk.Button(
            right_panel,
            text="СОХРАНИТЬ РАКУРС",
            bg="#D32F2F",
            fg="white",
            font=("Arial", 14, "bold"),
            relief=tk.FLAT,
            command=self.save_reference_from_ui,
        )
        self.btn_save_reference.pack(fill=tk.X, ipady=10)

        self.lbl_reg_status = tk.Label(
            right_panel,
            text="",
            bg="#E5E7EB",
            font=("Arial", 12, "bold"),
        )
        self.lbl_reg_status.pack(pady=15)

    def setup_gallery_ui(self, parent):
        top_frame = tk.Frame(
            parent,
            bg="#F9FAFB",
            pady=15,
            padx=20,
        )
        top_frame.pack(fill=tk.X)

        tk.Button(
            top_frame,
            text="Обновить список",
            command=self.load_gallery,
            font=("Arial", 12),
            bg="#374151",
            fg="white",
        ).pack(side=tk.LEFT)

        tk.Button(
            top_frame,
            text="Синхронизировать",
            command=self.sync_database_ui,
            font=("Arial", 12),
            bg="#059669",
            fg="white",
        ).pack(side=tk.LEFT, padx=10)

        self.tree = ttk.Treeview(
            parent,
            columns=("type", "part", "count", "files"),
            show="headings",
            height=18,
        )

        self.tree.heading(
            "type",
            text="Тип сборки",
        )
        self.tree.heading(
            "part",
            text="Деталь",
        )
        self.tree.heading(
            "count",
            text="Записей",
        )
        self.tree.heading(
            "files",
            text="Фото",
        )

        self.tree.column("type", width=220)
        self.tree.column("part", width=350)
        self.tree.column(
            "count",
            width=120,
            anchor=tk.CENTER,
        )
        self.tree.column(
            "files",
            width=120,
            anchor=tk.CENTER,
        )

        self.tree.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=10,
        )

        btn_frame = tk.Frame(
            parent,
            bg="#E5E7EB",
            pady=10,
        )
        btn_frame.pack(
            fill=tk.X,
            padx=20,
        )

        tk.Button(
            btn_frame,
            text="Просмотреть фото",
            command=self.view_part_photos,
            font=("Arial", 12, "bold"),
            bg="#1D4ED8",
            fg="white",
        ).pack(side=tk.LEFT, padx=10)

        self.btn_del_all = tk.Button(
            btn_frame,
            text="Удалить всю деталь",
            command=self.delete_selected_image,
            font=("Arial", 12, "bold"),
            bg="#D32F2F",
            fg="white",
        )
        self.btn_del_all.pack(side=tk.RIGHT)

        self.lbl_gallery_status = tk.Label(
            parent,
            text="",
            bg="#E5E7EB",
            font=("Arial", 11, "bold"),
        )
        self.lbl_gallery_status.pack(pady=5)

    def setup_metadata_ui(self, parent):
        main_frame = tk.Frame(
            parent,
            bg="#2b2b2b",
            padx=20,
            pady=20,
        )
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        tk.Label(
            main_frame,
            text="⚙️ Управление справочниками",
            font=("Arial", 16, "bold"),
            bg="#2b2b2b",
            fg="white",
        ).pack(pady=(0, 20))

        container = tk.Frame(
            main_frame,
            bg="#2b2b2b",
        )
        container.pack(
            fill=tk.BOTH,
            expand=True,
        )

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        f_type = tk.LabelFrame(
            container,
            text=" Типы сборки ",
            font=("Arial", 11, "bold"),
            bg="#374151",
            fg="white",
            padx=15,
            pady=15,
        )
        f_type.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
        )

        self.lb_types = tk.Listbox(
            f_type,
            font=("Arial", 12),
            height=12,
        )
        self.lb_types.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(0, 10),
        )

        self.ent_type = tk.Entry(
            f_type,
            font=("Arial", 12),
        )
        self.ent_type.pack(
            fill=tk.X,
            pady=(0, 10),
        )

        b_frame2 = tk.Frame(
            f_type,
            bg="#374151",
        )
        b_frame2.pack(fill=tk.X)

        tk.Button(
            b_frame2,
            text="+",
            bg="#10B981",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.meta_add_type,
        ).pack(side=tk.LEFT, expand=True)

        tk.Button(
            b_frame2,
            text="X",
            bg="#D32F2F",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.meta_delete_type,
        ).pack(side=tk.LEFT, expand=True)

        f_part = tk.LabelFrame(
            container,
            text=" Детали ",
            font=("Arial", 11, "bold"),
            bg="#374151",
            fg="white",
            padx=15,
            pady=15,
        )
        f_part.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=10,
        )

        self.cb_meta_type = ttk.Combobox(
            f_part,
            font=("Arial", 11),
            state="readonly",
        )
        self.cb_meta_type.pack(
            fill=tk.X,
            pady=(2, 10),
        )
        self.cb_meta_type.bind(
            "<<ComboboxSelected>>",
            self.load_meta_parts,
        )

        self.lb_parts = tk.Listbox(
            f_part,
            font=("Arial", 12),
            height=9,
        )
        self.lb_parts.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(0, 10),
        )

        self.ent_part = tk.Entry(
            f_part,
            font=("Arial", 12),
        )
        self.ent_part.pack(
            fill=tk.X,
            pady=(0, 10),
        )

        b_frame3 = tk.Frame(
            f_part,
            bg="#374151",
        )
        b_frame3.pack(fill=tk.X)

        tk.Button(
            b_frame3,
            text="+",
            bg="#10B981",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.meta_add_part,
        ).pack(side=tk.LEFT, expand=True)

        tk.Button(
            b_frame3,
            text="X",
            bg="#D32F2F",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.meta_delete_part,
        ).pack(side=tk.LEFT, expand=True)

    # =================================================================
    # CAMERA
    # =================================================================

    def refresh_camera_list(self):
        if self._closing:
            return
        if not self.camera_refresh_lock.acquire(blocking=False):
            return

        self.camera_scan_running.set()
        self.btn_refresh_cameras.config(state=tk.DISABLED, text="Поиск...")
        self.lbl_camera_status.config(text="камера: поиск...", fg="#F59E0B")

        def worker():
            try:
                cameras = detect_available_cameras(max_tested=8)
                if not cameras:
                    cameras = [0]
                self.ui_queue.put(lambda cams=cameras: self._apply_camera_list(cams))
            except Exception as e:
                logging.exception("Ошибка поиска камер")
                self.ui_queue.put(lambda err=str(e): self._camera_scan_error(err))
            finally:
                self.camera_scan_running.clear()
                self.camera_refresh_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def _apply_camera_list(self, cameras):
        if self._closing:
            return

        previous = self.requested_camera_index
        self.available_cameras = list(cameras)
        self.cb_camera["values"] = [f"Камера {idx}" for idx in cameras]

        selected = previous if previous in cameras else cameras[0]
        self.cb_camera.set(f"Камера {selected}")
        self.requested_camera_index = selected

        self.btn_refresh_cameras.config(state=tk.NORMAL, text="Обновить список")
        self.lbl_camera_status.config(
            text=f"найдено: {len(cameras)}",
            fg="#10B981" if cameras else "#EF4444",
        )

    def _camera_scan_error(self, error_text):
        if self._closing:
            return
        self.btn_refresh_cameras.config(state=tk.NORMAL, text="Обновить список")
        self.lbl_camera_status.config(text="камера: ошибка поиска", fg="#EF4444")
        logging.error(error_text)

    def on_camera_change(self, event):
        selected = self.cb_camera.get()

        if selected:
            try:
                self.requested_camera_index = int(
                    selected.split()[-1]
                )
            except ValueError:
                pass

    def _camera_worker(self):
        logging.info("Фоновый поток камеры запущен.")
        cap = None
        active_index = None
        read_failures = 0

        while self.is_running.is_set():
            # Во время полного сканирования списка камер освобождаем активную
            # камеру, чтобы probe-поток мог проверить тот же индекс.
            if self.camera_scan_running.is_set():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                active_index = None
                time.sleep(0.05)
                continue

            requested = self.requested_camera_index
            if requested is None:
                time.sleep(0.1)
                continue

            if cap is None or requested != active_index:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None

                active_index = requested
                read_failures = 0
                logging.info(f"Подключение к камере {active_index}...")

                cap, first_frame = open_camera(active_index, probe=True)
                if cap is None:
                    self.ui_queue.put(
                        lambda idx=active_index: self.lbl_camera_status.config(
                            text=f"камера {idx}: нет сигнала",
                            fg="#EF4444",
                        )
                    )
                    active_index = None
                    time.sleep(0.5)
                    continue

                # Настройки камеры применяются только здесь, в одном потоке.
                self._apply_camera_settings(cap)

                # После включения автофокуса/смены режима пропускаем несколько
                # кадров, чтобы фокус и экспозиция успели стабилизироваться.
                stable_frame = first_frame
                for _ in range(8):
                    try:
                        ret_warm, frame_warm = cap.read()
                        if ret_warm and frame_warm is not None and frame_warm.size > 0:
                            stable_frame = frame_warm
                    except Exception:
                        break
                    time.sleep(0.015)

                if stable_frame is not None:
                    with self.state_lock:
                        self.current_raw_frame = stable_frame

                self.ui_queue.put(
                    lambda idx=active_index: self.lbl_camera_status.config(
                        text=f"камера {idx}: работает",
                        fg="#10B981",
                    )
                )
                logging.info(f"✅ Камера {active_index} успешно открыта")

            if self.camera_settings_dirty.is_set():
                self._apply_camera_settings(cap)

            try:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    read_failures = 0
                    with self.state_lock:
                        self.current_raw_frame = frame
                else:
                    read_failures += 1
                    time.sleep(0.03)
            except Exception:
                read_failures += 1
                logging.exception("Ошибка чтения кадра")
                time.sleep(0.05)

            # Камера иногда остаётся "открытой", но постоянно возвращает пустой
            # кадр. Переподключаем её автоматически.
            if read_failures >= 20:
                logging.warning(f"Переподключение камеры {active_index}")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                active_index = None
                read_failures = 0
                time.sleep(0.25)

            time.sleep(0.005)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        logging.info("Фоновый поток камеры остановлен.")

    def process_ui_queue(self):
        if self._closing:
            return

        processed = 0
        try:
            while processed < 50:
                task = self.ui_queue.get_nowait()
                try:
                    task()
                except tk.TclError:
                    if not self._closing:
                        logging.exception("Tkinter UI-задача завершилась ошибкой")
                except Exception:
                    logging.exception("Ошибка UI-задачи")
                processed += 1
        except queue.Empty:
            pass

        if not self._closing:
            try:
                self.root.after(30, self.process_ui_queue)
            except tk.TclError:
                pass

    def _inference_worker(self):
        while self.is_running.is_set():
            try:
                crop = self.inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if crop is None or not self.backend_ready.is_set():
                continue

            try:
                if self.db is None or self.resnet is None or self.preprocess is None:
                    continue

                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                with self.model_lock:
                    with torch.inference_mode():
                        tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
                        query_vector = (
                            self.resnet(tensor)[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        )

                search_result = self.db.search(query_vector, limit=3)

                if self.is_running.is_set():
                    self.ui_queue.put(
                        lambda r=search_result: self.process_inference_result(r)
                    )

            except Exception as e:
                logging.exception(f"Ошибка инференса: {e}")
                # Не спамим UI на каждом кадре: состояние камеры продолжает жить.
                time.sleep(0.1)

    def update_frame(self):
        if self._closing:
            return

        try:
            with self.state_lock:
                frame = self.current_raw_frame.copy() if self.current_raw_frame is not None else None

            if frame is not None and frame.size > 0:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Ползунок меняет именно центральную область сканирования.
                # 1.0x -> вся доступная квадратная область; 2.0x -> прежний
                # размер; 4.0x -> сильное цифровое приближение к центру.
                zoom = max(1.0, min(4.0, float(self.scan_zoom)))
                size = max(32, int(min(h, w) / zoom))
                size = min(size, h, w)
                start_y = max(0, (h - size) // 2)
                start_x = max(0, (w - size) // 2)
                crop = frame[start_y:start_y + size, start_x:start_x + size]

                if crop.size > 0:
                    resized_for_ai = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)
                    with self.state_lock:
                        self.latest_raw_crop = crop.copy()
                        self.latest_crop = resized_for_ai

                    self.frame_counter += 1
                    if self.frame_counter % 10 == 0 and self.backend_ready.is_set():
                        try:
                            self.inference_queue.put_nowait(resized_for_ai.copy())
                        except queue.Full:
                            pass

                    try:
                        tab_id = self.notebook.index(self.notebook.select())
                    except tk.TclError:
                        return

                    display_crop = crop.copy()
                    crop_h, crop_w = display_crop.shape[:2]

                    if tab_id == 0:
                        if not self.is_detected:
                            step = max(1, int(crop_h * 0.05))
                            self.scan_line_y += step * self.scan_line_dir
                            if self.scan_line_y >= crop_h or self.scan_line_y <= 0:
                                self.scan_line_dir *= -1
                                self.scan_line_y = max(0, min(self.scan_line_y, crop_h - 1))

                            cv2.line(
                                display_crop,
                                (0, self.scan_line_y),
                                (crop_w, self.scan_line_y),
                                (0, 255, 0),
                                3,
                            )
                            cv2.drawMarker(
                                display_crop,
                                (crop_w // 2, crop_h // 2),
                                (0, 165, 255),
                                cv2.MARKER_CROSS,
                                40,
                                2,
                            )
                        else:
                            cv2.rectangle(
                                display_crop,
                                (0, 0),
                                (crop_w - 1, crop_h - 1),
                                (0, 255, 0),
                                8,
                            )
                    elif tab_id == 1:
                        cv2.drawMarker(
                            display_crop,
                            (crop_w // 2, crop_h // 2),
                            (255, 255, 255),
                            cv2.MARKER_CROSS,
                            40,
                            2,
                        )

                    # Нет смысла рендерить видео на вкладках, где его не видно.
                    if tab_id in (0, 1):
                        ui_frame = cv2.resize(
                            display_crop,
                            (650, 650),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        img_tk = ImageTk.PhotoImage(
                            image=Image.fromarray(cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB))
                        )

                        if tab_id == 0:
                            self.video_label_scanner.imgtk = img_tk
                            self.video_label_scanner.configure(image=img_tk)
                            self.sync_scanner_ui()
                        else:
                            self.video_label_register.imgtk = img_tk
                            self.video_label_register.configure(image=img_tk)

        except tk.TclError:
            if not self._closing:
                logging.exception("Ошибка Tkinter в update_frame")
        except Exception:
            # Раньше любая единичная ошибка здесь навсегда останавливала after().
            logging.exception("Ошибка обновления видеокадра")

        finally:
            if not self._closing:
                try:
                    self.root.after(30, self.update_frame)
                except tk.TclError:
                    pass

    def process_inference_result(self, search_result):
        points = getattr(
            search_result,
            "points",
            None,
        )

        if points:
            best = points[0]

            if best.score >= self.MIN_SCORE:
                payload = best.payload or {}

                part_name = payload.get(
                    "name",
                    "N/A",
                )

                self.patience_counter = 0

                if part_name == self.candidate_name:
                    self.confirm_counter += 1
                else:
                    self.candidate_name = part_name
                    self.confirm_counter = 1

                req_frames = (
                    self.CONFIRM_FRAMES_FAST
                    if best.score >= 0.85
                    else self.CONFIRM_FRAMES_SLOW
                )

                if self.confirm_counter >= req_frames:
                    self.is_detected = True

                    v_type = payload.get(
                        "type",
                        "",
                    )

                    image_path = self.resolve_payload_image(
                        payload,
                        best.id,
                        v_type,
                        part_name,
                    )

                    with self.state_lock:
                        self.scan_results["part"] = (
                            f"Деталь: {part_name}"
                        )
                        self.scan_results["score"] = (
                            f"Точность: "
                            f"{int(best.score * 100)}%"
                        )

                        self.scan_results["path_best"] = (
                            image_path
                        )

                        self.scan_results["status_text"] = (
                            "✅ РАСПОЗНАНО"
                            if best.score >= 0.85
                            else "⚠️ СЛАБОЕ СОВПАДЕНИЕ"
                        )

                        self.scan_results["status_bg"] = (
                            "#10B981"
                            if best.score >= 0.85
                            else "#D97706"
                        )

                        self.scan_results[
                            "sim1_name"
                        ] = "---"
                        self.scan_results[
                            "sim1_score"
                        ] = "0%"
                        self.scan_results[
                            "path_sim1"
                        ] = ""

                        self.scan_results[
                            "sim2_name"
                        ] = "---"
                        self.scan_results[
                            "sim2_score"
                        ] = "0%"
                        self.scan_results[
                            "path_sim2"
                        ] = ""

                        if len(points) > 1:
                            p1 = points[1]
                            p1_payload = p1.payload or {}

                            p1_name = p1_payload.get(
                                "name",
                                "---",
                            )
                            p1_type = p1_payload.get(
                                "type",
                                "",
                            )

                            self.scan_results[
                                "sim1_name"
                            ] = p1_name
                            self.scan_results[
                                "sim1_score"
                            ] = (
                                f"{int(p1.score * 100)}%"
                            )
                            self.scan_results[
                                "path_sim1"
                            ] = self.resolve_payload_image(
                                p1_payload,
                                p1.id,
                                p1_type,
                                p1_name,
                            )

                        if len(points) > 2:
                            p2 = points[2]
                            p2_payload = p2.payload or {}

                            p2_name = p2_payload.get(
                                "name",
                                "---",
                            )
                            p2_type = p2_payload.get(
                                "type",
                                "",
                            )

                            self.scan_results[
                                "sim2_name"
                            ] = p2_name
                            self.scan_results[
                                "sim2_score"
                            ] = (
                                f"{int(p2.score * 100)}%"
                            )
                            self.scan_results[
                                "path_sim2"
                            ] = self.resolve_payload_image(
                                p2_payload,
                                p2.id,
                                p2_type,
                                p2_name,
                            )

                    if part_name != self.last_spoken_name:
                        self.play_success_sound(
                            part_name
                        )
                        self.last_spoken_name = part_name

            else:
                self.handle_detection_loss()
        else:
            self.handle_detection_loss()

    def resolve_payload_image(
        self,
        payload,
        point_id,
        v_type,
        part,
    ):
        """
        В первую очередь используем image_path.
        Для старых записей используем старую структуру папок.
        """
        image_path = payload.get("image_path")

        if image_path:
            image_path = os.path.abspath(
                str(image_path)
            )

            if os.path.exists(image_path):
                return image_path

        if v_type and part:
            fallback = reference_file(
                v_type,
                part,
                str(point_id),
            )

            if os.path.exists(fallback):
                return fallback

        # Дополнительный поиск по имени UUID.
        if v_type and part:
            p_dir = reference_dir(
                v_type,
                part,
            )

            if os.path.isdir(p_dir):
                for filename in os.listdir(p_dir):
                    if (
                        os.path.splitext(filename)[0]
                        == str(point_id)
                    ):
                        candidate = os.path.join(
                            p_dir,
                            filename,
                        )
                        if os.path.exists(candidate):
                            return candidate

        return ""

    def handle_detection_loss(self):
        self.candidate_name = None
        self.confirm_counter = 0

        if self.is_detected:
            self.patience_counter += 1

            if self.patience_counter >= self.MAX_PATIENCE:
                self.is_detected = False
                self.last_spoken_name = None

                with self.state_lock:
                    self.scan_results = {
                        "part": "Ожидание...",
                        "score": "0%",
                        "status_text": "СКАНИРОВАНИЕ...",
                        "status_bg": "#F59E0B",
                        "sim1_name": "---",
                        "sim1_score": "0%",
                        "path_sim1": "",
                        "sim2_name": "---",
                        "sim2_score": "0%",
                        "path_sim2": "",
                        "path_best": "",
                    }

    def sync_scanner_ui(self):
        with self.state_lock:
            sr = self.scan_results.copy()

        state = (
            sr.get("part"),
            sr.get("score"),
            sr.get("status_text"),
            sr.get("status_bg"),
            sr.get("sim1_name"),
            sr.get("sim1_score"),
            sr.get("path_sim1"),
            sr.get("sim2_name"),
            sr.get("sim2_score"),
            sr.get("path_sim2"),
            sr.get("path_best"),
        )

        if state == self._last_scanner_ui_state:
            return
        self._last_scanner_ui_state = state

        self.lbl_part.config(text=sr["part"])
        self.lbl_score.config(text=sr["score"])
        self.status_box.config(text=sr["status_text"], bg=sr["status_bg"])

        img_best = self.load_ref_image(sr["path_best"], (300, 300))
        self.lbl_best_img.config(image=img_best)
        self.lbl_best_img.image = img_best

        self.lbl_sim1_name.config(text=f"{sr['sim1_name']} ({sr['sim1_score']})")
        img_sim1 = self.load_ref_image(sr["path_sim1"], (200, 200))
        self.lbl_sim1_img.config(image=img_sim1)
        self.lbl_sim1_img.image = img_sim1

        self.lbl_sim2_name.config(text=f"{sr['sim2_name']} ({sr['sim2_score']})")
        img_sim2 = self.load_ref_image(sr["path_sim2"], (200, 200))
        self.lbl_sim2_img.config(image=img_sim2)
        self.lbl_sim2_img.image = img_sim2

    def create_blank_image(self, size, text):
        key = (tuple(size), str(text))
        cached = self._blank_cache.get(key) if hasattr(self, "_blank_cache") else None
        if cached is not None:
            return cached

        img = Image.new("RGB", size, color="#4B5563")
        draw = ImageDraw.Draw(img)

        try:
            bbox = draw.textbbox((0, 0), text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(str(text)) * 6
            text_h = 12

        draw.text(
            ((size[0] - text_w) // 2, (size[1] - text_h) // 2),
            text,
            fill=(200, 200, 200),
        )
        photo = ImageTk.PhotoImage(img)

        if hasattr(self, "_blank_cache"):
            self._blank_cache[key] = photo
        return photo

    def load_ref_image(self, path, size):
        if path and os.path.exists(path):
            try:
                abs_path = os.path.abspath(path)
                mtime = os.path.getmtime(abs_path)
                key = (abs_path, tuple(size), mtime)

                cached = self._image_cache.get(key)
                if cached is not None:
                    return cached

                with Image.open(abs_path) as source:
                    img = source.convert("RGB")
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                # Убираем старые версии этого же файла и ограничиваем кэш.
                for old_key in list(self._image_cache.keys()):
                    if old_key[0] == abs_path and old_key != key:
                        self._image_cache.pop(old_key, None)
                self._image_cache[key] = photo

                while len(self._image_cache) > 64:
                    first_key = next(iter(self._image_cache))
                    self._image_cache.pop(first_key, None)

                return photo
            except Exception:
                logging.exception(f"Не удалось открыть фото: {path}")

        return self.create_blank_image(size, "Нет файла")

    def load_registration_lists(self):
        types = load_list(
            "types.txt",
            ["metiz", "bigdetail"],
        )

        self.cb_type["values"] = types

        if types:
            if (
                not self.cb_type.get()
                or self.cb_type.get() not in types
            ):
                self.cb_type.set(types[0])

        self.update_part_list()

    def update_part_list(self, *args):
        v_type = self.cb_type.get()

        if not v_type:
            return

        parts = load_list(
            f"parts_{v_type}.txt",
            [f"Деталь_{v_type}"],
        )

        self.cb_part["values"] = parts

        if parts:
            if (
                not self.cb_part.get()
                or self.cb_part.get() not in parts
            ):
                self.cb_part.set(parts[0])

    def save_reference_from_ui(self):
        if self.save_in_progress:
            return

        v_type = self.cb_type.get().strip()
        part = self.cb_part.get().strip()

        with self.state_lock:
            raw_crop = self.latest_raw_crop.copy() if self.latest_raw_crop is not None else None
            crop = self.latest_crop.copy() if self.latest_crop is not None else None

        if not self.backend_ready.is_set() or self.db is None:
            self.lbl_reg_status.config(
                text="⏳ ИИ/БД ещё загружаются. Дождитесь статуса 'готово'.",
                fg="#D97706",
            )
            return

        if not v_type or not part or raw_crop is None or crop is None or crop.size == 0:
            self.lbl_reg_status.config(
                text="❌ Заполните поля и проверьте изображение с камеры!",
                fg="#EF4444",
            )
            return

        # Справочники — быстрые локальные операции, их можно делать в UI-потоке.
        types = load_list("types.txt", [])
        if v_type not in types:
            types.append(v_type)
            if not save_list("types.txt", types):
                self.lbl_reg_status.config(text="❌ Не удалось сохранить тип", fg="#EF4444")
                return

        parts = load_list(f"parts_{v_type}.txt", [])
        if part not in parts:
            parts.append(part)
            if not save_list(f"parts_{v_type}.txt", parts):
                self.lbl_reg_status.config(text="❌ Не удалось сохранить деталь", fg="#EF4444")
                return

        self.save_in_progress = True
        self.btn_save_reference.config(state=tk.DISABLED, text="СОХРАНЕНИЕ...")
        self.lbl_reg_status.config(text="⏳ Сохранение фото и БД...", fg="#D97706")

        def _save_worker():
            try:
                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                with self.model_lock:
                    with torch.inference_mode():
                        tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
                        vector = (
                            self.resnet(tensor)[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        )

                _, path = self.db.save_reference(raw_crop, vector, v_type, part)
                self.ui_queue.put(lambda p=path: self.registration_success(p))

            except Exception as e:
                logging.exception("Ошибка сохранения эталона")
                self.ui_queue.put(lambda err=str(e): self.registration_error(err))

        threading.Thread(target=_save_worker, daemon=True).start()

    def registration_success(self, path):
        self.save_in_progress = False
        self.btn_save_reference.config(state=tk.NORMAL, text="СОХРАНИТЬ РАКУРС")
        self.lbl_reg_status.config(
            text=f"✅ Эталон сохранён\n{os.path.basename(path)}",
            fg="#10B981",
        )
        self.load_gallery()
        self.load_registration_lists()

    def registration_error(self, error_text):
        self.save_in_progress = False
        self.btn_save_reference.config(state=tk.NORMAL, text="СОХРАНИТЬ РАКУРС")
        self.lbl_reg_status.config(
            text=f"❌ Ошибка сохранения\n{error_text}",
            fg="#EF4444",
        )

    def play_success_sound(self, text_to_speak):
        if os.name != "nt" or self._closing:
            return

        def _play():
            if not self.audio_lock.acquire(blocking=False):
                return
            try:
                safe_text = str(text_to_speak).replace("'", " ")
                ps_cmd = (
                    "[System.Console]::Beep(1200, 100); "
                    "Add-Type -AssemblyName System.Speech; "
                    "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                    f".Speak('{safe_text}')"
                )
                subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-WindowStyle",
                        "Hidden",
                        "-Command",
                        ps_cmd,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=6,
                    creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
            except subprocess.TimeoutExpired:
                logging.warning("Озвучивание превысило таймаут и было остановлено")
            except Exception as e:
                logging.error(f"Ошибка звука: {e}")
            finally:
                self.audio_lock.release()

        threading.Thread(target=_play, daemon=True).start()

    def on_tab_changed(self, event):
        tab_id = self.notebook.index(
            self.notebook.select()
        )

        if tab_id == 1:
            self.load_registration_lists()

        elif tab_id == 2:
            self.load_gallery()

        elif tab_id == 3:
            self.load_metadata_tab()

    # =================================================================
    # УПРАВЛЕНИЕ БАЗОЙ
    # =================================================================

    def load_gallery(self):
        if self._closing:
            return

        self.gallery_generation += 1
        generation = self.gallery_generation
        self.delete_all_confirmed = False

        self.btn_del_all.config(
            text="X Удалить всю деталь",
            bg="#D32F2F",
            state=tk.NORMAL,
        )

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.lbl_gallery_status.config(text="⏳ Загрузка базы данных...", fg="#D97706")

        def _fetch_worker():
            try:
                if self.db is not None and self.backend_ready.is_set():
                    groups = self.db.list_groups_from_qdrant()
                    source_note = ""
                else:
                    groups = [
                        {"type": t, "part": p, "count": 0, "files": files}
                        for t, p, files in scan_reference_groups(BASE_DIR)
                    ]
                    source_note = " (локальные фото; Qdrant не готов)"

                groups.sort(key=lambda x: (x["type"].lower(), x["part"].lower()))

                def _update_ui():
                    if self._closing or generation != self.gallery_generation:
                        return

                    for group in groups:
                        self.tree.insert(
                            "",
                            "end",
                            values=(
                                group["type"],
                                group["part"],
                                group["count"],
                                len(group["files"]),
                            ),
                        )

                    self.lbl_gallery_status.config(
                        text=f"Найдено деталей: {len(groups)}{source_note}",
                        fg="#047857" if groups else "#6B7280",
                    )
                    logging.info(f"📚 Управление базой: {len(groups)} групп{source_note}")

                self.ui_queue.put(_update_ui)

            except Exception as e:
                logging.exception("Ошибка загрузки управления базой")

                # Даже если Qdrant упал, попробуем показать физические фото.
                try:
                    groups = [
                        {"type": t, "part": p, "count": 0, "files": files}
                        for t, p, files in scan_reference_groups(BASE_DIR)
                    ]
                except Exception:
                    groups = []

                def _update_error(err=str(e), local_groups=groups):
                    if self._closing or generation != self.gallery_generation:
                        return
                    for group in local_groups:
                        self.tree.insert(
                            "",
                            "end",
                            values=(
                                group["type"],
                                group["part"],
                                group["count"],
                                len(group["files"]),
                            ),
                        )
                    self.lbl_gallery_status.config(
                        text=f"⚠️ Qdrant: {err}. Локальных деталей: {len(local_groups)}",
                        fg="#B45309",
                    )

                self.ui_queue.put(_update_error)

        threading.Thread(target=_fetch_worker, daemon=True).start()

    def sync_database_ui(self):
        if self._closing:
            return

        self.lbl_gallery_status.config(text="⏳ Синхронизация...", fg="#D97706")

        if self.db is None or not self.backend_ready.is_set():
            self.lbl_gallery_status.config(
                text="⏳ Повторное подключение ИИ/Qdrant...",
                fg="#D97706",
            )
            threading.Thread(target=self._initialize_backend, daemon=True).start()
            return

        def worker():
            try:
                self.db.repair_database()
                self.ui_queue.put(self._sync_database_success)
            except Exception as e:
                logging.exception("Ошибка синхронизации")
                self.ui_queue.put(
                    lambda err=str(e): self.lbl_gallery_status.config(
                        text=f"❌ {err}",
                        fg="#B91C1C",
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def _sync_database_success(self):
        if self._closing:
            return
        self.load_gallery()
        self.lbl_gallery_status.config(text="✅ База синхронизирована", fg="#047857")

    def delete_selected_image(self):
        selected = self.tree.selection()
        if not selected:
            self.lbl_gallery_status.config(text="Выберите деталь в списке", fg="#6B7280")
            return

        if self.db is None or not self.backend_ready.is_set():
            self.lbl_gallery_status.config(
                text="❌ Qdrant не готов. Сначала восстановите подключение кнопкой 'Синхронизировать'.",
                fg="#B91C1C",
            )
            return

        if not self.delete_all_confirmed:
            self.delete_all_confirmed = True
            self.btn_del_all.config(text="⚠️ НАЖМИТЕ ЕЩЕ РАЗ", bg="#B71C1C")
            return

        items = [self.tree.item(item, "values") for item in selected]
        self.tree.selection_remove(*selected)
        self.btn_del_all.config(state=tk.DISABLED, text="Удаление...")
        self.lbl_gallery_status.config(text="⏳ Удаление...", fg="#D97706")

        def _delete_process(items_to_delete):
            errors = []
            for item_values in items_to_delete:
                v_type, part = item_values[0], item_values[1]
                try:
                    self.db.delete_part(v_type, part)
                except Exception as e:
                    errors.append(f"{v_type}/{part}: {e}")

            self.ui_queue.put(lambda errs=errors: self._finish_delete_part(errs))

        threading.Thread(target=_delete_process, args=(items,), daemon=True).start()

    def _finish_delete_part(self, errors):
        if self._closing:
            return
        self.btn_del_all.config(state=tk.NORMAL, text="X Удалить всю деталь", bg="#D32F2F")
        self.delete_all_confirmed = False
        self._image_cache.clear()

        if errors:
            self.lbl_gallery_status.config(
                text="❌ Ошибки удаления: " + " | ".join(errors),
                fg="#B91C1C",
            )
        else:
            self.lbl_gallery_status.config(text="✅ Деталь удалена", fg="#047857")

        self.load_gallery()

    def view_part_photos(self):
        selected = self.tree.selection()

        if not selected:
            return

        values = self.tree.item(
            selected[0],
            "values",
        )

        v_type = values[0]
        part = values[1]

        p_path = reference_dir(
            v_type,
            part,
        )

        if not os.path.isdir(p_path):
            messagebox.showwarning(
                "Нет фото",
                "Папка с фотографиями не найдена.",
            )
            return

        photos = [
            f for f in os.listdir(p_path)
            if f.lower().endswith(
                DatabaseManager.IMAGE_EXTENSIONS
            )
        ]

        if not photos:
            messagebox.showwarning(
                "Нет фото",
                "Для этой детали фотографии отсутствуют.",
            )
            return

        viewer = tk.Toplevel(
            self.root
        )
        viewer.title(
            f"{v_type} / {part}"
        )
        viewer.geometry("700x700")
        viewer.configure(
            bg="#2b2b2b"
        )

        viewer.current_idx = 0
        viewer.photos_list = photos
        viewer.delete_confirmed = False

        lbl_info = tk.Label(
            viewer,
            text="",
            font=("Arial", 14, "bold"),
            bg="#2b2b2b",
            fg="white",
        )
        lbl_info.pack(pady=10)

        lbl_img = tk.Label(
            viewer,
            bg="#2b2b2b",
        )
        lbl_img.pack(
            expand=True
        )

        btn_frame = tk.Frame(
            viewer,
            bg="#2b2b2b",
        )
        btn_frame.pack(
            fill=tk.X,
            pady=15,
            padx=30,
        )

        def render_photo():
            viewer.delete_confirmed = False

            btn_del_photo.config(
                text="🗑️ Удалить фото",
                bg="#D32F2F",
            )

            if not viewer.photos_list:
                viewer.destroy()
                self.load_gallery()
                return

            viewer.current_idx %= len(
                viewer.photos_list
            )

            current_file = (
                viewer.photos_list[
                    viewer.current_idx
                ]
            )

            lbl_info.config(
                text=(
                    f"{part} — Фото "
                    f"{viewer.current_idx + 1} "
                    f"из "
                    f"{len(viewer.photos_list)}"
                )
            )

            img_path = os.path.join(
                p_path,
                current_file,
            )

            try:
                img = Image.open(
                    img_path
                ).convert("RGB")

                img.thumbnail(
                    (450, 450),
                    Image.Resampling.LANCZOS,
                )

                viewer.img_tk = (
                    ImageTk.PhotoImage(img)
                )

                lbl_img.config(
                    image=viewer.img_tk
                )

            except Exception as e:
                logging.exception(
                    f"Ошибка открытия {img_path}"
                )
                lbl_img.config(
                    image=self.blank_sim
                )

        def delete_current_photo():
            if not viewer.photos_list:
                return

            if self.db is None or not self.backend_ready.is_set():
                messagebox.showerror(
                    "Ошибка",
                    "Qdrant не готов. Сначала восстановите подключение.",
                    parent=viewer,
                )
                return

            if not viewer.delete_confirmed:
                viewer.delete_confirmed = True

                btn_del_photo.config(
                    text="⚠️ ПОДТВЕРДИТЕ",
                    bg="#B71C1C",
                )
                return

            current_file = (
                viewer.photos_list[
                    viewer.current_idx
                ]
            )
            btn_del_photo.config(state=tk.DISABLED, text="Удаление...")

            def _del_single():
                try:
                    self.db.delete_reference(
                        v_type,
                        part,
                        current_file,
                    )

                    def _update_ui():
                        if not viewer.winfo_exists():
                            self.load_gallery()
                            return
                        btn_del_photo.config(state=tk.NORMAL)
                        if viewer.photos_list:
                            viewer.photos_list.pop(viewer.current_idx)
                        self._image_cache.clear()
                        render_photo()
                        self.load_gallery()

                    self.ui_queue.put(
                        _update_ui
                    )

                except Exception as e:
                    logging.exception(
                        "Ошибка удаления фото"
                    )

                    def _show_error(err=str(e)):
                        if viewer.winfo_exists():
                            btn_del_photo.config(state=tk.NORMAL, text="🗑️ Удалить фото")
                            messagebox.showerror("Ошибка", err, parent=viewer)
                    self.ui_queue.put(_show_error)

            threading.Thread(
                target=_del_single,
                daemon=True,
            ).start()

        tk.Button(
            btn_frame,
            text="← Назад",
            command=lambda: (
                setattr(
                    viewer,
                    "current_idx",
                    viewer.current_idx - 1,
                ),
                render_photo(),
            ),
            font=("Arial", 12),
            bg="#374151",
            fg="white",
            width=10,
        ).pack(side=tk.LEFT)

        btn_del_photo = tk.Button(
            btn_frame,
            text="🗑️ Удалить фото",
            command=delete_current_photo,
            font=("Arial", 11, "bold"),
            bg="#D32F2F",
            fg="white",
        )
        btn_del_photo.pack(
            side=tk.LEFT,
            expand=True,
            padx=10,
        )

        tk.Button(
            btn_frame,
            text="Вперед →",
            command=lambda: (
                setattr(
                    viewer,
                    "current_idx",
                    viewer.current_idx + 1,
                ),
                render_photo(),
            ),
            font=("Arial", 12),
            bg="#374151",
            fg="white",
            width=10,
        ).pack(side=tk.RIGHT)

        render_photo()

    # =================================================================
    # СПРАВОЧНИКИ
    # =================================================================

    def load_metadata_tab(self):
        self.lb_types.delete(
            0,
            tk.END,
        )

        types = load_list(
            "types.txt",
            ["metiz", "bigdetail"],
        )

        for t in types:
            self.lb_types.insert(
                tk.END,
                t,
            )

        self.cb_meta_type["values"] = types

        if types:
            self.cb_meta_type.set(
                types[0]
            )
            self.load_meta_parts()
        else:
            self.cb_meta_type.set("")
            self.lb_parts.delete(
                0,
                tk.END,
            )

    def load_meta_parts(self, event=None):
        self.lb_parts.delete(
            0,
            tk.END,
        )

        v_type = self.cb_meta_type.get()

        if not v_type:
            return

        parts = load_list(
            f"parts_{v_type}.txt",
            [f"Деталь_{v_type}"],
        )

        for p in parts:
            self.lb_parts.insert(
                tk.END,
                p,
            )

    def meta_add_type(self):
        new_type = self.ent_type.get().strip()

        if not new_type:
            return

        types = load_list(
            "types.txt",
            ["metiz", "bigdetail"],
        )

        if new_type not in types:
            types.append(new_type)

            if not save_list(
                "types.txt",
                types,
            ):
                messagebox.showerror(
                    "Ошибка",
                    "Не удалось сохранить тип.",
                )
                return

        self.ent_type.delete(
            0,
            tk.END,
        )

        self.load_metadata_tab()
        self.load_registration_lists()

    def meta_delete_type(self):
        selection = self.lb_types.curselection()

        if not selection:
            return

        type_to_del = self.lb_types.get(
            selection[0]
        )

        types = load_list(
            "types.txt",
            ["metiz", "bigdetail"],
        )

        if type_to_del in types:
            types.remove(
                type_to_del
            )

            save_list(
                "types.txt",
                types,
            )

        self.load_metadata_tab()
        self.load_registration_lists()

    def meta_add_part(self):
        new_part = self.ent_part.get().strip()
        v_type = self.cb_meta_type.get()

        if not new_part or not v_type:
            return

        parts = load_list(
            f"parts_{v_type}.txt",
            [f"Деталь_{v_type}"],
        )

        if new_part not in parts:
            parts.append(new_part)

            if not save_list(
                f"parts_{v_type}.txt",
                parts,
            ):
                messagebox.showerror(
                    "Ошибка",
                    "Не удалось сохранить деталь.",
                )
                return

        self.ent_part.delete(
            0,
            tk.END,
        )

        self.load_meta_parts()
        self.update_part_list()

    def meta_delete_part(self):
        selection = self.lb_parts.curselection()
        v_type = self.cb_meta_type.get()

        if not selection or not v_type:
            return

        part_to_del = self.lb_parts.get(
            selection[0]
        )

        parts = load_list(
            f"parts_{v_type}.txt",
            [f"Деталь_{v_type}"],
        )

        if part_to_del in parts:
            parts.remove(
                part_to_del
            )

            save_list(
                f"parts_{v_type}.txt",
                parts,
            )

        self.load_meta_parts()
        self.update_part_list()

    # =================================================================
    # ЗАКРЫТИЕ
    # =================================================================

    def on_closing(self):
        if self._closing:
            return
        self._closing = True
        logging.info("Остановка системных потоков...")
        self.is_running.clear()
        self.backend_ready.clear()
        self.camera_scan_running.clear()

        # Разблокируем inference_queue, если поток ждёт get().
        try:
            self.inference_queue.put_nowait(None)
        except queue.Full:
            pass

        # Если какой-либо C/C++ backend камеры/torch завис на системном вызове,
        # через небольшой интервал гарантированно завершаем процесс. Таймер daemon,
        # поэтому нормальному выходу он не мешает.
        def force_exit_fallback():
            time.sleep(1.5)
            os._exit(0)

        threading.Thread(target=force_exit_fallback, daemon=True).start()

        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


# =====================================================================
# START
# =====================================================================

if __name__ == "__main__":
    root = tk.Tk()

    app = IndustrialVisionApp(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.on_closing,
    )

    root.mainloop()