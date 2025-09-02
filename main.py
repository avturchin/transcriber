import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, api_key: str, source_dir: str = "audio_files", 
                 output_dir: str = "transcriptions", delay: float = 2.0):
        self.api_key = api_key
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_file_size = 20 * 1024 * 1024  # 20MB лимит для Gemini
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}
        
        # Настройка Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Создание выходной директории
        self.output_dir.mkdir(exist_ok=True)
        
        # Файл для отслеживания обработанных файлов
        self.processed_files_log = self.output_dir / "processed_files.json"
        self.processed_files = self.load_processed_files()
        
        # Файл с ошибками
        self.error_log = self.output_dir / "failed_files.json"
        self.failed_files = self.load_failed_files()

    def load_processed_files(self) -> set:
        """Загружает список уже обработанных файлов"""
        if self.processed_files_log.exists():
            try:
                with open(self.processed_files_log, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Не удалось загрузить список обработанных файлов: {e}")
        return set()

    def save_processed_files(self):
        """Сохраняет список обработанных файлов"""
        try:
            with open(self.processed_files_log, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_files), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения списка обработанных файлов: {e}")

    def load_failed_files(self) -> dict:
        """Загружает список файлов с ошибками"""
        if self.error_log.exists():
            try:
                with open(self.error_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить список файлов с ошибками: {e}")
        return {}

    def save_failed_files(self):
        """Сохраняет список файлов с ошибками"""
        try:
            with open(self.error_log, 'w', encoding='utf-8') as f:
                json.dump(self.failed_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения списка файлов с ошибками: {e}")

    def find_audio_files(self) -> List[Path]:
        """Находит все аудио файлы в директории и поддиректориях"""
        audio_files = []
        if not self.source_dir.exists():
            logger.error(f"Директория {self.source_dir} не существует")
            return audio_files
        
        for file_path in self.source_dir.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_formats and
                str(file_path) not in self.processed_files and
                not file_path.stem.endswith('_transcribed')):
                
                # Проверка размера файла
                if file_path.stat().st_size > self.max_file_size:
                    logger.warning(f"Файл {file_path} слишком большой ({file_path.stat().st_size} байт)")
                    self.failed_files[str(file_path)] = "Файл превышает максимальный размер"
                    continue
                
                audio_files.append(file_path)
        
        logger.info(f"Найдено {len(audio_files)} аудио файлов для обработки")
        return audio_files

    def create_transcription_prompt(self, filename: str) -> str:
        """Создает промпт для транскрипции"""
        return f"""
Пожалуйста, транскрибируй этот аудио файл ({filename}) максимально точно. 
Требования:
- Сохрани все слова и фразы как есть
- Добавь знаки препинания для лучшей читаемости
- Если слышишь несколько говорящих, укажи это
- Если что-то неразборчиво, отметь как [неразборчиво]
- Если есть длинные паузы, отметь как [пауза]

Верни только текст транскрипции без дополнительных комментариев.
"""

    def transcribe_audio(self, file_path: Path) -> Optional[str]:
        """Транскрибирует один аудио файл"""
        try:
            logger.info(f"Транскрибирую файл: {file_path}")
            
            # Загрузка файла
            audio_file = genai.upload_file(path=str(file_path))
            
            # Ожидание обработки файла
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise Exception("Не удалось обработать аудио файл")
            
            # Создание промпта
            prompt = self.create_transcription_prompt(file_path.name)
            
            # Отправка запроса на транскрипцию
            response = self.model.generate_content(
                [prompt, audio_file],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Удаление временного файла из Gemini
            genai.delete_file(audio_file.name)
            
            if response.text:
                logger.info(f"Успешно транскрибирован файл: {file_path}")
                return response.text.strip()
            else:
                raise Exception("Пустой ответ от API")
                
        except Exception as e:
            logger.error(f"Ошибка транскрипции файла {file_path}: {e}")
            self.failed_files[str(file_path)] = str(e)
            return None

    def save_transcription(self, file_path: Path, transcription: str):
        """Сохраняет транскрипцию в отдельный файл"""
        # Создание структуры папок
        relative_path = file_path.relative_to(self.source_dir)
        output_file_dir = self.output_dir / relative_path.parent
        output_file_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание имени файла транскрипции
        output_file = output_file_dir / f"{file_path.stem}_transcription.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Транскрипция файла: {file_path.name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(transcription)
            logger.info(f"Транскрипция сохранена: {output_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения транскрипции {output_file}: {e}")

    def rename_processed_file(self, file_path: Path):
        """Переименовывает обработанный файл, добавляя _transcribed"""
        try:
            new_name = file_path.parent / f"{file_path.stem}_transcribed{file_path.suffix}"
            file_path.rename(new_name)
            logger.info(f"Файл переименован: {file_path} -> {new_name}")
        except Exception as e:
            logger.error(f"Ошибка переименования файла {file_path}: {e}")

    def create_combined_transcription(self, transcriptions: Dict[str, str]):
        """Создает общий файл со всеми транскрипциями"""
        combined_file = self.output_dir / "all_transcriptions.txt"
        
        try:
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write("СВОДНАЯ ТРАНСКРИПЦИЯ ВСЕХ АУДИО ФАЙЛОВ\n")
                f.write("=" * 60 + "\n\n")
                
                # Оглавление
                f.write("ОГЛАВЛЕНИЕ:\n")
                f.write("-" * 20 + "\n")
                for i, filename in enumerate(transcriptions.keys(), 1):
                    f.write(f"{i}. {filename}\n")
                f.write("\n" + "=" * 60 + "\n\n")
                
                # Транскрипции
                for i, (filename, transcription) in enumerate(transcriptions.items(), 1):
                    f.write(f"{i}. ФАЙЛ: {filename}\n")
                    f.write("-" * (len(filename) + 10) + "\n\n")
                    f.write(transcription)
                    f.write("\n\n" + "=" * 60 + "\n\n")
            
            logger.info(f"Создан сводный файл: {combined_file}")
        except Exception as e:
            logger.error(f"Ошибка создания сводного файла: {e}")

    def run(self):
        """Основной метод выполнения транскрипции"""
        logger.info("Запуск процесса транскрипции")
        
        # Поиск аудио файлов
        audio_files = self.find_audio_files()
        if not audio_files:
            logger.info("Нет файлов для обработки")
            return
        
        transcriptions = {}
        
        for file_path in audio_files:
            try:
                # Транскрипция
                transcription = self.transcribe_audio(file_path)
                
                if transcription:
                    # Сохранение отдельной транскрипции
                    self.save_transcription(file_path, transcription)
                    
                    # Добавление в общий список
                    transcriptions[file_path.name] = transcription
                    
                    # Переименование файла
                    self.rename_processed_file(file_path)
                    
                    # Отметка как обработанный
                    self.processed_files.add(str(file_path))
                    self.save_processed_files()
                
                # Задержка между запросами
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Критическая ошибка при обработке {file_path}: {e}")
                self.failed_files[str(file_path)] = str(e)
        
        # Создание сводного файла
        if transcriptions:
            self.create_combined_transcription(transcriptions)
        
        # Сохранение логов ошибок
        if self.failed_files:
            self.save_failed_files()
            logger.warning(f"Обработано с ошибками: {len(self.failed_files)} файлов")
        
        logger.info(f"Процесс завершен. Успешно обработано: {len(transcriptions)} файлов")

def main():
    # Получение API ключа из переменной окружения
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("Не найден GEMINI_API_KEY в переменных окружения")
        return
    
    # Настройки из переменных окружения или значения по умолчанию
    source_dir = os.getenv('SOURCE_DIR', 'audio_files')
    output_dir = os.getenv('OUTPUT_DIR', 'transcriptions')
    delay = float(os.getenv('API_DELAY', '2.0'))
    
    # Создание и запуск транскрайбера
    transcriber = AudioTranscriber(
        api_key=api_key,
        source_dir=source_dir,
        output_dir=output_dir,
        delay=delay
    )
    
    transcriber.run()

if __name__ == "__main__":
    main()
