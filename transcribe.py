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
                 output_dir: str = "transcriptions", delay: float = 2.0, 
                 use_reference_audio: bool = True):
        self.api_key = api_key
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.use_reference_audio = use_reference_audio
        self.max_file_size = 2 * 1024 * 1024 * 1024  # 2GB лимит для Gemini 2.5 Pro
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.aac'}
        
        # Настройка Gemini API - используем Gemini 2.5 Pro
        genai.configure(api_key=api_key)
        try:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Используется модель: gemini-2.5-pro")
        except Exception as e:
            logger.warning(f"Не удалось загрузить gemini-2.5-pro: {e}")
            logger.info("Переключаемся на gemini-2.0-flash-exp")
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Создание выходной директории
        self.output_dir.mkdir(exist_ok=True)
        
        # Файл для отслеживания обработанных файлов
        self.processed_files_log = self.output_dir / "processed_files.json"
        self.processed_files = self.load_processed_files()
        
        # Файл с ошибками
        self.error_log = self.output_dir / "failed_files.json"
        self.failed_files = self.load_failed_files()
        
        # Кэш для образца голоса
        self.reference_audio_file = None
        self.reference_loaded = False

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
        
        # Паттерны для образца голоса
        reference_patterns = ['alexey_sample', 'alex_sample', 'turchin_sample', 'reference_alexey']
        
        for file_path in self.source_dir.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_formats and
                str(file_path) not in self.processed_files and
                not file_path.stem.endswith('_transcribed')):
                
                # Пропускаем файлы-образцы голоса
                if any(pattern.lower() in file_path.stem.lower() for pattern in reference_patterns):
                    logger.info(f"Пропускаем файл-образец голоса: {file_path}")
                    continue
                
                # Проверка размера файла
                if file_path.stat().st_size > self.max_file_size:
                    logger.warning(f"Файл {file_path} слишком большой ({file_path.stat().st_size / (1024*1024*1024):.2f} GB)")
                    self.failed_files[str(file_path)] = "Файл превышает максимальный размер"
                    continue
                
                audio_files.append(file_path)
        
        logger.info(f"Найдено {len(audio_files)} аудио файлов для обработки")
        return audio_files

    def find_and_load_reference_audio(self) -> bool:
        """Ищет и загружает образец голоса Алексея Турчина один раз"""
        if self.reference_loaded or not self.use_reference_audio:
            return self.reference_audio_file is not None
        
        self.reference_loaded = True
        reference_patterns = ['alexey_sample', 'alex_sample', 'turchin_sample', 'reference_alexey']
        
        for pattern in reference_patterns:
            for file_path in self.source_dir.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.supported_formats and
                    pattern.lower() in file_path.stem.lower()):
                    
                    try:
                        logger.info(f"Найден образец голоса: {file_path}")
                        
                        # Проверка размера образца
                        if file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB для образца
                            logger.warning(f"Образец голоса слишком большой: {file_path}")
                            continue
                        
                        self.reference_audio_file = genai.upload_file(path=str(file_path))
                        
                        # Ожидание обработки файла
                        while self.reference_audio_file.state.name == "PROCESSING":
                            time.sleep(1)
                            self.reference_audio_file = genai.get_file(self.reference_audio_file.name)
                        
                        if self.reference_audio_file.state.name == "FAILED":
                            logger.warning("Не удалось обработать образец голоса")
                            self.reference_audio_file = None
                            continue
                        
                        logger.info("Образец голоса успешно загружен и будет использован для всех файлов")
                        return True
                        
                    except Exception as e:
                        logger.warning(f"Ошибка при загрузке образца голоса {file_path}: {e}")
                        self.reference_audio_file = None
                        continue
        
        if self.use_reference_audio:
            logger.info("Образец голоса не найден, транскрипция будет выполнена без него")
        else:
            logger.info("Использование образца голоса отключено")
        return False

    def create_transcription_prompt(self, filename: str, has_reference: bool) -> str:
        """Создает промпт для транскрипции"""
        base_prompt = f"""
Пожалуйста, выполни дословную транскрипцию речи на русском из этого аудиофайла ({filename}), следуя приведенным ниже инструкциям:

вначале пройди по тексту и Идентифицируй имена говорящих и затем указывай говорящих в транскрипте, вставляя их имена, которые становятся понятны из контекста разговора.

Распознай и запиши в текстовом виде абсолютно все произнесенные слова, не пропуская, не суммируя и не обобщая содержание. Запиши точно все, что было сказано, и только это.

Каждые 10 минут вставляй summary

Выполни полную транскрипцию аудиофайла от начала до конца, не пропуская никаких фрагментов.

Добавь таймкоды к каждой реплике, чтобы указать, в какой момент времени она была произнесена. каждые 26 токенов - это одна секунда

Если во время выполнения задачи тебе покажется, что нужно написать "(продолжение следует)", вместо этого продолжай распознавание речи до полного завершения аудиофайла.

Обеспечь исчерпывающую и точную транскрипцию всего текста из аудиофайла, не упуская никаких деталей.
продолжай до конца
не повторяйся
check names correctedness"""

        if has_reference:
            base_prompt += "\nИспользуй прикрепленный образец голоса алексея турчина для правильного проставления его имени."
        
        base_prompt += """
если ты дошёл до конца файла в распозновании, то напиши в конце: (конец файла)
если реплики повторяются по кругу больше 4 раз, то игнорируй дальше эти повторы и не вставляй в файл.

Пример желаемого формата:
[00:00:05] (Алексей): Привет, как дела?
[00:00:08] (лиза): Привет! У меня все хорошо, спасибо. А у тебя?
[00:00:12] (Алексей): Да вот, решил тебе позвонить и узнать, как ты справляешься с новым проектом.
[00:00:17] (лиза): Пока все идет по плану, но есть пара вопросов, которые хотела бы с тобой обсудить.

Наиболее вероятные имена участников беседы:
Алексей, друг лизы
Лиза однограница
муж Евгений лизы
Мура дочь лизы

Верни только транскрипцию в указанном формате, без дополнительных комментариев.
"""
        return base_prompt

    def transcribe_audio(self, file_path: Path) -> Optional[str]:
        """Транскрибирует один аудио файл"""
        try:
            logger.info(f"Транскрибирую файл: {file_path} (размер: {file_path.stat().st_size / (1024*1024):.2f} MB)")
            
            # Загрузка основного аудио файла
            audio_file = genai.upload_file(path=str(file_path))
            
            # Ожидание обработки файла
            while audio_file.state.name == "PROCESSING":
                logger.info("Обработка файла в процессе...")
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise Exception("Не удалось обработать аудио файл")
            
            logger.info("Файл успешно загружен в Gemini")
            
            # Попытка загрузить образец голоса (только один раз)
            has_reference = self.find_and_load_reference_audio()
            
            # Создание промпта
            prompt = self.create_transcription_prompt(file_path.name, has_reference)
            
            # Подготовка содержимого для запроса
            content = [prompt, audio_file]
            if has_reference and self.reference_audio_file:
                content.append(self.reference_audio_file)
                logger.info("Используется образец голоса для улучшения распознавания")
            
            # Настройки генерации для лучшей обработки длинных файлов
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,  # Низкая температура для точности
                top_p=0.8,
                top_k=40,
                max_output_tokens=8192,  # Максимальная длина ответа
                response_mime_type="text/plain",
            )
            
            logger.info("Отправляем запрос на транскрипцию...")
            
            # Отправка запроса на транскрипцию
            response = self.model.generate_content(
                content,
                generation_config=generation_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Удаление временного файла основного аудио
            genai.delete_file(audio_file.name)
            
            if response.text:
                logger.info(f"Успешно транскрибирован файл: {file_path}")
                logger.info(f"Длина транскрипции: {len(response.text)} символов")
                return response.text.strip()
            else:
                raise Exception("Пустой ответ от API")
                
        except Exception as e:
            logger.error(f"Ошибка транскрипции файла {file_path}: {e}")
            self.failed_files[str(file_path)] = str(e)
            # Попытка удалить файл при ошибке
            try:
                if 'audio_file' in locals():
                    genai.delete_file(audio_file.name)
            except:
                pass
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
                f.write(f"Модель: {self.model.model_name}\n")
                f.write(f"Дата создания: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                f.write(f"Модель: {self.model.model_name}\n")
                f.write(f"Дата создания: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    def cleanup(self):
        """Очистка ресурсов"""
        if self.reference_audio_file:
            try:
                genai.delete_file(self.reference_audio_file.name)
                logger.info("Образец голоса удален из Gemini")
            except Exception as e:
                logger.warning(f"Ошибка при удалении образца голоса: {e}")

    def run(self):
        """Основной метод выполнения транскрипции"""
        logger.info("Запуск процесса транскрипции")
        
        try:
            # Поиск аудио файлов
            audio_files = self.find_audio_files()
            if not audio_files:
                logger.info("Нет файлов для обработки")
                return
            
            transcriptions = {}
            
            for i, file_path in enumerate(audio_files, 1):
                try:
                    logger.info(f"Обработка файла {i}/{len(audio_files)}: {file_path.name}")
                    
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
                        
                        logger.info(f"Файл {file_path.name} успешно обработан")
                    
                    # Задержка между запросами (увеличена для Gemini 2.5 Pro)
                    if i < len(audio_files):  # Не ждать после последнего файла
                        logger.info(f"Ожидание {self.delay} секунд перед следующим файлом...")
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
            
        finally:
            # Очистка ресурсов
            self.cleanup()

def main():
    # Получение API ключа из переменной окружения
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("Не найден GEMINI_API_KEY в переменных окружения")
        return
    
    # Настройки из переменных окружения или значения по умолчанию
    source_dir = os.getenv('SOURCE_DIR', 'audio_files')
    output_dir = os.getenv('OUTPUT_DIR', 'transcriptions')
    delay = float(os.getenv('API_DELAY', '3.0'))  # Увеличена для Pro версии
    use_reference = os.getenv('USE_REFERENCE_AUDIO', 'true').lower() in ('true', '1', 'yes')
    
    # Создание и запуск транскрайбера
    transcriber = AudioTranscriber(
        api_key=api_key,
        source_dir=source_dir,
        output_dir=output_dir,
        delay=delay,
        use_reference_audio=use_reference
    )
    
    transcriber.run()

if __name__ == "__main__":
    main()
