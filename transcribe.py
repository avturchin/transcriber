import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import subprocess
import tempfile

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
        self.chunk_duration = 15 * 60  # 15 минут в секундах
        self.ffmpeg_available = False
        
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
        
        # Папка для временных файлов
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Проверяем наличие ffmpeg
        self.ffmpeg_available = self.check_ffmpeg()
        
        # Файл для отслеживания обработанных файлов
        self.processed_files_log = self.output_dir / "processed_files.json"
        self.processed_files = self.load_processed_files()
        
        # Файл с ошибками
        self.error_log = self.output_dir / "failed_files.json"
        self.failed_files = self.load_failed_files()
        
        # Кэш для образца голоса
        self.reference_audio_file = None
        self.reference_loaded = False

    def check_ffmpeg(self):
        """Проверяет наличие ffmpeg и ffprobe"""
        try:
            # Проверяем ffmpeg
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError("ffmpeg не работает")
            
            # Проверяем ffprobe
            result = subprocess.run(['ffprobe', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError("ffprobe не работает")
                
            logger.info("FFmpeg и FFprobe найдены и готовы к использованию")
            return True
            
        except FileNotFoundError:
            logger.warning("FFmpeg/FFprobe не найдены")
            logger.info("Для разделения длинных файлов на куски потребуется ffmpeg")
            logger.info("Будет использована альтернативная стратегия для длинных файлов")
            return False

    def get_audio_duration(self, file_path: Path) -> Optional[float]:
        """Получает точную длительность аудио файла с помощью ffprobe"""
        if self.ffmpeg_available:
            try:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 
                    'format=duration', '-of', 'csv=p=0', str(file_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    logger.info(f"Точная длительность файла {file_path.name}: {duration/60:.1f} минут")
                    return duration
            except Exception as e:
                logger.warning(f"Не удалось получить длительность {file_path}: {e}")
        
        # Fallback на оценку по размеру
        size_mb = file_path.stat().st_size / (1024 * 1024)
        estimated_duration = size_mb * 60  # Приблизительно 1MB = 1 минута
        logger.info(f"Оценочная длительность {file_path.name}: {estimated_duration/60:.1f} минут")
        return estimated_duration

    def split_audio_file(self, file_path: Path) -> List[Path]:
        """Разделяет аудио файл на куски по 15 минут"""
        if not self.ffmpeg_available:
            logger.warning("FFmpeg недоступен, не могу разделить файл на куски")
            return []
            
        try:
            duration = self.get_audio_duration(file_path)
            if not duration:
                logger.error(f"Не удалось определить длительность {file_path}")
                return []
            
            if duration <= self.chunk_duration:
                logger.info(f"Файл {file_path.name} короче 15 минут, разделение не требуется")
                return [file_path]
            
            logger.info(f"Разделяю файл {file_path.name} на куски по 15 минут...")
            
            # Создаем папку для кусков этого файла
            chunks_dir = self.temp_dir / f"{file_path.stem}_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            chunk_files = []
            chunk_number = 1
            start_time = 0
            
            while start_time < duration:
                # Определяем время окончания куска
                end_time = min(start_time + self.chunk_duration, duration)
                chunk_duration_actual = end_time - start_time
                
                # Имя файла куска
                chunk_file = chunks_dir / f"{file_path.stem}_part{chunk_number:02d}{file_path.suffix}"
                
                # Команда ffmpeg для извлечения куска
                cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-ss', str(start_time), '-t', str(chunk_duration_actual),
                    '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                    str(chunk_file), '-y'
                ]
                
                logger.info(f"Создаю кусок {chunk_number}: {start_time/60:.1f}-{end_time/60:.1f} мин")
                logger.debug(f"FFmpeg команда: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    if chunk_file.exists() and chunk_file.stat().st_size > 0:
                        chunk_size = chunk_file.stat().st_size / 1024 / 1024  # MB
                        logger.info(f"Кусок {chunk_number} создан: {chunk_file.name} ({chunk_size:.2f} MB)")
                        chunk_files.append(chunk_file)
                    else:
                        logger.warning(f"Кусок {chunk_number} создан, но файл пуст или не существует")
                        logger.warning(f"FFmpeg stdout: {result.stdout}")
                        logger.warning(f"FFmpeg stderr: {result.stderr}")
                else:
                    logger.error(f"Ошибка создания куска {chunk_number} (код возврата: {result.returncode})")
                    logger.error(f"FFmpeg stdout: {result.stdout}")
                    logger.error(f"FFmpeg stderr: {result.stderr}")
                
                start_time = end_time
                chunk_number += 1
            
            if not chunk_files:
                logger.error(f"Не удалось создать ни одного куска из файла {file_path}")
                return []
            
            logger.info(f"Файл разделен на {len(chunk_files)} кусков")
            
            # Проверяем все созданные куски
            valid_chunks = []
            for chunk_file in chunk_files:
                if chunk_file.exists() and chunk_file.stat().st_size > 0:
                    valid_chunks.append(chunk_file)
                else:
                    logger.warning(f"Кусок {chunk_file} недействителен")
            
            logger.info(f"Валидных кусков: {len(valid_chunks)} из {len(chunk_files)}")
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Критическая ошибка разделения файла {file_path}: {e}")
            return []

    def transcribe_with_summary_approach(self, file_path: Path) -> Optional[str]:
        """Альтернативная стратегия: краткая транскрипция основных моментов"""
        try:
            logger.info(f"Использую summary-подход для длинного файла {file_path.name}")
            
            # Загрузка файла
            audio_file = genai.upload_file(path=str(file_path))
            
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise Exception("Не удалось обработать аудио файл")
            
            has_reference = self.find_and_load_reference_audio()
            
            # Промпт для краткой транскрипции
            summary_prompt = f"""
Пожалуйста, выполни структурированную транскрипцию основных моментов из длинного аудиофайла {file_path.name}.

Поскольку файл длинный и ffmpeg недоступен для разделения, сосредоточься на:
- Основных темах разговора
- Ключевых высказываниях и решениях  
- Важных фактах и информации
- Эмоциональных моментах и интересных деталях

Идентифицируй говорящих:
- Алексей (друг Лизы)
- Лиза  
- Мура (дочь Лизы)

Требования к формату:
- Используй таймкоды: [ММ:СС] (Имя): краткое изложение основной мысли
- Каждые 10 минут делай summary основных тем
- Не пропускай важные детали и интересные факты
- Обнаружение зацикливания: если фрагмент повторяется более 3 раз, отметь это

Структура транскрипции:
1. Начало разговора - основная тема
2. Основные моменты по времени с таймкодами
3. Ключевые выводы или решения
4. Интересные факты и детали
5. Завершение

В начале укажи: "КРАТКАЯ ТРАНСКРИПЦИЯ - основные моменты длинного файла (ffmpeg недоступен)"

В конце сделай выжимку: какие самые интересные факты мы узнали из разговора.

Затем добавь: "(Примечание: сокращенная версия, полная транскрипция требует установки ffmpeg для разделения файла на куски)"
"""

            content = [summary_prompt, audio_file]
            if has_reference and self.reference_audio_file:
                content.append(self.reference_audio_file)
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=6000,
                response_mime_type="text/plain",
            )
            
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
            
            genai.delete_file(audio_file.name)
            
            if response.candidates and response.candidates[0].content:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                
                if text_parts:
                    result = '\n'.join(text_parts)
                    logger.info(f"Summary транскрипция завершена: {len(result)} символов")
                    return result
            
            raise Exception("Нет результата в summary транскрипции")
                
        except Exception as e:
            logger.error(f"Ошибка summary транскрипции {file_path}: {e}")
            return None

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

    def create_chunk_prompt(self, filename: str, chunk_number: int, total_chunks: int, 
                           start_time_minutes: int, has_reference: bool) -> str:
        """Создает промпт для транскрипции куска"""
        prompt = f"""
Пожалуйста, выполни дословную транскрипцию речи на русском из куска {chunk_number} из {total_chunks} аудиофайла "{filename}".

ВАЖНО: Время начала куска: {start_time_minutes} минут от начала всего файла.

Основные требования:
- Распознай и запиши в текстовом виде абсолютно все произнесенные слова, не пропуская, не суммируя и не обобщая содержание
- Запиши точно все, что было сказано, и только это
- Если кусок начинается с середины фразы, отметь это как "...продолжение фразы с предыдущего куска..."
- Добавь таймкоды к каждой реплике (ОТНОСИТЕЛЬНО НАЧАЛА ВСЕГО ФАЙЛА, не куска!)
- Если никто ничего не говорит, просто пропускай этот таймкод
- Продолжай до конца куска, не повторяйся

Идентифицируй говорящих и указывай их имена в транскрипте:
- Алексей (друг Лизы) 
- Лиза
- Мура (дочь Лизы)

Обнаружение зацикливания:
1. Анализируй текст на предмет дословных повторов
2. Если последовательность из более чем 15 слов повторяется более 3 раз подряд, считай это техническим сбоем
3. Транскрибируй повторяющийся фрагмент только один раз
4. Вместо повторов вставь: [Обнаружен и пропущен многократно повторяющийся фрагмент]

Пример формата для этого куска (таймкоды от начала ВСЕГО файла):
[{start_time_minutes:02d}:15] (Алексей): ...продолжение фразы с предыдущего куска...
[{start_time_minutes:02d}:20] (Лиза): Да, я согласна с тобой по этому вопросу.

В конце куска напиши:
(конец куска {chunk_number}/{total_chunks})

Если разговор прерывается на середине фразы, отметь:
(фраза продолжается в следующем куске)"""

        if has_reference:
            prompt += "\n\nИспользуй прикрепленный образец голоса Алексея Турчина для правильного проставления его имени."
        
        return prompt

    def transcribe_chunk(self, chunk_path: Path, chunk_number: int, total_chunks: int, 
                        start_time_minutes: int, original_filename: str) -> Optional[str]:
        """Транскрибирует один кусок файла"""
        try:
            logger.info(f"Транскрибирую кусок {chunk_number}/{total_chunks}: {chunk_path.name}")
            
            # Проверяем существование и размер файла куска
            if not chunk_path.exists():
                raise Exception(f"Файл куска {chunk_number} не существует: {chunk_path}")
            
            chunk_size = chunk_path.stat().st_size
            logger.info(f"Размер куска {chunk_number}: {chunk_size / 1024 / 1024:.2f} MB")
            
            if chunk_size == 0:
                raise Exception(f"Файл куска {chunk_number} пустой")
            
            # Загрузка куска
            logger.info(f"Загружаем кусок {chunk_number} в Gemini...")
            audio_file = genai.upload_file(path=str(chunk_path))
            logger.info(f"Файл куска {chunk_number} загружен с ID: {audio_file.name}")
            
            # Ожидание обработки файла с таймаутом
            processing_time = 0
            max_processing_time = 300  # 5 минут максимум
            
            while audio_file.state.name == "PROCESSING":
                logger.info(f"Обработка куска {chunk_number} в процессе... ({processing_time}с)")
                time.sleep(5)
                processing_time += 5
                
                if processing_time > max_processing_time:
                    raise Exception(f"Превышено время ожидания обработки куска {chunk_number}")
                
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise Exception(f"Gemini не смог обработать кусок {chunk_number}. Возможно, файл поврежден или в неподдерживаемом формате")
            
            logger.info(f"Кусок {chunk_number} успешно загружен в Gemini, статус: {audio_file.state.name}")
            
            # Получение образца голоса
            has_reference = self.find_and_load_reference_audio()
            
            # Создание промпта для куска
            prompt = self.create_chunk_prompt(
                original_filename, chunk_number, total_chunks, 
                start_time_minutes, has_reference
            )
            
            # Подготовка содержимого для запроса
            content = [prompt, audio_file]
            if has_reference and self.reference_audio_file:
                content.append(self.reference_audio_file)
                logger.info(f"Используется образец голоса для куска {chunk_number}")
            
            # Настройки генерации
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=6000,
                response_mime_type="text/plain",
            )
            
            logger.info(f"Отправляем запрос на транскрипцию куска {chunk_number}...")
            
            # Отправка запроса на транскрипцию с повторными попытками
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
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
                    break
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Попытка {attempt + 1} для куска {chunk_number} не удалась: {e}. Повторяем через 10 секунд...")
                        time.sleep(10)
                    else:
                        raise Exception(f"Все попытки транскрипции куска {chunk_number} исчерпаны. Последняя ошибка: {e}")
            
            # Удаление временного файла куска из Gemini
            genai.delete_file(audio_file.name)
            logger.info(f"Файл куска {chunk_number} удален из Gemini")
            
            # Детальная обработка ответа
            if not response.candidates:
                raise Exception(f"Нет кандидатов в ответе для куска {chunk_number}")
            
            if len(response.candidates) == 0:
                raise Exception(f"Пустой список кандидатов для куска {chunk_number}")
            
            candidate = response.candidates[0]
            
            # Проверяем причину блокировки
            if hasattr(candidate, 'finish_reason'):
                logger.info(f"Причина завершения для куска {chunk_number}: {candidate.finish_reason}")
                if candidate.finish_reason.name in ['SAFETY', 'RECITATION']:
                    raise Exception(f"Ответ заблокирован системой безопасности для куска {chunk_number}: {candidate.finish_reason}")
            
            if not candidate.content:
                raise Exception(f"Нет содержимого в кандидате для куска {chunk_number}")
            
            if not candidate.content.parts:
                raise Exception(f"Нет частей в содержимом для куска {chunk_number}")
            
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            if not text_parts:
                raise Exception(f"Нет текстовых частей в ответе для куска {chunk_number}")
            
            full_text = '\n'.join(text_parts)
            logger.info(f"Кусок {chunk_number} успешно транскрибирован: {len(full_text)} символов")
            return full_text.strip()
                
        except Exception as e:
            logger.error(f"ДЕТАЛЬНАЯ ОШИБКА транскрипции куска {chunk_number}: {e}")
            # Сохраняем детали ошибки для отладки
            error_details = {
                'chunk_number': chunk_number,
                'chunk_path': str(chunk_path),
                'error': str(e),
                'chunk_exists': chunk_path.exists() if chunk_path else False,
                'chunk_size': chunk_path.stat().st_size if chunk_path and chunk_path.exists() else 0
            }
            self.failed_files[f"chunk_{chunk_number}_{original_filename}"] = error_details
            return None

    def transcribe_audio(self, file_path: Path) -> Optional[str]:
        """Транскрибирует аудио файл, выбирая оптимальную стратегию"""
        try:
            logger.info(f"Начинаю обработку файла: {file_path}")
            
            # Получаем длительность файла
            duration = self.get_audio_duration(file_path)
            if not duration:
                logger.error(f"Не удалось определить длительность файла {file_path}")
                return None
            
            duration_minutes = duration / 60
            
            # Стратегия обработки в зависимости от длительности и наличия ffmpeg
            if duration <= self.chunk_duration:
                logger.info(f"Файл короткий ({duration_minutes:.1f} мин), стандартная обработка")
                return self.transcribe_single_file(file_path)
            
            elif self.ffmpeg_available:
                logger.info(f"Файл длинный ({duration_minutes:.1f} мин), разделяем на куски")
                return self.transcribe_with_chunks(file_path)
            
            else:
                logger.info(f"Файл длинный ({duration_minutes:.1f} мин), но ffmpeg недоступен")
                logger.info("Использую summary-подход для извлечения основных моментов")
                return self.transcribe_with_summary_approach(file_path)
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            return None

    def transcribe_with_chunks(self, file_path: Path) -> Optional[str]:
        """Транскрибирует файл, разделяя на куски"""
        try:
            # Разделяем файл на куски
            chunk_files = self.split_audio_file(file_path)
            
            if not chunk_files:
                logger.error(f"Не удалось разделить файл {file_path}")
                # Fallback на summary подход
                logger.info("Переключаемся на summary-подход")
                return self.transcribe_with_summary_approach(file_path)
            
            # Транскрибируем каждый кусок
            all_transcriptions = []
            successful_chunks = 0
            
            for i, chunk_path in enumerate(chunk_files, 1):
                start_time_minutes = (i - 1) * 15  # Время начала куска в минутах
                
                transcription = self.transcribe_chunk(
                    chunk_path, i, len(chunk_files), 
                    start_time_minutes, file_path.name
                )
                
                if transcription:
                    all_transcriptions.append(transcription)
                    successful_chunks += 1
                    logger.info(f"Кусок {i}/{len(chunk_files)} успешно обработан")
                else:
                    logger.warning(f"Кусок {i}/{len(chunk_files)} не удалось транскрибировать")
                    all_transcriptions.append(f"\n[ОШИБКА: Кусок {i} не удалось транскрибировать]\n")
                
                # Задержка между кусками
                if i < len(chunk_files):
                    logger.info(f"Пауза {self.delay} секунд перед следующим куском...")
                    time.sleep(self.delay)
            
            # Очистка временных файлов
            self.cleanup_chunks(chunk_files)
            
            if successful_chunks == 0:
                logger.error(f"Ни один кусок файла {file_path} не удалось транскрибировать")
                return None
            
            # Объединяем все транскрипции
            final_transcription = self.combine_chunk_transcriptions(
                all_transcriptions, file_path.name, len(chunk_files), successful_chunks
            )
            
            logger.info(f"Файл {file_path.name} полностью обработан: {successful_chunks}/{len(chunk_files)} кусков")
            return final_transcription
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла с кусками {file_path}: {e}")
            return None

    def transcribe_single_file(self, file_path: Path) -> Optional[str]:
        """Транскрибирует одиночный файл без разделения"""
        try:
            # Загрузка файла
            audio_file = genai.upload_file(path=str(file_path))
            
            # Ожидание обработки файла
            while audio_file.state.name == "PROCESSING":
                logger.info("Обработка файла в процессе...")
                time.sleep(3)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise Exception("Не удалось обработать аудио файл")
            
            logger.info("Файл успешно загружен в Gemini")
            
            # Получение образца голоса
            has_reference = self.find_and_load_reference_audio()
            
            # Полный детальный промпт
            prompt = f"""
Пожалуйста, выполни дословную транскрипцию речи на русском из аудиофайла {file_path.name}, следуя приведенным ниже инструкциям:

Вначале пройди по тексту и идентифицируй имена говорящих и затем указывай говорящих в транскрипте, вставляя их имена, которые становятся понятны из контекста разговора.

Основные требования:
- Распознай и запиши в текстовом виде абсолютно все произнесенные слова, не пропуская, не суммируя и не обобщая содержание
- Запиши точно все, что было сказано, и только это
- Каждые 10 минут вставляй summary
- Выполни полную транскрипцию аудиофайла от начала до конца, не пропуская никаких фрагментов
- Добавь таймкоды к каждой реплике (каждые 26 токенов - это одна секунда)
- Продолжай до конца, не повторяйся
- Если никто ничего не говорит, просто пропускай этот таймкод
- Если ты дошёл до конца файла в распознавании, то напиши в конце: (конец файла)

Обнаружение зацикливания:
1. В процессе транскрипции постоянно анализируй текст на предмет дословных повторов
2. Если последовательность из более чем 15 слов повторяется более 3 раз подряд, считай это техническим сбоем
3. Транскрибируй повторяющийся фрагмент только один раз
4. Вместо повторов вставь: [Обнаружен и пропущен многократно повторяющийся фрагмент с {{таймкод начала}} по {{таймкод конца}}]
5. Не считай зацикливанием короткие повторы слов ("нет, нет, нет")

Наиболее вероятные имена участников беседы:
- Алексей (друг Лизы)
- Лиза
- Мура (дочь Лизы)

Пример желаемого формата:
[00:00:05] (Алексей): Привет, как дела?
[00:00:08] (Лиза): Привет! У меня все хорошо, спасибо. А у тебя?
[00:00:12] (Алексей): Да вот, решил тебе позвонить и узнать, как ты справляешься с новым проектом.

Когда файл закончится, сделай выжимку текста в конце - какие самые интересные факты мы узнали."""

            if has_reference:
                prompt += "\n\nИспользуй прикрепленный образец голоса Алексея Турчина для правильного проставления его имени. Образец в коротком прикреплённом файле."

            # Подготовка содержимого для запроса
            content = [prompt, audio_file]
            if has_reference and self.reference_audio_file:
                content.append(self.reference_audio_file)
                logger.info("Используется образец голоса для улучшения распознавания")
            
            # Настройки генерации
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=8000,
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
            
            # Удаление временного файла
            genai.delete_file(audio_file.name)
            
            # Обработка ответа
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if candidate.content and candidate.content.parts:
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    if text_parts:
                        full_text = '\n'.join(text_parts)
                        logger.info(f"Файл успешно транскрибирован: {len(full_text)} символов")
                        return full_text.strip()
                
                raise Exception("Нет текстового содержимого в ответе")
            else:
                raise Exception("Нет кандидатов в ответе")
                
        except Exception as e:
            logger.error(f"Ошибка транскрипции файла {file_path}: {e}")
            return None

    def combine_chunk_transcriptions(self, transcriptions: List[str], filename: str, 
                                   total_chunks: int, successful_chunks: int) -> str:
        """Объединяет транскрипции кусков в один файл"""
        header = f"""ПОЛНАЯ ТРАНСКРИПЦИЯ ФАЙЛА: {filename}
Метод: Разделение на куски по 15 минут
Всего кусков: {total_chunks}
Успешно обработано: {successful_chunks}
Неудачных попыток: {total_chunks - successful_chunks}
Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}

{"="*60}

"""
        
        combined = header
        for i, transcription in enumerate(transcriptions, 1):
            combined += f"\n--- КУСОК {i} из {total_chunks} ---\n\n"
            
            # Проверяем, является ли это ошибкой
            if transcription.startswith("[ОШИБКА:"):
                combined += transcription
                combined += f"\n\nПодробности ошибки смотри в логах и failed_files.json\n"
            else:
                combined += transcription
            
            combined += f"\n\n{'='*40}\n"
        
        # Добавляем статистику в конец
        if successful_chunks < total_chunks:
            combined += f"\n\nВНИМАНИЕ: {total_chunks - successful_chunks} кусков не удалось обработать.\n"
            combined += "Проверьте логи для выяснения причин ошибок.\n"
            combined += "Возможные причины:\n"
            combined += "- Повреждение аудиофайла\n"
            combined += "- Превышение лимитов API\n"
            combined += "- Сетевые проблемы\n"
            combined += "- Ошибки при разделении файла\n\n"
        
        combined += f"(КОНЕЦ ПОЛНОЙ ТРАНСКРИПЦИИ ФАЙЛА {filename})"
        
        return combined

    def cleanup_chunks(self, chunk_files: List[Path]):
        """Удаляет временные файлы кусков"""
        try:
            for chunk_file in chunk_files:
                if chunk_file.exists():
                    chunk_file.unlink()
                    logger.debug(f"Удален временный файл: {chunk_file}")
            
            # Удаляем пустые папки кусков
            for chunk_file in chunk_files:
                chunk_dir = chunk_file.parent
                if chunk_dir.exists() and not any(chunk_dir.iterdir()):
                    chunk_dir.rmdir()
                    logger.debug(f"Удалена пустая папка: {chunk_dir}")
                    
        except Exception as e:
            logger.warning(f"Ошибка очистки временных файлов: {e}")

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
                f.write(f"FFmpeg доступен: {'Да' if self.ffmpeg_available else 'Нет'}\n")
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
                f.write(f"FFmpeg доступен: {'Да' if self.ffmpeg_available else 'Нет'}\n")
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
        
        # Очистка временной папки
        try:
            if self.temp_dir.exists():
                for item in self.temp_dir.rglob("*"):
                    if item.is_file():
                        item.unlink()
                for item in self.temp_dir.rglob("*"):
                    if item.is_dir():
                        item.rmdir()
                logger.info("Временные файлы очищены")
        except Exception as e:
            logger.warning(f"Ошибка очистки временных файлов: {e}")

    def run(self):
        """Основной метод выполнения транскрипции"""
        logger.info("Запуск процесса транскрипции с адаптивной обработкой")
        logger.info(f"FFmpeg доступен: {'Да' if self.ffmpeg_available else 'Нет'}")
        
        if not self.ffmpeg_available:
            logger.info("Длинные файлы будут обработаны с помощью summary-подхода")
        
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
                    
                    # Задержка между файлами
                    if i < len(audio_files):
                        logger.info(f"Ожидание {self.delay * 2} секунд перед следующим файлом...")
                        time.sleep(self.delay * 2)
                    
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
    delay = float(os.getenv('API_DELAY', '3.0'))
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
