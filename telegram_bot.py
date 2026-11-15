import os
import time
import telebot
import pickle
import string
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from freetrans import SyncTranslator
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- НАСТРОЙКИ ---
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

if not TOKEN:
    print("ОШИБКА: Переменная окружения TELEGRAM_BOT_TOKEN не найдена!")
    exit(1)

bot = telebot.TeleBot(TOKEN)

# Загрузка ресурсов
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
sentiment_analyzer = SentimentIntensityAnalyzer()

# Загрузка моделей
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Функции
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(words)

def analyze_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)

# --- КОМАНДЫ БОТА ---

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот для защиты цифровых двойников. Отправь мне текст на русском, и я проверю его на мошенничество и проанализирую настроение.")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    input_sms = message.text
    
    msg = bot.reply_to(message, "⏳ Анализирую текст...")

    try:
        # Пауза 2 секунды, чтобы не превышать лимиты
        time.sleep(2) 
        
        # Перевод текста (исправленные отступы)
        with SyncTranslator() as translator:
            input_sms_translated = translator.translate(input_sms, dest='en').text
        
        transformed_sms = preprocess_text(input_sms_translated)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        sentiment = analyze_sentiment(input_sms_translated)
        blob = TextBlob(input_sms_translated)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity * 100

        if result == 0:
            pred_text = "🚨 *Вердикт:* Мошенничество (высокая вероятность фишинга)"
        else:
            pred_text = "✅ *Вердикт:* Не мошенничество (низкая вероятность фишинга)"
            
        if sentiment['compound'] >= 0.05:
            overall = "Позитивный"
        elif sentiment['compound'] <= -0.05:
            overall = "Негативный"
        else:
            overall = "Нейтральный"

        answer = (
            f"{pred_text}\n\n"
            f"📊 *Анализ настроений:*\n"
            f"Общий окрас: {overall}\n"
            f"Негатив: {sentiment['neg'] * 100:.1f}%\n"
            f"Нейтральность: {sentiment['neu'] * 100:.1f}%\n"
            f"Позитив: {sentiment['pos'] * 100:.1f}%\n\n"
            f"📈 *Детали:*\n"
            f"Полярность: {polarity:.2f}\n"
            f"Субъективность: {subjectivity:.1f}%"
        )
        
        bot.edit_message_text(chat_id=msg.chat.id, message_id=msg.message_id, text=answer, parse_mode='Markdown')

    except Exception as e:
        # Если Google заблокировал запрос, выводим понятный текст
        if "Server Error" in str(e) or "too many requests" in str(e):
            error_text = "⏳ Сервис перевода временно недоступен. Пожалуйста, подождите 30 минут и попробуйте снова."
        else:
            error_text = f"Произошла ошибка при анализе: {e}"
            
        bot.edit_message_text(chat_id=msg.chat.id, message_id=msg.message_id, text=error_text)

# Запуск бота
print("Бот запущен...")
bot.infinity_polling()
