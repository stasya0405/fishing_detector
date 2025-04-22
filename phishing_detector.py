import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import os
from Levenshtein import distance as lev_distance
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nltk
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel
import shutil
import json
from datetime import datetime

nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')


class PhishingDetector:
    def __init__(self, model_path='phishing_model.pkl', vectorizer_path='tfidf_vectorizer.pkl',
                 history_path='history.json', auto_retrain=False):
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.bert_model = None
        self.legit_domains = ['google.com', 'paypal.com', 'microsoft.com', 'amazon.com', 'yandex.ru', 'mail.ru']
        self.suspicious_keywords = ['срочно', 'подтвердите', 'ваш аккаунт заблокирован', 'пароль', 'кредитная карта',
                                    'urgent', 'verify now']
        self.suspicious_tlds = ['.xyz', '.top', '.club', '.online']
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.history_path = history_path
        self.history = self.load_history()
        self.corrections = []
        self.auto_retrain = auto_retrain

        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Очищен кэш transformers: {cache_dir}")

        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
            self.bert_model.eval()
            print("Модель BERT успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели BERT: {str(e)}")
            print("Продолжаем без BERT-анализа.")
            self.bert_model = None
            self.tokenizer = None

        self.load_model()

    def load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add_to_history(self, filename, result, probability, suspicion_level, suspicion_category, explanation,
                       corrected=False, corrected_label=None):
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename,
            'result': result,
            'probability': float(probability),
            'suspicion_level': float(suspicion_level),
            'suspicion_category': suspicion_category,
            'explanation': explanation,
            'corrected': corrected,
            'corrected_label': corrected_label
        }
        self.history.append(entry)
        self.save_history()

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Модель и векторизатор загружены.")
        else:
            print("Модель не найдена. Необходимо обучить модель.")

    def _get_bert_embedding(self, text):
        if self.bert_model is None or self.tokenizer is None:
            return np.zeros(768)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def train_model(self, data_path='emails.csv'):
        data = pd.read_csv(data_path)
        data = data.rename(columns={'Email Text': 'text', 'Email Type': 'label'})
        data['label'] = data['label'].map({'Safe Email': 0, 'Phishing Email': 1})
        data = data.dropna(subset=['text', 'label'])

        if self.corrections:
            correction_texts = [corr['text'] for corr in self.corrections]
            correction_labels = [corr['corrected_label'] for corr in self.corrections]
            correction_df = pd.DataFrame({'text': correction_texts, 'label': correction_labels})
            data = pd.concat([data, correction_df], ignore_index=True)

        X_texts = data['text']
        y = data['label']

        self.vectorizer = TfidfVectorizer(max_features=500, stop_words=russian_stop_words)
        X_tfidf = self.vectorizer.fit_transform(X_texts).toarray()
        print(f"Размер X_tfidf: {X_tfidf.shape}")

        X_bert = np.array([self._get_bert_embedding(text) for text in X_texts])
        print(f"Размер X_bert: {X_bert.shape}")

        X_links = data['text'].apply(self._count_links).values.reshape(-1, 1)
        X_suspicious = data['text'].apply(
            lambda x: 1 if self._check_suspicious_links(re.findall(r'http[s]?://[^\s]+', x)) else 0).values.reshape(-1,
                                                                                                                    1)
        X_keywords = data['text'].apply(self._check_keywords).values.reshape(-1, 1)

        X = np.hstack((X_tfidf, X_bert, X_links, X_suspicious, X_keywords))
        print(f"Общий размер X: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Модель обучена и сохранена.")

    def _count_links(self, text):
        return len(re.findall(r'http[s]?://[^\s]+', text))

    def _check_suspicious_links(self, links):
        for link in links:
            domain = re.sub(r'http[s]?://([^/]+).*', r'\1', link)
            if domain.startswith('192.'):
                return True
            for legit in self.legit_domains:
                dist = lev_distance(domain, legit)
                if 0 < dist <= 2:
                    return True
        return False

    def _check_suspicious_tlds(self, links):
        for link in links:
            domain = re.sub(r'http[s]?://([^/]+).*', r'\1', link)
            if any(tld in domain for tld in self.suspicious_tlds):
                return True
        return False

    def _check_keywords(self, text):
        text_lower = text.lower()
        return sum(1 for keyword in self.suspicious_keywords if keyword in text_lower)

    def _has_legit_domain(self, links):
        for link in links:
            domain = re.sub(r'http[s]?://([^/]+).*', r'\1', link)
            if domain in self.legit_domains:
                return True
        return False

    def extract_features(self, text):
        tfidf = self.vectorizer.transform([text]).toarray()[0]
        bert_embedding = self._get_bert_embedding(text)
        links = re.findall(r'http[s]?://[^\s]+', text)
        link_count = len(links)
        suspicious_links = 1 if self._check_suspicious_links(links) else 0
        keyword_count = self._check_keywords(text)
        features = np.hstack((tfidf, bert_embedding, [link_count, suspicious_links, keyword_count]))
        print(f"Размер признаков при предсказании: {features.shape}")
        return features, links

    def _calculate_suspicion_factors(self, text, links, probability):
        keyword_count = self._check_keywords(text)
        has_legit_domain = self._has_legit_domain(links)

        factors = {
            "Ссылки": 0.5 if self._check_suspicious_links(links) else 0,
            "Домены": 0.3 if self._check_suspicious_tlds(links) else 0,
            "Ключ. слова": min(0.2, 0.05 * keyword_count),
            "Конф. данные": 0.15 if any(word in text.lower() for word in
                                        ['пароль', 'кредитная карта', 'данные']) and not has_legit_domain else 0,
            "Короткое": 0.05 if len(text) < 50 else 0,
            "Модель": 0.5 * probability
        }

        suspicion_level = sum(factors.values()) / (0.5 + 0.3 + 0.2 + 0.15 + 0.05 + 0.5)
        suspicion_level = min(1.0, max(0.0, suspicion_level))

        if suspicion_level < 0.3:
            suspicion_category = "Низкий"
        elif suspicion_level < 0.6:
            suspicion_category = "Средний"
        else:
            suspicion_category = "Высокий"

        return factors, suspicion_level, suspicion_category

    def analyze_email(self, file_path):
        if not self.model or not self.vectorizer:
            raise ValueError("Модель не загружена. Сначала обучите или загрузите модель.")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        features, links = self.extract_features(text)
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0][1]
        suspicion_factors, suspicion_level, suspicion_category = self._calculate_suspicion_factors(text, links,
                                                                                                   probability)
        final_result = "Фишинг" if (prediction == 1 or suspicion_level > 0.5) else "Безопасно"
        explanation = self._generate_explanation(text, links, final_result, suspicion_factors)
        self.add_to_history(os.path.basename(file_path), final_result, probability, suspicion_level, suspicion_category,
                            explanation)
        return final_result, probability, suspicion_level, suspicion_category, explanation, suspicion_factors, text, features

    def _generate_explanation(self, text, links, final_result, suspicion_factors):
        explanation = []
        has_legit_domain = self._has_legit_domain(links)
        keyword_count = self._check_keywords(text)

        if final_result == "Фишинг":
            explanation.append("Письмо классифицировано как фишинг по следующим причинам:")
            if suspicion_factors["Ссылки"] > 0:
                explanation.append(
                    "- Обнаружены подозрительные ссылки (например, домены, похожие на легитимные, но с отличиями)")
            if suspicion_factors["Домены"] > 0:
                explanation.append("- Ссылки содержат подозрительные домены верхнего уровня (например, .xyz, .top)")
            if suspicion_factors["Ключ. слова"] > 0:
                explanation.append(
                    f"- Найдено {keyword_count} подозрительных ключевых слов (например, 'срочно', 'пароль')")
            if suspicion_factors["Конф. данные"] > 0:
                explanation.append(
                    "- В письме запрашиваются конфиденциальные данные (например, пароль или данные карты)")
            if suspicion_factors["Короткое"] > 0:
                explanation.append("- Письмо слишком короткое, что часто встречается в фишинговых сообщениях")
            if suspicion_factors["Модель"] > 0.25:
                explanation.append("- Высокая вероятность фишинга по оценке модели машинного обучения")
            if not explanation[1:]:
                explanation.append("- Классификация основана на общей оценке модели")
        else:
            explanation.append("Письмо классифицировано как безопасное по следующим причинам:")
            if has_legit_domain:
                explanation.append("- Письмо содержит ссылки на легитимные домены (например, mail.ru, google.com)")
            if keyword_count == 0:
                explanation.append("- В письме отсутствуют подозрительные ключевые слова")
            else:
                explanation.append(
                    f"- Несмотря на наличие {keyword_count} потенциально подозрительных слов, другие признаки указывают на безопасность")
            if not self._check_suspicious_links(links):
                explanation.append("- Ссылки в письме не похожи на подозрительные")
            if suspicion_factors["Модель"] < 0.25:
                explanation.append("- Модель машинного обучения оценивает низкую вероятность фишинга")
            if not explanation[1:]:
                explanation.append("- Отсутствуют значимые признаки фишинга")

        return "\n".join(explanation)

    def analyze_folder(self, folder_path):
        results = []
        all_suspicion_factors = []
        filenames = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.txt', '.eml')):
                file_path = os.path.join(folder_path, filename)
                try:
                    result, prob, suspicion_level, suspicion_category, explanation, suspicion_factors, _, _ = self.analyze_email(
                        file_path)
                    results.append(
                        f"Файл: {filename}\nРезультат: {result}\nВероятность фишинга: {prob:.2f}\nУровень подозрительности: {suspicion_category} ({suspicion_level:.2f})\nОбъяснение:\n{explanation}\n{'-' * 50}")
                    all_suspicion_factors.append(suspicion_factors)
                    filenames.append(filename)
                except Exception as e:
                    results.append(f"Файл: {filename}\nОшибка: {str(e)}\n{'-' * 50}")
        return "\n".join(results), all_suspicion_factors, filenames

    def add_correction(self, text, corrected_label):
        self.corrections.append({
            'text': text,
            'corrected_label': 1 if corrected_label == "Фишинг" else 0
        })


class PhishingGUI:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("Детектор фишинга")
        self.root.geometry("800x600")
        self.root.configure(bg="#000000")
        self.root.resizable(True, True)

        self.label = tk.Label(
            self.root,
            text="Детектор фишинговых писем",
            font=("Helvetica", 16, "bold"),
            bg="#000000",
            fg="#FFFFFF"
        )
        self.label.pack(pady=10)

        try:
            from tkinter import ttk
            style = ttk.Style()
            style.configure("Custom.TButton", background="#000000", foreground="#FFFFFF", font=("Helvetica", 12))
            style.map("Custom.TButton",
                      background=[('active', '#333333')],
                      foreground=[('active', '#FFFFFF')])
            style.configure("Custom.TMenubutton", background="#1C2526", foreground="#FFFFFF", font=("Helvetica", 10))
            style.map("Custom.TMenubutton",
                      background=[('active', '#333333')],
                      foreground=[('active', '#FFFFFF')])

            self.train_button = ttk.Button(self.root, text="Обучить модель", command=self.train_model, width=20,
                                           style="Custom.TButton")
            self.train_button.pack(pady=5)

            self.scan_button = ttk.Button(self.root, text="Сканировать папку", command=self.scan_folder, width=20,
                                          style="Custom.TButton")
            self.scan_button.pack(pady=5)

            self.select_button = ttk.Button(self.root, text="Выбрать файл", command=self.select_file, width=20,
                                            style="Custom.TButton")
            self.select_button.pack(pady=5)

            self.history_button = ttk.Button(self.root, text="Показать историю", command=self.show_history, width=20,
                                             style="Custom.TButton")
            self.history_button.pack(pady=5)

            self.text_frame = tk.Frame(self.root, bg="#000000")
            self.text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

            self.result_text = scrolledtext.ScrolledText(self.text_frame, width=80, height=15, wrap=tk.WORD,
                                                         font=("Helvetica", 10),
                                                         bg="#1C2526", fg="#FFFFFF", bd=1, relief="solid",
                                                         insertbackground="#FFFFFF")
            self.result_text.pack(fill=tk.BOTH, expand=True)
            self.result_text.insert(tk.END, "Добро пожаловать!\nВыберите действие для анализа писем.\n")

            self.correct_frame = tk.Frame(self.root, bg="#000000")
            self.correct_frame.pack(pady=5)

            self.correct_label = tk.Label(self.correct_frame, text="Исправить результат:", bg="#000000", fg="#FFFFFF",
                                          font=("Helvetica", 10))
            self.correct_label.pack(side=tk.LEFT, padx=5)

            self.correct_var = tk.StringVar(value="Без изменений")
            self.correct_menu = ttk.OptionMenu(self.correct_frame, self.correct_var, "Без изменений", "Фишинг",
                                               "Безопасно", style="Custom.TMenubutton")
            self.correct_menu.pack(side=tk.LEFT, padx=5)

            self.correct_button = ttk.Button(self.correct_frame, text="Применить исправление",
                                             command=self.apply_correction,
                                             width=20, style="Custom.TButton")
            self.correct_button.pack(side=tk.LEFT, padx=5)

        except ImportError:
            self.train_button = tk.Button(self.root, text="Обучить модель", command=self.train_model, width=20,
                                          bg="#000000", fg="#FFFFFF",
                                          font=("Helvetica", 12), relief="flat", activebackground="#333333",
                                          highlightbackground="#000000", highlightthickness=0)
            self.train_button.pack(pady=5)

            self.scan_button = tk.Button(self.root, text="Сканировать папку", command=self.scan_folder, width=20,
                                         bg="#000000", fg="#FFFFFF",
                                         font=("Helvetica", 12), relief="flat", activebackground="#333333",
                                         highlightbackground="#000000", highlightthickness=0)
            self.scan_button.pack(pady=5)

            self.select_button = tk.Button(self.root, text="Выбрать файл", command=self.select_file, width=20,
                                           bg="#000000", fg="#FFFFFF",
                                           font=("Helvetica", 12), relief="flat", activebackground="#333333",
                                           highlightbackground="#000000", highlightthickness=0)
            self.select_button.pack(pady=5)

            self.history_button = tk.Button(self.root, text="Показать историю", command=self.show_history, width=20,
                                            bg="#000000", fg="#FFFFFF",
                                            font=("Helvetica", 12), relief="flat", activebackground="#333333",
                                            highlightbackground="#000000", highlightthickness=0)
            self.history_button.pack(pady=5)

            self.text_frame = tk.Frame(self.root, bg="#000000")
            self.text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

            self.result_text = scrolledtext.ScrolledText(self.text_frame, width=80, height=15, wrap=tk.WORD,
                                                         font=("Helvetica", 10),
                                                         bg="#1C2526", fg="#FFFFFF", bd=1, relief="solid",
                                                         insertbackground="#FFFFFF")
            self.result_text.pack(fill=tk.BOTH, expand=True)
            self.result_text.insert(tk.END, "Добро пожаловать!\nВыберите действие для анализа писем.\n")

            self.correct_frame = tk.Frame(self.root, bg="#000000")
            self.correct_frame.pack(pady=5)

            self.correct_label = tk.Label(self.correct_frame, text="Исправить результат:", bg="#000000", fg="#FFFFFF",
                                          font=("Helvetica", 10))
            self.correct_label.pack(side=tk.LEFT, padx=5)

            self.correct_var = tk.StringVar(value="Без изменений")
            self.correct_menu = tk.OptionMenu(self.correct_frame, self.correct_var, "Без изменений", "Фишинг",
                                              "Безопасно")
            self.correct_menu.config(bg="#1C2526", fg="#FFFFFF", activebackground="#333333", activeforeground="#FFFFFF",
                                     highlightbackground="#000000", highlightthickness=0)
            self.correct_menu["menu"].config(bg="#1C2526", fg="#FFFFFF", activebackground="#333333",
                                             activeforeground="#FFFFFF")
            self.correct_menu.pack(side=tk.LEFT, padx=5)

            self.correct_button = tk.Button(self.correct_frame, text="Применить исправление",
                                            command=self.apply_correction,
                                            width=20, bg="#000000", fg="#FFFFFF", font=("Helvetica", 12),
                                            relief="flat", activebackground="#333333", highlightbackground="#000000",
                                            highlightthickness=0)
            self.correct_button.pack(side=tk.LEFT, padx=5)

        self.plot_frame = tk.Frame(self.root, bg="#000000")
        self.plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.canvas = None
        self.last_analysis = None
        self.root.mainloop()

    def train_model(self):
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Обучение модели...\n")
            self.detector.train_model()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Модель успешно обучена!\nТеперь можно анализировать письма.\n")
            messagebox.showinfo("Успех", "Модель обучена")
            print("Обучение завершено")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка обучения: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка обучения: {str(e)}\n")

    def scan_folder(self):
        default_folder = os.path.join(os.getcwd(), "Emails")
        if os.path.exists(default_folder):
            folder_path = default_folder
        else:
            folder_path = filedialog.askdirectory(title="Выберите папку")
            if not folder_path:
                return
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Сканирование папки: {folder_path}\n\n")
            results, all_suspicion_factors, filenames = self.detector.analyze_folder(folder_path)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Анализ папки: {folder_path}\n\n{results}")
            print("Результат сканирования выведен")
            self._plot_suspicion_factors(all_suspicion_factors, filenames)
            self.last_analysis = None
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка сканирования: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка сканирования: {str(e)}\n")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("Email files", "*.eml")])
        if file_path:
            try:
                print(f"Выбран файл: {file_path}")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Анализ файла: {os.path.basename(file_path)}\n\n")
                result, prob, suspicion_level, suspicion_category, explanation, suspicion_factors, text, features = self.detector.analyze_email(
                    file_path)
                output = f"Файл: {os.path.basename(file_path)}\nРезультат: {result}\nВероятность фишинга: {prob:.2f}\nУровень подозрительности: {suspicion_category} ({suspicion_level:.2f})\nОбъяснение:\n{explanation}"
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, output)
                print("Результат анализа выведен")
                self._plot_suspicion_factors([suspicion_factors], [os.path.basename(file_path)])
                self.last_analysis = (
                file_path, result, prob, suspicion_level, suspicion_category, explanation, suspicion_factors, text,
                features)
                self.correct_var.set("Без изменений")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                print(f"Ошибка анализа: {str(e)}")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Ошибка анализа: {str(e)}\n")

    def apply_correction(self):
        if not self.last_analysis:
            messagebox.showwarning("Предупреждение", "Сначала проанализируйте файл!")
            return

        correction = self.correct_var.get()
        if correction == "Без изменений":
            return

        file_path, result, prob, suspicion_level, suspicion_category, explanation, suspicion_factors, text, features = self.last_analysis
        new_result = correction
        original_result = result

        if new_result != original_result:
            for entry in self.detector.history:
                if entry['filename'] == os.path.basename(file_path) and not entry['corrected']:
                    entry['corrected'] = True
                    entry['corrected_label'] = new_result
                    break

            self.detector.add_correction(text, new_result)
            self.detector.save_history()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END,
                                    f"Файл: {os.path.basename(file_path)}\nРезультат: {new_result}\nИсправление применено.\n")

            if self.detector.auto_retrain:
                self.result_text.insert(tk.END, "Дообучение модели...\n")
                self.detector.train_model()
                self.result_text.insert(tk.END, "Модель дообучена.\n")
            messagebox.showinfo("Успех",
                                f"Результат исправлен на: {new_result}\nМодель будет дообучена при следующем обучении.")

    def show_history(self):
        self.result_text.delete(1.0, tk.END)
        if not self.detector.history:
            self.result_text.insert(tk.END, "История проверок пуста.\n")
            return

        history_text = "История проверок:\n\n"
        for entry in self.detector.history:
            history_text += f"Время: {entry['timestamp']}\n"
            history_text += f"Файл: {entry['filename']}\n"
            history_text += f"Результат: {entry['result']}\n"
            history_text += f"Вероятность фишинга: {entry['probability']:.2f}\n"
            suspicion_category = entry.get('suspicion_category', "Не определено")
            history_text += f"Уровень подозрительности: {suspicion_category} ({entry['suspicion_level']:.2f})\n"
            history_text += f"Объяснение:\n{entry['explanation']}\n"
            if entry['corrected']:
                history_text += f"Исправлено на: {entry['corrected_label']}\n"
            history_text += "-" * 50 + "\n"

        self.result_text.insert(tk.END, history_text)

    def _plot_suspicion_factors(self, all_suspicion_factors, filenames):
        try:
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='#000000')

            if len(filenames) == 1:
                factors = all_suspicion_factors[0]
                labels = [k for k, v in factors.items() if v > 0]
                values = [v for v in factors.values() if v > 0]
                if not values:
                    values = [1]
                    labels = ["Нет факторов"]
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': '#FFFFFF'})
                ax.set_title(f"Факторы для {filenames[0]}", color='#FFFFFF')
            else:
                avg_factors = {k: np.mean([f[k] for f in all_suspicion_factors]) for k in all_suspicion_factors[0]}
                labels = list(avg_factors.keys())
                values = list(avg_factors.values())
                ax.bar(labels, values, color='skyblue')
                ax.set_title("Средний вклад факторов по папке", color='#FFFFFF')
                ax.set_ylabel("Средний вклад", color='#FFFFFF')
                plt.xticks(rotation=45, color='#FFFFFF')

            ax.set_facecolor('#1C2526')
            plt.tight_layout()
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            print("График отображён")
        except Exception as e:
            print(f"Ошибка в графике: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка в графике: {str(e)}\n")


if __name__ == "__main__":
    detector = PhishingDetector(auto_retrain=False)
    if not os.path.exists('emails.csv'):
        print("Скачайте данные с Kaggle и поместите их в 'emails.csv'")
    else:
        print("Обучение будет использовано на данных из 'emails.csv' с Kaggle")
    if not os.path.exists("Emails"):
        os.makedirs("Emails")
        with open("Emails/email1.txt", "w", encoding='utf-8') as f:
            f.write("Срочно! Нажмите http://fake.com для сброса пароля.")
        with open("Emails/email2.txt", "w", encoding='utf-8') as f:
            f.write("Ваш заказ: http://mail.ru/details")
    gui = PhishingGUI(detector)