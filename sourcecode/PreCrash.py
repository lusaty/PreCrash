import os

from imblearn.over_sampling import SMOTE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack
from gensim.models import Word2Vec, FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM
import warnings
import joblib

# 忽略Keras的警告
warnings.filterwarnings("ignore")


class PreCrash:
    def __init__(self, excel_path, log_dir='logs', model_dir='saved_models'):
        """
        初始化分类器，加载数据并进行预处理，同时配置日志记录。
        """
        self.excel_path = excel_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.test_texts = []
        self.test_labels = []
        self.top_keywords = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.models_ml = {}
        self.models_dl = {}
        self.vectorizer_tfidf = None
        self.vectorizer_count = None
        self.clf_ml = {}
        self.clf_dl = {}
        self.word2vec_model = None
        self.fasttext_model = None
        self.tokenizer_cnn_lstm = None
        self.max_sequence_length = 100  # 可根据需要调整
        self._setup_logging()

        # 创建模型保存目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self._load_data()
        self._prepare_word_embeddings()
        self._prepare_textcnn_lstm_tokenizer()

    def _setup_logging(self):
        """
        配置日志记录，将日志保存到指定的目录中。
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_file = os.path.join(self.log_dir, 'PreCrash.log')

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("日志记录已配置。")

    def _load_data(self):
        """
        加载并预处理数据，选择关键词。
        """
        self.logger.info("开始加载数据...")
        try:
            self.df = pd.read_excel(self.excel_path)
            self.logger.info("数据加载成功。")
        except Exception as e:
            self.logger.error(f"读取Excel文件失败: {e}")
            raise e

        # 删除缺失值
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=['text', 'label'])
        final_size = len(self.df)
        self.logger.info(f"删除缺失值，数据量从 {initial_size} 减少到 {final_size}。")

        # 确保标签为整数类型
        self.df['label'] = self.df['label'].astype(int)

        # 检查标签值是否为0和1
        if not set(self.df['label'].unique()).issubset({0, 1}):
            self.logger.error("标签列中存在非0和1的值。请检查数据。")
            raise ValueError("标签列中存在非0和1的值。")

        label_counts = self.df['label'].value_counts()
        self.logger.info(f"标签分布:\n{label_counts}")

        # 划分训练集和测试集
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=0.2, random_state=42, shuffle=True
        )
        self.test_texts = self.test_df['text'].tolist()
        self.test_labels = self.test_df['label'].tolist()
        self.logger.info(f"训练集大小: {len(self.train_df)}")
        self.logger.info(f"测试集大小: {len(self.test_df)}")

        # 选择训练集中标签为1的最频繁的10个词作为关键词
        train_repro_df = self.train_df[self.train_df['label'] == 1]
        if train_repro_df.empty:
            self.logger.warning("训练集中没有标签为1的样本，无法提取关键词。")
            self.top_keywords = []
        else:
            count_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
            X_counts = count_vectorizer.fit_transform(train_repro_df['text'])
            word_freq = np.array(X_counts.sum(axis=0)).flatten()
            vocab = count_vectorizer.get_feature_names_out()
            top_indices = word_freq.argsort()[::-1][:10]
            self.top_keywords = [vocab[i] for i in top_indices]
            self.logger.info(f"选取的关键词(top10): {self.top_keywords}")

    def _prepare_word_embeddings(self):
        """
        训练Word2Vec和FastText模型。
        """
        self.logger.info("开始训练Word2Vec和FastText模型...")
        texts = self.train_df['text'].tolist()
        tokenized_texts = [text.lower().split() for text in texts]

        # 训练Word2Vec
        self.word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        self.logger.info("Word2Vec模型训练完成。")

        # 训练FastText
        self.fasttext_model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        self.logger.info("FastText模型训练完成。")

    def _prepare_textcnn_lstm_tokenizer(self):
        """
        准备用于TextCNN和LSTM的Tokenizer。
        """
        self.logger.info("开始准备TextCNN和LSTM的Tokenizer...")
        self.max_num_words = 60000  # 设置最大词汇数量，根据需要调整
        self.tokenizer_cnn_lstm = Tokenizer(num_words=self.max_num_words, oov_token='<OOV>')
        self.tokenizer_cnn_lstm.fit_on_texts(self.train_df['text'].tolist())
        self.logger.info("Tokenizer准备完成。")

    def prepare_ml_features(self):
        """
        提取并准备机器学习的特征，包括TF-IDF和关键词计数。
        应用SMOTE对训练数据进行过采样以平衡类别。
        """
        # TF-IDF 特征
        self.vectorizer_tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
        X_train_tfidf = self.vectorizer_tfidf.fit_transform(self.train_df['text'])
        X_test_tfidf = self.vectorizer_tfidf.transform(self.test_df['text'])

        # 关键词特征
        if self.top_keywords:
            X_train_kw = np.array([
                sum(text.lower().count(kw) for kw in self.top_keywords)
                for text in self.train_df['text']
            ]).reshape(-1, 1)
            X_test_kw = np.array([
                sum(text.lower().count(kw) for kw in self.top_keywords)
                for text in self.test_df['text']
            ]).reshape(-1, 1)
        else:
            # 如果没有关键词，则使用0作为特征
            X_train_kw = np.zeros((self.train_df.shape[0], 1))
            X_test_kw = np.zeros((self.test_df.shape[0], 1))

        # 合并特征
        X_train_ml = hstack([X_train_tfidf, X_train_kw])
        X_test_ml = hstack([X_test_tfidf, X_test_kw])

        # 提取标签
        y_train = self.train_df['label'].values

        # 应用SMOTE对训练数据进行过采样
        self.logger.info("应用SMOTE对训练数据进行过采样以平衡类别...")
        smote = SMOTE(random_state=42)
        X_train_ml_resampled, y_train_resampled = smote.fit_resample(X_train_ml, y_train)
        self.logger.info(
            f"SMOTE应用完成。原始训练集大小: {X_train_ml.shape}, 过采样后训练集大小: {X_train_ml_resampled.shape}")

        return X_train_ml_resampled, y_train_resampled, X_test_ml

    def train_ml_models(self, X_train_ml, y_train):
        """
        训练多种传统机器学习模型。
        """
        self.logger.info("开始训练传统机器学习模型...")

        # Logistic Regression
        X_train_ml_resampled, y_train_resampled, _ = self.prepare_ml_features()
        clf_lr = LogisticRegression(max_iter=1000)
        clf_lr.fit(X_train_ml, y_train_resampled)
        self.models_ml['Logistic Regression'] = clf_lr
        self.logger.info("Logistic Regression 模型训练完成。")
        # 保存模型

        # Random Forest
        self.logger.info("开始训练 Random Forest 模型...")
        X_train_ml_resampled, y_train_resampled, _ = self.prepare_ml_features()
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(X_train_ml, y_train_resampled)
        self.models_ml['Random Forest'] = clf_rf
        self.logger.info("Random Forest 模型训练完成。")
        # 保存模型

        # SVM
        self.logger.info("开始训练 SVM 模型...")
        X_train_ml_resampled, y_train_resampled, _ = self.prepare_ml_features()
        clf_svm = SVC(probability=True, random_state=42)
        clf_svm.fit(X_train_ml_resampled, y_train_resampled)
        self.models_ml['SVM'] = clf_svm
        self.logger.info("SVM 模型训练完成。")
        # 保存模型

        # Decision Tree
        self.logger.info("开始训练 Decision Tree 模型...")
        X_train_ml_resampled, y_train_resampled, _ = self.prepare_ml_features()
        clf_dt = DecisionTreeClassifier(random_state=42)
        clf_dt.fit(X_train_ml_resampled, y_train_resampled)
        self.models_ml['Decision Tree'] = clf_dt
        self.logger.info("Decision Tree 模型训练完成。")
        # 保存模型

        # XGBoost
        self.logger.info("开始训练 XGBoost 模型...")
        X_train_ml_resampled, y_train_resampled, _ = self.prepare_ml_features()
        clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        clf_xgb.fit(X_train_ml_resampled, y_train_resampled)
        self.models_ml['XGBoost'] = clf_xgb
        self.logger.info("XGBoost训练完成。")

        # 保存训练好的传统机器学习模型
        self.save_ml_models()

    def prepare_bert_embeddings(self, model_name='bert-base-uncased'):
        """
        使用BERT模型提取文本的[CLS]嵌入作为深度学习特征。
        """
        self.logger.info(f"开始提取{model_name}嵌入...")
        tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
        model_bert = AutoModel.from_pretrained(model_name)
        model_bert.to(self.device)
        model_bert.eval()

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, max_len=256):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }

        def get_embeddings(dataloader, model):
            embeddings = []
            for batch in tqdm(dataloader, desc="提取嵌入"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
            embeddings = np.vstack(embeddings)
            return embeddings

        # 创建数据集与dataloader
        train_dataset = TextDataset(self.train_df['text'].tolist(), tokenizer_bert)
        test_dataset = TextDataset(self.test_df['text'].tolist(), tokenizer_bert)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 提取嵌入
        X_train_bert = get_embeddings(train_loader, model_bert)
        X_test_bert = get_embeddings(test_loader, model_bert)

        self.logger.info(f"{model_name}嵌入提取完成。")
        return X_train_bert, X_test_bert

    def train_bert_model(self, X_train_bert, y_train, model_name='albert-base-v2'):
        """
        训练简单的MLP分类器基于BERT嵌入。
        """
        self.logger.info(f"开始训练基于{model_name}的MLP分类器...")
        clf_mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, random_state=42)
        clf_mlp.fit(X_train_bert, y_train)
        self.models_dl[f'{model_name} MLP'] = clf_mlp
        self.logger.info(f"基于{model_name}的MLP分类器训练完成。")

    def prepare_word2vec_features(self):
        """
        提取Word2Vec特征。
        """
        # 假设你有一个函数用来生成Word2Vec的嵌入
        word2vec_model = self.load_word2vec_model()  # 加载训练好的Word2Vec模型
        X_test_w2v = [self.get_word2vec_embedding(text, word2vec_model) for text in self.test_df['text']]
        return np.array(X_test_w2v)

    def prepare_fasttext_features(self):
        """
        提取FastText特征。
        """
        # 假设你有一个函数用来生成FastText的嵌入
        fasttext_model = self.load_fasttext_model()  # 加载训练好的FastText模型
        X_test_ft = [self.get_fasttext_embedding(text, fasttext_model) for text in self.test_df['text']]
        return np.array(X_test_ft)

    def prepare_textcnn(self):
        """
        准备并训练TextCNN模型。
        """
        self.logger.info("开始训练TextCNN模型...")
        # 准备数据
        X_train = self.tokenizer_cnn_lstm.texts_to_sequences(self.train_df['text'].tolist())
        X_test = self.tokenizer_cnn_lstm.texts_to_sequences(self.test_df['text'].tolist())
        X_train = pad_sequences(X_train, maxlen=self.max_sequence_length)
        X_test = pad_sequences(X_test, maxlen=self.max_sequence_length)
        y_train = self.train_df['label'].values
        y_test = self.test_df['label'].values

        # 创建TextCNN模型
        model = Sequential()
        model.add(Embedding(input_dim=self.max_num_words,  # 使用定义的最大词汇数量
                            output_dim=100,
                            input_length=self.max_sequence_length))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # 训练模型
        model.fit(X_train, y_train,
                  epochs=5,
                  batch_size=32,
                  validation_data=(X_test, y_test),
                  verbose=1)

        self.models_dl['TextCNN'] = model
        self.logger.info("TextCNN模型训练完成。")

    def prepare_lstm(self):
        """
        准备并训练LSTM模型。
        """
        self.logger.info("开始训练LSTM模型...")
        # 准备数据
        X_train = self.tokenizer_cnn_lstm.texts_to_sequences(self.train_df['text'].tolist())
        X_test = self.tokenizer_cnn_lstm.texts_to_sequences(self.test_df['text'].tolist())
        X_train = pad_sequences(X_train, maxlen=self.max_sequence_length)
        X_test = pad_sequences(X_test, maxlen=self.max_sequence_length)
        y_train = self.train_df['label'].values
        y_test = self.test_df['label'].values

        # 创建LSTM模型
        model = Sequential()
        model.add(Embedding(input_dim=self.max_num_words,  # 使用定义的最大词汇数量
                            output_dim=100,
                            input_length=self.max_sequence_length))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # 训练模型
        model.fit(X_train, y_train,
                  epochs=5,
                  batch_size=32,
                  validation_data=(X_test, y_test),
                  verbose=1)

        self.models_dl['LSTM'] = model
        self.logger.info("LSTM模型训练完成。")

    def prepare_word2vec_classifier(self):
        """
        准备并训练基于Word2Vec特征的分类器。
        """
        self.logger.info("开始训练基于Word2Vec的分类器...")

        # 准备特征
        def get_avg_word2vec(text):
            words = text.lower().split()
            vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[word])
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.word2vec_model.vector_size)

        X_train_w2v = np.array([get_avg_word2vec(text) for text in self.train_df['text'].tolist()])
        X_test_w2v = np.array([get_avg_word2vec(text) for text in self.test_df['text'].tolist()])
        y_train = self.train_df['label'].values
        y_test = self.test_df['label'].values

        # 训练分类器
        clf_w2v = LogisticRegression(max_iter=1000)
        clf_w2v.fit(X_train_w2v, y_train)
        self.models_ml['Word2Vec Logistic Regression'] = clf_w2v  # 存储在 models_ml
        self.logger.info("基于Word2Vec的Logistic Regression分类器训练完成。")


    def prepare_fasttext_classifier(self):
        """
        准备并训练基于FastText特征的分类器。
        """
        self.logger.info("开始训练基于FastText的分类器...")

        # 准备特征
        def get_avg_fasttext(text):
            words = text.lower().split()
            vectors = []
            for word in words:
                if word in self.fasttext_model.wv:
                    vectors.append(self.fasttext_model.wv[word])
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.fasttext_model.vector_size)

        X_train_ft = np.array([get_avg_fasttext(text) for text in self.train_df['text'].tolist()])
        X_test_ft = np.array([get_avg_fasttext(text) for text in self.test_df['text'].tolist()])
        y_train = self.train_df['label'].values
        y_test = self.test_df['label'].values

        # 训练分类器
        clf_ft = LogisticRegression(max_iter=1000)
        clf_ft.fit(X_train_ft, y_train)
        self.models_ml['FastText Logistic Regression'] = clf_ft  # 存储在 models_ml
        self.logger.info("基于FastText的Logistic Regression分类器训练完成。")

    def train_dl_models(self):
        """
        训练所有深度学习模型。
        """
        # ALBERT模型
        X_train_bert, X_test_bert = self.prepare_bert_embeddings(model_name='albert-base-v2')
        self.train_bert_model(X_train_bert, self.train_df['label'], model_name='albert-base-v2')

        # TextCNN模型
        self.prepare_textcnn()

        # LSTM模型
        self.prepare_lstm()

        # Word2Vec分类器（已移动到 models_ml）
        self.prepare_word2vec_classifier()

        # FastText分类器（已移动到 models_ml）
        self.prepare_fasttext_classifier()

        # 保存训练好的深度学习模型
        self.save_dl_models()

    def save_ml_models(self):
        ml_model_dir = os.path.join(self.model_dir, 'ml_models')
        if not os.path.exists(ml_model_dir):
            os.makedirs(ml_model_dir)

        # 保存传统机器学习模型
        for name, model in self.models_ml.items():
            if isinstance(model, LogisticRegression):  # 其他机器学习模型
                model_path = os.path.join(ml_model_dir, f"{name.replace(' ', '_')}.joblib")
                joblib.dump(model, model_path)
                self.logger.info(f"保存传统机器学习模型: {name} 到 {model_path}")

        # 如果存在Word2Vec或FastText模型，保存它们
        if hasattr(self, 'word2vec_model'):  # 假设你在实例中有word2vec_model
            word2vec_model_path = os.path.join(ml_model_dir, 'word2vec_model.bin')
            self.word2vec_model.save(word2vec_model_path)
            self.logger.info(f"保存Word2Vec模型到 {word2vec_model_path}")

        if hasattr(self, 'fasttext_model'):  # 假设你在实例中有fasttext_model
            fasttext_model_path = os.path.join(ml_model_dir, 'fasttext_model.bin')
            self.fasttext_model.save(fasttext_model_path)
            self.logger.info(f"保存FastText模型到 {fasttext_model_path}")

    def load_ml_models(self):
        """
        从磁盘加载所有传统机器学习模型。
        """
        ml_model_dir = os.path.join(self.model_dir, 'ml_models')
        if not os.path.exists(ml_model_dir):
            self.logger.warning("传统机器学习模型目录不存在，需重新训练模型。")
            return False

        model_files = os.listdir(ml_model_dir)
        for file in model_files:
            if file.endswith('.joblib'):
                model_name = file.replace('.joblib', '').replace('_', ' ')
                model_path = os.path.join(ml_model_dir, file)
                self.models_ml[model_name] = joblib.load(model_path)
                self.logger.info(f"加载传统机器学习模型: {model_name} 从 {model_path}")
        return True

    def save_dl_models(self):
        """
        保存所有深度学习模型到磁盘。
        """
        dl_model_dir = os.path.join(self.model_dir, 'dl_models')
        if not os.path.exists(dl_model_dir):
            os.makedirs(dl_model_dir)

        for name, model in self.models_dl.items():
            if isinstance(model, Sequential):
                # Keras 模型
                model_path = os.path.join(dl_model_dir, f"{name.replace(' ', '_')}.h5")
                model.save(model_path)
                self.logger.info(f"保存Keras深度学习模型: {name} 到 {model_path}")
            elif isinstance(model, MLPClassifier):
                # scikit-learn 的 MLPClassifier
                model_path = os.path.join(dl_model_dir, f"{name.replace(' ', '_')}.joblib")
                joblib.dump(model, model_path)
                self.logger.info(f"保存深度学习模型: {name} 到 {model_path}")
            else:
                # 其他 PyTorch 模型（假设所有其他模型都是 PyTorch）
                model_path = os.path.join(dl_model_dir, f"{name.replace(' ', '_')}.pt")
                torch.save(model.state_dict(), model_path)
                self.logger.info(f"保存PyTorch深度学习模型: {name} 到 {model_path}")

    def load_dl_models(self):
        """
        从磁盘加载所有深度学习模型。
        """
        dl_model_dir = os.path.join(self.model_dir, 'dl_models')
        if not os.path.exists(dl_model_dir):
            self.logger.warning("深度学习模型目录不存在，需重新训练模型。")
            return False

        model_files = os.listdir(dl_model_dir)
        for file in model_files:
            if file.endswith('.h5'):
                # 加载Keras模型
                model_name = file.replace('.h5', '').replace('_', ' ')
                model_path = os.path.join(dl_model_dir, file)
                self.models_dl[model_name] = keras_load_model(model_path)
                self.logger.info(f"加载Keras深度学习模型: {model_name} 从 {model_path}")
            elif file.endswith('.pt'):
                # 加载PyTorch模型
                model_name = file.replace('.pt', '').replace('_', ' ')
                model_path = os.path.join(dl_model_dir, file)

                # 需要重新定义模型结构才能加载 state_dict
                if model_name == 'TextCNN':
                    model = self._build_textcnn_model()
                elif model_name == 'LSTM':
                    model = self._build_lstm_model()
                else:
                    self.logger.warning(f"未知的 PyTorch 模型: {model_name}")
                    continue

                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models_dl[model_name] = model
                self.logger.info(f"加载PyTorch深度学习模型: {model_name} 从 {model_path}")
            elif file.endswith('.joblib'):
                # 加载其他模型（如 scikit-learn 的 MLPClassifier）
                model_name = file.replace('.joblib', '').replace('_', ' ')
                model_path = os.path.join(dl_model_dir, file)
                self.models_dl[model_name] = joblib.load(model_path)
                self.logger.info(f"加载深度学习模型: {model_name} 从 {model_path}")
        return True

    def _build_textcnn_model(self):
        """
        定义并返回一个TextCNN模型结构。
        """
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer_cnn_lstm.word_index) + 1,
                            output_dim=100,
                            input_length=self.max_sequence_length))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def _build_lstm_model(self):
        """
        定义并返回一个LSTM模型结构。
        """
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer_cnn_lstm.word_index) + 1,
                            output_dim=100,
                            input_length=self.max_sequence_length))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def predict_ml_models(self):
        self.logger.info("开始获取传统机器学习模型的预测概率...")
        probs_ml = {}
        _, _, X_test_ml = self.prepare_ml_features()  # 正确接收第三个返回值
        for name, model in self.models_ml.items():
            try:
                probs_ml[name] = model.predict_proba(X_test_ml)[:, 1]
                self.logger.info(f"{name} 预测完成，输出形状: {probs_ml[name].shape}")
            except Exception as e:
                self.logger.error(f"{name} 预测时出错: {e}")
        return probs_ml

    def predict_dl_models(self):
        """
        获取所有深度学习模型的预测概率。
        """
        self.logger.info("开始获取深度学习模型的预测概率...")
        probs_dl = {}

        # ALBERT MLP
        if 'albert-base-v2 MLP' in self.models_dl:
            clf = self.models_dl['albert-base-v2 MLP']
            X_test_bert, _ = self.prepare_bert_embeddings(model_name='albert-base-v2')
            probs_dl['ALBERT MLP'] = clf.predict_proba(X_test_bert)[:, 1]
            self.logger.info(f"ALBERT MLP预测完成，输出形状: {probs_dl['ALBERT MLP'].shape}")

        # TextCNN
        if 'TextCNN' in self.models_dl:
            model = self.models_dl['TextCNN']
            X_test = self.tokenizer_cnn_lstm.texts_to_sequences(self.test_df['text'].tolist())
            X_test = pad_sequences(X_test, maxlen=self.max_sequence_length)
            probs = model.predict(X_test, verbose=0).flatten()
            probs_dl['TextCNN'] = probs
            self.logger.info(f"TextCNN预测完成，输出形状: {probs_dl['TextCNN'].shape}")

        # LSTM
        if 'LSTM' in self.models_dl:
            model = self.models_dl['LSTM']
            X_test = self.tokenizer_cnn_lstm.texts_to_sequences(self.test_df['text'].tolist())
            X_test = pad_sequences(X_test, maxlen=self.max_sequence_length)
            probs = model.predict(X_test, verbose=0).flatten()
            probs_dl['LSTM'] = probs
            self.logger.info(f"LSTM预测完成，输出形状: {probs_dl['LSTM'].shape}")

        # Word2Vec Logistic Regression 和 FastText Logistic Regression 已移动到 models_ml

        return probs_dl

    def predict(self):
        """
        进行预测与融合。
        """
        # 获取机器学习模型的预测概率
        self.logger.info("开始获取传统机器学习模型的预测概率...")
        probs_ml_test = self.predict_ml_models()
        self.logger.info("传统机器学习模型预测完成。")

        # 获取深度学习模型的预测概率
        self.logger.info("开始获取深度学习模型的预测概率...")
        probs_dl_test = self.predict_dl_models()
        self.logger.info("深度学习模型预测完成。")

        # 融合
        self.logger.info("开始融合模型预测结果...")
        all_probs = np.zeros(len(self.test_labels))
        total_models = 0

        for name, probs in probs_ml_test.items():
            if len(probs) != len(all_probs):
                self.logger.warning(
                    f"模型 {name} 的预测结果长度 {len(probs)} 与测试集长度 {len(all_probs)} 不匹配，跳过该模型。")
                continue
            all_probs += probs
            total_models += 1

        for name, probs in probs_dl_test.items():
            if len(probs) != len(all_probs):
                self.logger.warning(
                    f"模型 {name} 的预测结果长度 {len(probs)} 与测试集长度 {len(all_probs)} 不匹配，跳过该模型。")
                continue
            all_probs += probs
            total_models += 1

        if total_models == 0:
            self.logger.error("没有任何模型进行预测。")
            return None

        probs_final = all_probs / total_models
        y_pred = (probs_final >= 0.5).astype(int)

        print("total_models:", total_models,"==============================================================")

        # 评估
        self.logger.info("--- 融合模型评估 ---")
        accuracy = accuracy_score(self.test_labels, y_pred)
        roc_auc = roc_auc_score(self.test_labels, probs_final)
        precision = precision_score(self.test_labels, y_pred, zero_division=0)
        recall = recall_score(self.test_labels, y_pred, zero_division=0)
        f1 = f1_score(self.test_labels, y_pred, zero_division=0)

        report = classification_report(self.test_labels, y_pred, target_names=['No', 'Yes'], zero_division=0)
        cm = confusion_matrix(self.test_labels, y_pred)

        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"ROC AUC: {roc_auc:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        self.logger.info(f"Confusion Matrix:\n{cm}\n")

        return y_pred

    def run(self):
        self.logger.info("开始进行预测...")

        # 尝试加载传统机器学习模型
        ml_loaded = self.load_ml_models()

        # 尝试加载深度学习模型
        dl_loaded = self.load_dl_models()

        # 如果任一类别的模型未加载成功，则进行训练
        if not ml_loaded or not dl_loaded:
            if not ml_loaded:
                self.logger.info("开始训练传统机器学习模型...")
                X_train_ml_resampled, y_train_resampled, X_test_ml = self.prepare_ml_features()
                self.train_ml_models(X_train_ml_resampled, y_train_resampled)

            if not dl_loaded:
                self.logger.info("开始训练深度学习模型...")
                self.train_dl_models()

        # 无论是加载还是训练，进行预测
        y_pred = self.predict()
        if y_pred is not None:
            self.logger.info("预测完成。")
        else:
            self.logger.error("预测未完成，由于缺少有效的模型。")

    def unload_models(self):
        """
        释放模型内存。
        """
        self.logger.info("开始释放模型内存...")
        for model_name, model in self.models_ml.items():
            del model
        for model_name, model in self.models_dl.items():
            del model
        torch.cuda.empty_cache()
        self.logger.info("所有模型内存已释放。")


if __name__ == "__main__":
    # 请确保Excel文件路径正确，并包含 'text' 和 'label' 两列
    classifier = PreCrash(excel_path='Final_DeepL_dataset_v1.0.xlsx')
    classifier.run()
    classifier.unload_models()

