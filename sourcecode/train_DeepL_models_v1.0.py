import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import re
import random
import logging
import warnings

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import gensim
from gensim.models import Word2Vec, FastText

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

import torch

# 导入SMOTE
from imblearn.over_sampling import SMOTE

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 忽略FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 设置随机种子以确保结果可复现
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保PyTorch的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# 定义全局路径
DATA_PATH = 'Final_DeepL_dataset_v1.0.xlsx'
OUTPUT_DIR = 'outputs'

MODEL_DIR = os.path.join(OUTPUT_DIR, 'trained_models')
os.makedirs(MODEL_DIR, exist_ok=True)
EVAL_DIR = os.path.join(OUTPUT_DIR, 'evaluation_results')
os.makedirs(EVAL_DIR, exist_ok=True)

ALBERT_DIR = os.path.join(MODEL_DIR, 'albert_model')
LSTM_DIR = os.path.join(MODEL_DIR, 'lstm_model.keras')
TEXTCNN_DIR = os.path.join(MODEL_DIR, 'textcnn_model.keras')
WORD2VEC_DIR = os.path.join(MODEL_DIR, 'word2vec_model.model')
FASTTEXT_DIR = os.path.join(MODEL_DIR, 'fasttext_model.model')

RESULTS_FILE = os.path.join(EVAL_DIR, 'DeepL_models_results.xlsx')

# 文本预处理函数
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 加载数据函数
def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df[['label'] + [col for col in df.columns if col != 'label']].dropna()
    return df

# 标签编码函数
def encode_labels(y):
    le = LabelEncoder()
    y_str = y.astype(str)  # 转换为字符串
    y_encoded = le.fit_transform(y_str)
    return y_encoded, le

# 保存结果函数
def save_results(results, model_name):
    results['model'] = model_name
    results_df = pd.DataFrame([{
        'model': results['model'],
        'accuracy': results.get('accuracy', results.get('eval_accuracy')),
        'precision': results.get('precision', results.get('eval_precision')),
        'recall': results.get('recall', results.get('eval_recall')),
        'f1_score': results.get('f1_score', results.get('eval_f1_score'))
    }])
    if os.path.exists(RESULTS_FILE):
        existing_df = pd.read_excel(RESULTS_FILE)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        combined_df = results_df
    combined_df.to_excel(RESULTS_FILE, index=False)

# 保存详细分类报告
def save_detailed_report(report, model_name):
    report_file = os.path.join(EVAL_DIR, f'{model_name}_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, label_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(EVAL_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

# 评估模型函数
def evaluate_model(y_true, y_pred, num_labels, label_names, model_name):
    # 将 label_names 转换为字符串类型
    label_names_str = [str(name) for name in label_names]

    if num_labels > 2:
        average_type = 'macro'
    else:
        average_type = 'binary'

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=label_names_str, zero_division=0)

    logging.info(f"Accuracy: {acc}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"\nClassification Report:\n{report}")

    # 保存详细分类报告
    save_detailed_report(report, model_name)

    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_true, y_pred, label_names_str, model_name)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# 自定义 Trainer 以支持类权重
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # 确保类权重在与 logits 相同的设备上
        class_weights = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ALBERT模型训练函数
def train_albert(df_train, df_val, label_names, num_labels, epochs=5, batch_size=16):
    logging.info("训练 ALBERT 模型...")
    pretrained_model_name = 'albert-base-v2'
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name)

    train_texts = df_train['text'].tolist()
    val_texts = df_val['text'].tolist()
    train_labels = df_train['label'].tolist()
    val_labels = df_val['label'].tolist()

    # 计算类权重
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logging.info(f"类权重: {class_weights}")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    class AlbertDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = AlbertDataset(train_encodings, train_labels)
    val_dataset = AlbertDataset(val_encodings, val_labels)

    model = AlbertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=ALBERT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",  # 将 evaluation_strategy 替换为 eval_strategy
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_dir=os.path.join(ALBERT_DIR, 'logs'),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1_score',
        save_total_limit=1,
        seed=42,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=500
    )

    def compute_metrics_func(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            'accuracy': accuracy_score(p.label_ids, preds),
            'precision': precision_score(p.label_ids, preds, average='macro', zero_division=0),
            'recall': recall_score(p.label_ids, preds, average='macro', zero_division=0),
            'f1_score': f1_score(p.label_ids, preds, average='macro', zero_division=0)
        }

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights_tensor
    )

    trainer.train()
    model.save_pretrained(ALBERT_DIR)
    tokenizer.save_pretrained(ALBERT_DIR)

    results = trainer.evaluate(val_dataset)
    logging.info(f"ALBERT evaluation results: {results}")  # 添加调试日志
    save_results(results, "ALBERT")
    logging.info("ALBERT 模型训练和评估完成。")
    return model

# TextCNN模型训练函数
def train_textcnn(df_train, df_val, label_names, num_labels, epochs=3, batch_size=16):
    logging.info("训练 TextCNN 模型...")
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(df_train['text'])
    train_sequences = tokenizer.texts_to_sequences(df_train['text'])
    val_sequences = tokenizer.texts_to_sequences(df_val['text'])

    max_len = max([len(seq) for seq in train_sequences])
    train_data = pad_sequences(train_sequences, maxlen=max_len)
    val_data = pad_sequences(val_sequences, maxlen=max_len)

    # 计算类权重
    from sklearn.utils.class_weight import compute_class_weight
    unique_labels = np.unique(df_train['label'])
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=df_train['label'])
    class_weights_dict = {i: class_weights[i] for i in unique_labels}
    logging.info(f"类权重: {class_weights_dict}")

    # 保持多个输出神经元和 softmax 激活函数
    model = Sequential([
        Embedding(input_dim=100000, output_dim=100),  # 移除 input_length 参数
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_labels, activation='softmax')  # 修改为多分类
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = 'sparse_categorical_crossentropy'  # 保持为 sparse_categorical_crossentropy
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Convert labels to numpy arrays
    y_train = df_train['label'].values
    y_val = df_val['label'].values

    # 打印唯一标签以验证
    logging.info(f"Unique labels in training set: {np.unique(y_train)}")
    logging.info(f"Unique labels in validation set: {np.unique(y_val)}")

    # 训练模型时传递类权重
    try:
        model.fit(
            train_data,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, y_val),
            shuffle=True,
            class_weight=class_weights_dict
        )
    except Exception as e:
        logging.error("模型训练过程中发生错误。", exc_info=True)
        raise e

    model.save(TEXTCNN_DIR)

    # 评估
    try:
        val_predictions = model.predict(val_data)
        val_predictions = np.argmax(val_predictions, axis=1)

        results = evaluate_model(y_val, val_predictions, num_labels, label_names, "TextCNN")
        save_results(results, "TextCNN")
        logging.info("TextCNN 模型训练和评估完成。")
    except Exception as e:
        logging.error("模型评估过程中发生错误。", exc_info=True)
        raise e

    return model

# LSTM模型训练函数
def train_lstm(df_train, df_val, label_names, num_labels, epochs=3, batch_size=16):
    logging.info("训练 LSTM 模型...")

    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(df_train['text'])

    train_sequences = tokenizer.texts_to_sequences(df_train['text'])
    val_sequences = tokenizer.texts_to_sequences(df_val['text'])

    max_len = max([len(seq) for seq in train_sequences])
    train_data = pad_sequences(train_sequences, maxlen=max_len)
    val_data = pad_sequences(val_sequences, maxlen=max_len)

    # 计算类权重
    from sklearn.utils.class_weight import compute_class_weight
    unique_labels = np.unique(df_train['label'])
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=df_train['label'])
    class_weights_dict = {i: class_weights[i] for i in unique_labels}
    logging.info(f"类权重: {class_weights_dict}")

    # LSTM 模型定义
    model = Sequential([
        Embedding(input_dim=100000, output_dim=100),  # 移除 input_length 参数
        LSTM(64, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_labels, activation='softmax')  # 修改为多分类
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Convert labels to numpy arrays
    y_train = df_train['label'].values
    y_val = df_val['label'].values

    # 打印唯一标签以验证
    logging.info(f"Unique labels in training set: {np.unique(y_train)}")
    logging.info(f"Unique labels in validation set: {np.unique(y_val)}")

    try:
        model.fit(
            train_data,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, y_val),
            shuffle=True,
            class_weight=class_weights_dict
        )
    except Exception as e:
        logging.error("模型训练过程中发生错误。", exc_info=True)
        raise e

    model.save(LSTM_DIR)

    # 评估
    try:
        val_predictions = model.predict(val_data)
        val_predictions = np.argmax(val_predictions, axis=1)

        # 打印预测和真实标签的形状
        logging.info(f"y_true shape: {y_val.shape}, y_pred shape: {val_predictions.shape}")

        results = evaluate_model(y_val, val_predictions, num_labels, label_names, "LSTM")
        save_results(results, "LSTM")
        logging.info("LSTM 模型训练和评估完成。")
    except Exception as e:
        logging.error("模型评估过程中发生错误。", exc_info=True)
        raise e

    return model

# Word2Vec模型训练函数
def train_word2vec(df_train, df_val, label_names, num_labels, le, epochs=3, batch_size=16):
    logging.info("训练 Word2Vec 模型...")
    sentences = [text.split() for text in df_train['text']]
    model = Word2Vec(sentences, vector_size=300, window=10, min_count=10, workers=4, epochs=epochs)
    output_dir = os.path.dirname(WORD2VEC_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(WORD2VEC_DIR)

    def vectorize_text(text, model):
        word_vectors = [model.wv[word] if word in model.wv else np.zeros(model.vector_size) for word in text.split()]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    X_train = np.array([vectorize_text(text, model) for text in df_train['text']])
    X_val = np.array([vectorize_text(text, model) for text in df_val['text']])

    y_train = df_train['label'].values
    y_val = df_val['label'].values

    # 应用SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logging.info(f"SMOTE后训练集大小: {X_train_res.shape}, {y_train_res.shape}")

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_res, y_train_res)

    y_pred = classifier.predict(X_val)
    results = evaluate_model(y_val, y_pred, num_labels, le.classes_, "Word2Vec")
    save_results(results, "Word2Vec")
    logging.info("Word2Vec 模型训练和评估完成。")
    return model

# FastText模型训练函数
def train_fasttext(df_train, df_val, label_names, num_labels, le, epochs=3, batch_size=16):
    logging.info("训练 FastText 模型...")
    sentences = [text.split() for text in df_train['text']]
    model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=epochs)
    model.save(FASTTEXT_DIR)

    def vectorize_text(text, model):
        word_vectors = [model.wv[word] if word in model.wv else np.zeros(model.vector_size) for word in text.split()]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    X_train = np.array([vectorize_text(text, model) for text in df_train['text']])
    X_val = np.array([vectorize_text(text, model) for text in df_val['text']])

    y_train = df_train['label'].values
    y_val = df_val['label'].values

    # 应用SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logging.info(f"SMOTE后训练集大小: {X_train_res.shape}, {y_train_res.shape}")

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_res, y_train_res)

    y_pred = classifier.predict(X_val)
    results = evaluate_model(y_val, y_pred, num_labels, le.classes_, "FastText")
    save_results(results, "FastText")
    logging.info("FastText 模型训练和评估完成。")
    return model

# 加载和预处理数据函数
def load_and_preprocess_data(file_path):
    logging.info("加载和预处理数据...")
    df = load_data(file_path)
    df['text'] = df['text'].apply(preprocess_text)
    y, le = encode_labels(df['label'])
    df['label'] = y
    logging.info("数据加载和预处理完成。")
    return df, le

# 汇总并打印所有模型的评估结果
def summarize_results():
    results_file = RESULTS_FILE
    if os.path.exists(results_file):
        df_results = pd.read_excel(results_file)
        print("\n================= 所有模型的评估结果 =================")
        print(df_results.to_string(index=False))
        print("=======================================================")
    else:
        print("评估结果文件不存在，请确保模型已成功训练和评估。")

# 主函数
def main():
    try:
        # 1. 加载和预处理数据
        df, le = load_and_preprocess_data(DATA_PATH)
        label_names = le.classes_.astype(str)
        num_labels = len(label_names)
        logging.info(f"标签数量: {num_labels} - {label_names}")

        # 2. 数据划分
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        logging.info(f"训练样本数量: {len(df_train)}, 验证样本数量: {len(df_val)}")

        # 打印训练和验证集的标签分布
        logging.info(f"训练集标签分布:\n{df_train['label'].value_counts()}")
        logging.info(f"验证集标签分布:\n{df_val['label'].value_counts()}")

        # 3. 训练模型
        models = {}
        # 对于深度学习模型，使用类权重而不是SMOTE
        models['ALBERT'] = train_albert(df_train, df_val, label_names, num_labels, epochs=3, batch_size=16)
        models['TextCNN'] = train_textcnn(df_train, df_val, label_names, num_labels, epochs=3, batch_size=16)
        models['LSTM'] = train_lstm(df_train, df_val, label_names, num_labels, epochs=3, batch_size=16)
        models['Word2Vec'] = train_word2vec(df_train, df_val, label_names, num_labels, le, epochs=3, batch_size=16)
        models['FastText'] = train_fasttext(df_train, df_val, label_names, num_labels, le, epochs=3, batch_size=16)

        logging.info("所有模型已成功训练和评估。")

        # 4. 汇总并输出所有模型的评估结果
        summarize_results()

    except Exception as e:
        logging.error("模型训练过程中发生错误。", exc_info=True)

if __name__ == '__main__':
    main()
