#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
import optuna

# ============================
# 1. Завантаження даних
# ============================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ============================
# 2. EDA
# ============================
plt.figure(figsize=(6,4))
sns.countplot(data=train, x="AdoptionSpeed", palette="viridis")
plt.title("Розподіл AdoptionSpeed у train")
plt.show()

train["desc_len"] = train["Description"].fillna("").apply(len)
plt.figure(figsize=(6,4))
sns.histplot(train["desc_len"], bins=30, kde=True)
plt.title("Довжина описів у train")
plt.show()

print("\n=== Приклади описів ===")
for i in range(3):
    print(f"{train.iloc[i]['PetID']}: {train.iloc[i]['Description'][:120]}...")

print("\nПропуски у train:\n", train.isnull().sum())
print("\nПропуски у test:\n", test.isnull().sum())

# ============================
# 3. Text features (TF-IDF)
# ============================
def clean_text(text):
    if pd.isnull(text): return ""
    text = text.lower()
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())

tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_train = tfidf.fit_transform(train["Description"].fillna("").apply(clean_text))
tfidf_test = tfidf.transform(test["Description"].fillna("").apply(clean_text))

# ============================
# 4. Image features (ResNet50)
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
resnet_features = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_features(df, folder):
    feats = []
    for pet_id in tqdm(df["PetID"], desc=f"Extracting {folder}"):
        path = os.path.join(folder, f"{pet_id}-1.jpg")
        if not os.path.exists(path):
            feats.append(np.zeros(2048)); continue
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = resnet_features(img).view(-1).cpu().numpy()
        feats.append(f)
    return np.array(feats)

train_img = extract_features(train, "images/images/train")
test_img = extract_features(test, "images/images/test")

# ============================
# 5. Combine features
# ============================
X_train = hstack([csr_matrix(tfidf_train, dtype=np.float32), csr_matrix(train_img, dtype=np.float32)])
X_test = hstack([csr_matrix(tfidf_test, dtype=np.float32), csr_matrix(test_img, dtype=np.float32)])

y = train["AdoptionSpeed"].astype(int)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 6. Optuna + LightGBM
# ============================
def objective(trial):
    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    params = {
        "objective": "multiclass",
        "num_class": 5,                 # 0–4
        "metric": "multi_logloss",
        "boosting_type": boosting,
        "n_estimators": 4000,           # більше дерев + lower lr
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),

        # обмежуємо складність (менше оверфіту)
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),

        # стохастика для кращої генералізації
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 3),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),

        # контроль розщеплень
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),

        # м’яка регуляризація (зашкалюючі значення часто шкодять)
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),

        # додатково стабілізує тренування
        "max_bin": trial.suggest_int("max_bin", 127, 255),

        # допомагає при дисбалансі (якщо він є)
        "class_weight": "balanced",

        # відтворюваність
        "random_state": 42,
        "n_jobs": -1,
    }

    # специфічні налаштування для dart
    if boosting == "dart":
        params.update({
            "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.3),
            "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
            "max_drop": trial.suggest_int("max_drop", 20, 100),
        })

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_metric=["multi_logloss", "multi_error"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )

    preds = model.predict(X_val)
    score = cohen_kappa_score(y_val, preds, weights="quadratic")
    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:", study.best_trial.params)

# ============================
# 7. Train final model
# ============================
best_params = study.best_trial.params
best_params.update({"objective": "multiclass", "num_class": 5})
final_model = lgb.LGBMClassifier(**best_params, n_estimators=1000)

final_model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],   # і train, і validation
    eval_metric=["multi_logloss", "multi_error"],
    callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)]
)

# Збереження результатів
evals_result = final_model.evals_result_

# Loss
train_loss = evals_result['training']['multi_logloss']
val_loss   = evals_result['valid_1']['multi_logloss']

# Accuracy
train_acc = [1 - x for x in evals_result['training']['multi_error']]
val_acc   = [1 - x for x in evals_result['valid_1']['multi_error']]

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.xlabel('Iteration')
plt.ylabel('Multi Logloss')
plt.title('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1,2,2)
plt.plot(train_acc, label='Train Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='red')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (1 - multi_error)')
plt.title('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================
# 8. Predict test + save
# ============================
y_test_pred = final_model.predict(X_test)

submission = pd.DataFrame({
    "PetID": test["PetID"],
    "AdoptionSpeed": y_test_pred.astype(int)
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv збережено")

# ============================
# 9. Аналіз результатів
# ============================
print("\nРозподіл прогнозів на тесті:")
print(submission["AdoptionSpeed"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=submission, x="AdoptionSpeed", palette="viridis")
plt.title("Розподіл AdoptionSpeed у прогнозах")
plt.show()
