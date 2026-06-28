"""Train optimal resume classifier on 2,484 real resumes and save model."""
import pandas as pd, numpy as np, sys, time, re, pickle, os
from collections import Counter
sys.path.insert(0, '.')

# Load dataset
df = pd.read_csv('C:/Users/Tingli/.cache/huggingface/hub/datasets--opensporks--resumes/snapshots/ed4cb5f3fd1ce7e0a0e74e1a09c1a3b702c2c2eb/Resume/Resume.csv')

CAT_MAP = {
    'ACCOUNTANT': 'finance', 'ADVOCATE': 'legal', 'AGRICULTURE': 'agriculture',
    'APPAREL': 'design', 'ARTS': 'arts-media', 'AUTOMOBILE': 'skilled-trade',
    'AVIATION': 'logistics', 'BANKING': 'finance', 'BPO': 'support',
    'BUSINESS-DEVELOPMENT': 'sales', 'CHEF': 'food-service', 'CONSTRUCTION': 'skilled-trade',
    'CONSULTANT': 'ops', 'DESIGNER': 'design', 'DIGITAL-MEDIA': 'arts-media',
    'ENGINEERING': 'technology', 'FINANCE': 'finance', 'FITNESS': 'personal-care',
    'HEALTHCARE': 'healthcare', 'HR': 'administrative', 'INFORMATION-TECHNOLOGY': 'technology',
    'PUBLIC-RELATIONS': 'arts-media', 'SALES': 'sales', 'TEACHER': 'education',
}
df['function'] = df['Category'].map(CAT_MAP)
df = df.dropna(subset=['function'])

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean'] = df['Resume_str'].apply(clean)
print('Loaded {} resumes, {} functions'.format(len(df), df['function'].nunique()))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Try multiple configurations
configs = [
    ('LogReg-5K-1gram', Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1), stop_words='english', min_df=2)),
        ('clf', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')),
    ])),
    ('LogReg-10K-12gram', Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', min_df=2)),
        ('clf', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')),
    ])),
    ('LogReg-15K-123gram', Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words='english', min_df=2, max_df=0.8)),
        ('clf', LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')),
    ])),
    ('SVM-10K', Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', min_df=2)),
        ('clf', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000), cv=3)),
    ])),
    ('SGD-10K', Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', min_df=2)),
        ('clf', CalibratedClassifierCV(SGDClassifier(loss='log_loss', max_iter=2000, class_weight='balanced', random_state=42), cv=3)),
    ])),
    ('Ensemble', Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', min_df=2)),
        ('clf', VotingClassifier([
            ('lr', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')),
            ('svm', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000), cv=3)),
        ], voting='soft')),
    ])),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_name = None
best_pipeline = None

print('\nTesting {} configurations...'.format(len(configs)))
print('-' * 70)
for name, pipeline in configs:
    t0 = time.time()
    scores = cross_val_score(pipeline, df['clean'], df['function'], cv=cv, scoring='accuracy')
    elapsed = time.time() - t0
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Also compute top-3
    pipeline.fit(df['clean'], df['function'])
    probas = cross_val_predict(pipeline, df['clean'], df['function'], cv=5, method='predict_proba')
    classes = pipeline.classes_
    top3 = 0
    for i, (true_func, proba) in enumerate(zip(df['function'], probas)):
        top3_idx = np.argsort(-proba)[:3]
        if true_func in [classes[j] for j in top3_idx]:
            top3 += 1
    top3_pct = top3 / len(df)

    print('{:30s} Top-1: {:.1%} (+/-{:.1%})  Top-3: {:.1%}  Time: {:.0f}s'.format(
        name, mean_score, std_score, top3_pct, elapsed))

    if mean_score > best_score:
        best_score = mean_score
        best_name = name
        best_pipeline = pipeline

print('\nBest: {} @ {:.1%}'.format(best_name, best_score))

# Train best model on full data
print('\nTraining best model on full dataset...')
t0 = time.time()
best_pipeline.fit(df['clean'], df['function'])
print('Trained in {:.0f}s'.format(time.time() - t0))

# Save model
model_path = 'recommender/data/resume_classifier.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_pipeline, f)
print('Saved to {}'.format(model_path))

# Save function list
func_path = 'recommender/data/resume_classifier_classes.json'
import json
with open(func_path, 'w') as f:
    json.dump(list(best_pipeline.classes_), f)
print('Saved classes to {}'.format(func_path))

# Per-function accuracy on full training set
print('\nPer-function accuracy (5-fold CV):')
preds = cross_val_predict(best_pipeline, df['clean'], df['function'], cv=5)
for func in sorted(df['function'].unique()):
    mask = df['function'] == func
    correct = (preds[mask] == func).sum()
    total = mask.sum()
    bar = '#' * int(correct/total*10) + '.' * (10-int(correct/total*10))
    print('  {:25s} [{}] {:4.0f}% (n={:3d})'.format(func, bar, correct/total*100, total))
