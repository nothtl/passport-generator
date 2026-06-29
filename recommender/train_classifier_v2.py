"""Train optimized classifier targeting 95% accuracy."""
import pandas as pd, numpy as np, sys, time, re, pickle, os, json
sys.path.insert(0, '.')

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
for f, c in sorted(df['function'].value_counts().items()):
    print('  {}: {}'.format(f, c))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Approach 1: Rich features with LogisticRegression
# Word + character n-grams for subword patterns
best_configs = []

# Config A: Word 1-3 grams + Char 3-5 grams, LogisticRegression
tfidf_word = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), stop_words='english', min_df=2, max_df=0.8)
tfidf_char = TfidfVectorizer(max_features=10000, analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_df=0.8)
combined = FeatureUnion([('word', tfidf_word), ('char', tfidf_char)])
pipeline_a = Pipeline([
    ('tfidf', combined),
    ('clf', LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced', solver='saga')),
])
best_configs.append(('Word+Char-LR', pipeline_a))

# Config B: Same features, LinearSVC (our previous winner, with more features)
pipeline_b = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1, 3), stop_words='english', min_df=2, max_df=0.8)),
    ('clf', CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', max_iter=3000), cv=3)),
])
best_configs.append(('SVC-25K-3gram', pipeline_b))

# Config C: SGDClassifier with early stopping
pipeline_c = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1, 3), stop_words='english', min_df=2, max_df=0.8)),
    ('clf', CalibratedClassifierCV(SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=100, class_weight='balanced', random_state=42, learning_rate='adaptive', eta0=0.1), cv=3)),
])
best_configs.append(('SGD-25K', pipeline_c))

# Config D: Stacking ensemble
base_models = [
    ('lr', LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')),
    ('svc', CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', max_iter=2000), cv=3)),
]
pipeline_d = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english', min_df=2, max_df=0.8)),
    ('clf', StackingClassifier(base_models, final_estimator=LogisticRegression(max_iter=1000), cv=3)),
])
best_configs.append(('Stacking-LR+SVC', pipeline_d))

print('\nTesting {} configurations...'.format(len(best_configs)))
print('-' * 70)

best_score = 0
best_name = None
best_pipeline = None

for name, pipeline in best_configs:
    t0 = time.time()
    scores = cross_val_score(pipeline, df['clean'], df['function'], cv=cv, scoring='accuracy')
    elapsed = time.time() - t0
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Top-3
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

# Save
with open('recommender/data/resume_classifier.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
with open('recommender/data/resume_classifier_classes.json', 'w') as f:
    json.dump(list(best_pipeline.classes_), f)
print('Model saved')

# Per-function
print('\nPer-function (5-fold CV):')
preds = cross_val_predict(best_pipeline, df['clean'], df['function'], cv=5)
for func in sorted(df['function'].unique()):
    mask = df['function'] == func
    correct = (preds[mask] == func).sum()
    total = mask.sum()
    bar = '#' * int(correct/total*10) + '.' * (10-int(correct/total*10))
    print('  {:25s} [{}] {:4.0f}% (n={:3d})'.format(func, bar, correct/total*100, total))
