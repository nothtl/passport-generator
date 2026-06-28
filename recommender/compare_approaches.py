"""Compare TF-IDF classifier vs SentenceTransformer on 2,484 real resumes."""
import pandas as pd, numpy as np, sys, time, re
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
print('Distribution:', dict(df['function'].value_counts()))

# ============================================================
# APPROACH 1: TF-IDF + Logistic Regression
# ============================================================
print('\n' + '='*60)
print('APPROACH 1: TF-IDF + Logistic Regression')
print('='*60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

t0 = time.time()

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2)),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', multi_class='multinomial')),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_top1 = cross_val_score(pipeline, df['clean'], df['function'], cv=cv, scoring='accuracy')
train_time = time.time() - t0

scores_formatted = ', '.join(['{:.1%}'.format(s) for s in scores_top1])
print('Training time: {:.0f}s'.format(train_time))
print('5-fold CV scores: [{}]'.format(scores_formatted))
print('Mean top-1 accuracy: {:.1%} (+/- {:.1%})'.format(np.mean(scores_top1), np.std(scores_top1)))

# Top-3
pipeline.fit(df['clean'], df['function'])
probas = cross_val_predict(pipeline, df['clean'], df['function'], cv=5, method='predict_proba')
classes = pipeline.classes_

top3_correct = 0
for i, (true_func, proba) in enumerate(zip(df['function'], probas)):
    top3_idx = np.argsort(-proba)[:3]
    top3_funcs = [classes[j] for j in top3_idx]
    if true_func in top3_funcs:
        top3_correct += 1

classifier_top3 = top3_correct / len(df)
print('Top-3 accuracy: {:.1%}'.format(classifier_top3))

print('\nPer-function (classifier):')
from sklearn.model_selection import cross_val_predict as cvp
preds = cross_val_predict(pipeline, df['clean'], df['function'], cv=5)
for func in sorted(df['function'].unique()):
    mask = df['function'] == func
    correct = (preds[mask] == func).sum()
    total = mask.sum()
    print('  {}: {:.0f}% (n={})'.format(func, correct/total*100, total))

# ============================================================
# APPROACH 2: SentenceTransformer
# ============================================================
print('\n' + '='*60)
print('APPROACH 2: SentenceTransformer Embeddings')
print('='*60)

t0 = time.time()
from sentence_transformers import SentenceTransformer
print('Loading model...')
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print('Loaded in {:.0f}s'.format(time.time()-t0))

# Embed O*NET occupation titles
from recommender.match.onet_matcher import _load_index, _occ_to_function
index = _load_index()
occs = index['occs']

occ_texts = [occ['title'] for occ in occs]
occ_functions = [_occ_to_function(occ['title']) or 'unmapped' for occ in occs]

print('Embedding {} occupations...'.format(len(occ_texts)))
t0 = time.time()
occ_embeddings = model.encode(occ_texts, normalize_embeddings=True, show_progress_bar=True)
print('Embedded in {:.0f}s'.format(time.time()-t0))

# Test on 500 sample
sample_size = min(500, len(df))
sample = df.sample(n=sample_size, random_state=42)

print('Embedding {} resumes...'.format(sample_size))
t0 = time.time()
resume_embeddings = model.encode(sample['clean'].tolist(), normalize_embeddings=True, show_progress_bar=True)
similarities = model.similarity(resume_embeddings, occ_embeddings)

st_correct_top1 = 0
st_correct_top3 = 0

for i, (_, row) in enumerate(sample.iterrows()):
    true_func = row['function']
    sim = similarities[i].numpy()
    top_idx = np.argsort(-sim)[:20]

    func_scores = {}
    for idx in top_idx:
        func = occ_functions[idx]
        score = float(sim[idx])
        if func not in func_scores or score > func_scores[func]:
            func_scores[func] = score

    ranked = sorted(func_scores.items(), key=lambda x: -x[1])
    pred_func = ranked[0][0] if ranked else None
    top3_funcs = [f for f, _ in ranked[:3]]

    if pred_func == true_func: st_correct_top1 += 1
    if true_func in top3_funcs: st_correct_top3 += 1

st_time = time.time() - t0
print('\nInference time: {:.0f}s ({:.0f}ms per resume)'.format(st_time, st_time/sample_size*1000))
print('Top-1 accuracy: {}/{} = {:.1%}'.format(st_correct_top1, sample_size, st_correct_top1/sample_size))
print('Top-3 accuracy: {}/{} = {:.1%}'.format(st_correct_top3, sample_size, st_correct_top3/sample_size))

# ============================================================
# COMPARISON
# ============================================================
print('\n' + '='*60)
print('COMPARISON')
print('='*60)
print('{:40s} {:>8s} {:>8s} {:>10s}'.format('Approach', 'Top-1', 'Top-3', 'Time'))
print('-' * 70)
print('{:40s} {:>7.1%} {:>7.1%} {:>10s}'.format(
    'TF-IDF + Logistic Regression', np.mean(scores_top1), classifier_top3, '<5s train'))
print('{:40s} {:>7.1%} {:>7.1%} {:>10s}'.format(
    'SentenceTransformer + O*NET titles', st_correct_top1/sample_size, st_correct_top3/sample_size, '{:.0f}s'.format(st_time)))
print('{:40s} {:>7.1%} {:>7.1%} {:>10s}'.format(
    'Keyword + Bonus Words', 0.239, 0.393, '~8min'))
