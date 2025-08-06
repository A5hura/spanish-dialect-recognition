"""
This program analyzes the sentiment of a call transcript by:
- Splitting the transcript into chunks of sentences (default 3 sentences each)
- Scoring each chunk using VADER sentiment analysis to get compound sentiment scores
- Labeling each chunk as positive, neutral, or negative based on score thresholds
- Calculating:
    1-3) Percentage of chunks per sentiment category (positive, neutral, negative)
    4-6) Positional weighted sentiment scores for each sentiment category, 
         weighting chunks differently depending on their position in the transcript
    7) Overall weighted sentiment score across all chunks, weighted by chunk length (words)

The program is modular with separate functions for each step and returns a dictionary
of results, which can be printed or further processed.

Default chunk size is 3 sentences and positional weights default to [0.2, 0.3, 0.5]
for start, middle, and end segments respectively.
"""

import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

nltk.download('punkt')
nltk.download('vader_lexicon')

def chunk_sentences(text, chunk_size=3):
    """Split text into chunks of chunk_size sentences."""
    sentences = sent_tokenize(text)
    return [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]

def score_chunks(chunks, analyzer):
    """Return list of compound sentiment scores per chunk."""
    scores = []
    for chunk in chunks:
        combined_text = " ".join(chunk)
        score = analyzer.polarity_scores(combined_text)['compound']
        scores.append(score)
    return scores

def label_sentiment(score, pos_thresh=0.05, neg_thresh=-0.05):
    """Return sentiment label for a compound score."""
    if score > pos_thresh:
        return 'positive'
    elif score < neg_thresh:
        return 'negative'
    else:
        return 'neutral'

def label_chunks(chunk_scores):
    """Label each chunk's sentiment."""
    return [label_sentiment(score) for score in chunk_scores]

def get_chunk_lengths(chunks):
    """Return list of chunk lengths in words."""
    return [len(" ".join(chunk).split()) for chunk in chunks]

def calculate_sentiment_distribution(chunk_labels):
    """Calculate % of positive, neutral, negative chunks."""
    counts = defaultdict(int)
    total = len(chunk_labels)
    for label in chunk_labels:
        counts[label] += 1
    return {k: (v / total) * 100 for k, v in counts.items()}

def calculate_positional_weighted_scores(chunk_scores, chunk_labels, weights):
    """Calculate positional weighted scores per sentiment category."""
    num_chunks = len(chunk_scores)
    segment_size = max(1, num_chunks // len(weights))

    pos_weighted_scores = defaultdict(float)
    pos_weight_totals = defaultdict(float)

    for i, weight in enumerate(weights):
        start_idx = i * segment_size
        end_idx = min(start_idx + segment_size, num_chunks)
        for idx in range(start_idx, end_idx):
            label = chunk_labels[idx]
            score = chunk_scores[idx]
            pos_weighted_scores[label] += score * weight
            pos_weight_totals[label] += weight

    # Normalize scores
    for sentiment in pos_weighted_scores:
        if pos_weight_totals[sentiment] > 0:
            pos_weighted_scores[sentiment] /= pos_weight_totals[sentiment]
        else:
            pos_weighted_scores[sentiment] = 0.0

    # Ensure all sentiments exist in the dict
    for s in ['positive', 'neutral', 'negative']:
        pos_weighted_scores.setdefault(s, 0.0)

    return pos_weighted_scores

def calculate_weighted_sentiment_score(chunk_scores, chunk_lengths):
    """Calculate weighted sentiment score across all chunks by chunk length."""
    total_length = sum(chunk_lengths)
    if total_length == 0:
        return 0.0
    return sum(score * (length / total_length) for score, length in zip(chunk_scores, chunk_lengths))

def analyze_transcript(text, chunk_size=3, positional_weights=None):
    if positional_weights is None:
        positional_weights = [0.2, 0.3, 0.5]  # default weights for 3 segments

    analyzer = SentimentIntensityAnalyzer()

    chunks = chunk_sentences(text, chunk_size)
    chunk_scores = score_chunks(chunks, analyzer)
    chunk_labels = label_chunks(chunk_scores)
    chunk_lengths = get_chunk_lengths(chunks)

    sentiment_distribution = calculate_sentiment_distribution(chunk_labels)
    pos_weighted_scores = calculate_positional_weighted_scores(chunk_scores, chunk_labels, positional_weights)
    weighted_sentiment_score = calculate_weighted_sentiment_score(chunk_scores, chunk_lengths)

    # Prepare final results (with safe defaults)
    results = {
        'sentiment_positive_pct': sentiment_distribution.get('positive', 0.0),
        'sentiment_neutral_pct': sentiment_distribution.get('neutral', 0.0),
        'sentiment_negative_pct': sentiment_distribution.get('negative', 0.0),
        'pos_weighted_score_positive': pos_weighted_scores['positive'],
        'pos_weighted_score_neutral': pos_weighted_scores['neutral'],
        'pos_weighted_score_negative': pos_weighted_scores['negative'],
        'weighted_sentiment_score': weighted_sentiment_score,
    }

    return results

def print_results(results):
    print(f"1. Sentiment - Positive (% chunks): {results['sentiment_positive_pct']:.1f}%")
    print(f"2. Sentiment - Neutral (% chunks): {results['sentiment_neutral_pct']:.1f}%")
    print(f"3. Sentiment - Negative (% chunks): {results['sentiment_negative_pct']:.1f}%")
    print(f"4. Positional Weighted Score - Positive: {results['pos_weighted_score_positive']:.3f}")
    print(f"5. Positional Weighted Score - Neutral: {results['pos_weighted_score_neutral']:.3f}")
    print(f"6. Positional Weighted Score - Negative: {results['pos_weighted_score_negative']:.3f}")
    print(f"7. Weighted Sentiment Score: {results['weighted_sentiment_score']:.3f}")

# === Example usage ===
if __name__ == "__main__":
    sample_text = """
    Hello, thanks for calling. How can I help you today?
    I'm having an issue with my bill. It seems higher than expected.
    Let me check that for you. Could you provide your account number?
    Sure, it's 123456.
    Thanks, I see a late fee here. That might be why it's higher.
    Oh, I didn't know about that fee. Can it be waived?
    Let me see what I can do for you.
    """
    results = analyze_transcript(sample_text)
    print_results(results)
