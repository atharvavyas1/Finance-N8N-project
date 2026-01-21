"""
Stock Research Script for N8N Code Node
========================================
Fetches news articles, performs sentiment analysis (FinBERT), 
extracts topics (LDA), and retrieves insider trading data (SEC EDGAR).

Input: Ticker symbol from previous node ($json.output)
Output: Structured JSON for summarization agent
"""

import re
import time
import string
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional

import feedparser
import nltk
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertForSequenceClassification, BertTokenizer


# =============================================================================
# CONFIGURATION
# =============================================================================

# SEC EDGAR requires identification
SEC_USER_AGENT = "StockResearchBot contact@example.com"  # UPDATE THIS

# Article scraping settings
MAX_ARTICLES = 5
MIN_WORD_COUNT = 150

# LDA settings
LDA_N_TOPICS = 3
LDA_MAX_ITER = 50
LDA_N_WORDS = 8

# SEC EDGAR settings
INSIDER_DAYS_BACK = 30
MAX_INSIDER_FILINGS = 5


# =============================================================================
# NLTK DATA DOWNLOAD (run once)
# =============================================================================

def ensure_nltk_data():
    """Download required NLTK data if not present."""
    datasets = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if 'punkt' in dataset else f'corpora/{dataset}')
        except LookupError:
            nltk.download(dataset, quiet=True)


# =============================================================================
# ARTICLE SCRAPING (Yahoo Finance RSS)
# =============================================================================

class ArticleScraper:
    """Fetches full article content from Yahoo Finance RSS feeds."""

    def __init__(self):
        self.base_rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={}&region=US&lang=en-US"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    def _extract_article_text(self, url: str) -> Optional[str]:
        """Extract full article text from a Yahoo Finance article URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            selectors = [
                'article',
                '[data-module="ArticleBody"]',
                '.caas-body',
                '.article-body',
                '[class*="article"]',
                '[class*="content"]'
            ]

            article_content = None
            for selector in selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    break

            if not article_content:
                article_content = soup.find('main') or soup.find('article')

            if article_content:
                for tag in article_content(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = article_content.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text).strip()

            return None
        except Exception:
            return None

    def get_articles(self, ticker: str, max_articles: int = MAX_ARTICLES) -> List[Dict]:
        """Fetch articles for a ticker symbol."""
        articles = []

        try:
            feed_url = self.base_rss_url.format(ticker.upper())
            feed = feedparser.parse(feed_url)

            if not feed.entries:
                return articles

            for entry in feed.entries:
                if len(articles) >= max_articles:
                    break

                article_url = entry.get('link', '').strip()
                title = entry.get('title', '').strip()

                if not article_url:
                    continue

                full_text = self._extract_article_text(article_url)
                word_count = len(full_text.split()) if full_text else 0

                if word_count <= MIN_WORD_COUNT:
                    continue

                published = entry.get('published', '')
                published_dt = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published_dt = datetime(*entry.published_parsed[:6]).isoformat()
                    except Exception:
                        pass

                articles.append({
                    'title': title,
                    'link': article_url,
                    'published': published_dt or published,
                    'full_text': full_text,
                    'word_count': word_count
                })

            return articles

        except Exception:
            return articles


# =============================================================================
# SENTIMENT ANALYSIS (FinBERT)
# =============================================================================

class SentimentAnalyzer:
    """FinBERT-based sentiment analysis for financial text."""

    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Lazy load the model."""
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()

    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of a single text."""
        self._load_model()

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        predicted_class = probabilities.argmax().item()
        sentiment_label = self.model.config.id2label[predicted_class]

        return {
            'label': sentiment_label,
            'confidence': float(probabilities[predicted_class]),
            'probabilities': {
                'positive': float(probabilities[0]),
                'negative': float(probabilities[1]),
                'neutral': float(probabilities[2])
            }
        }

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment across multiple articles."""
        if not articles:
            return {'overall': 'neutral', 'details': []}

        sentiments = []
        for article in articles:
            text = article.get('full_text', '')
            if text:
                result = self.analyze(text)
                sentiments.append({
                    'title': article.get('title', ''),
                    'sentiment': result['label'],
                    'confidence': result['confidence']
                })

        # Calculate overall sentiment
        if sentiments:
            labels = [s['sentiment'] for s in sentiments]
            overall = max(set(labels), key=labels.count)
        else:
            overall = 'neutral'

        return {
            'overall': overall,
            'article_count': len(sentiments),
            'details': sentiments
        }


# =============================================================================
# TOPIC EXTRACTION (LDA)
# =============================================================================

class TopicExtractor:
    """LDA-based topic extraction for stock news articles."""

    def __init__(self, n_topics: int = LDA_N_TOPICS):
        self.n_topics = n_topics
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update({
            'said', 'say', 'says', 'company', 'companies', 'stock', 'stocks',
            'share', 'shares', 'market', 'markets', 'new', 'also', 'would',
            'could', 'may', 'might', 'one', 'two', 'first', 'last', 'year',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        })

    def _preprocess(self, text: str) -> str:
        """Clean and preprocess text."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        tokens = word_tokenize(text)
        cleaned = []
        for token in tokens:
            if len(token) >= 3 and token not in self.stop_words and token not in string.punctuation:
                lemma = self.lemmatizer.lemmatize(token, pos='v')
                lemma = self.lemmatizer.lemmatize(lemma, pos='n')
                cleaned.append(lemma)

        return ' '.join(cleaned)

    def extract_topics(self, articles: List[Dict]) -> Dict:
        """Extract topics from articles."""
        if len(articles) < 2:
            return {'topics': [], 'error': 'Need at least 2 articles for topic extraction'}

        texts = [self._preprocess(a.get('full_text', '')) for a in articles]
        texts = [t for t in texts if t.strip()]

        if len(texts) < 2:
            return {'topics': [], 'error': 'Insufficient text after preprocessing'}

        try:
            vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.95, ngram_range=(1, 2))
            doc_term_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            lda = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=LDA_MAX_ITER,
                random_state=42,
                learning_method='batch'
            )
            lda.fit(doc_term_matrix)

            topics = []
            for idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-LDA_N_WORDS:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append({
                    'topic_id': idx + 1,
                    'keywords': top_words
                })

            return {'topics': topics}

        except Exception as e:
            return {'topics': [], 'error': str(e)}


# =============================================================================
# SEC EDGAR INSIDER TRADING
# =============================================================================

_last_sec_request = 0

def _sec_rate_limit():
    """Ensure we don't exceed SEC's rate limit (10 req/sec)."""
    global _last_sec_request
    elapsed = time.time() - _last_sec_request
    if elapsed < 0.11:
        time.sleep(0.11 - elapsed)
    _last_sec_request = time.time()


class InsiderTradingTracker:
    """Fetches insider trading data from SEC EDGAR."""

    def __init__(self):
        self.headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate"
        }

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Look up CIK by ticker symbol."""
        _sec_rate_limit()
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            ticker_upper = ticker.upper()
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker_upper:
                    return str(entry['cik_str']).zfill(10)
            return None
        except Exception:
            return None

    def _get_submissions(self, cik: str) -> Optional[Dict]:
        """Get company submissions from SEC."""
        _sec_rate_limit()
        headers = {**self.headers, "Host": "data.sec.gov"}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _parse_form4(self, xml_content: str) -> Optional[Dict]:
        """Parse Form 4 XML content."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return None

        def get_text(elem, tag):
            if elem is None:
                return None
            found = elem.find(f'.//{tag}')
            if found is None:
                return None
            value = found.find('value')
            return value.text if value is not None else found.text

        owner = root.find('.//reportingOwner')
        owner_id = owner.find('.//reportingOwnerId') if owner else None
        relationship = owner.find('.//reportingOwnerRelationship') if owner else None

        transactions = []
        for trans in root.findall('.//nonDerivativeTransaction'):
            shares = get_text(trans, 'transactionShares')
            price = get_text(trans, 'transactionPricePerShare')
            
            transactions.append({
                'date': get_text(trans, 'transactionDate'),
                'shares': float(shares) if shares else None,
                'price': float(price) if price else None,
                'type': get_text(trans, 'transactionCode'),
                'acquired': get_text(trans, 'transactionAcquiredDisposedCode') == 'A'
            })

        roles = []
        if relationship:
            if get_text(relationship, 'isDirector') == '1':
                roles.append('Director')
            if get_text(relationship, 'isOfficer') == '1':
                title = get_text(relationship, 'officerTitle') or 'Officer'
                roles.append(title)
            if get_text(relationship, 'isTenPercentOwner') == '1':
                roles.append('10% Owner')

        return {
            'insider_name': get_text(owner_id, 'rptOwnerName') if owner_id else None,
            'roles': roles,
            'transactions': transactions
        }

    def get_insider_trades(self, ticker: str, days: int = INSIDER_DAYS_BACK) -> Dict:
        """Get recent insider trades for a ticker."""
        cik = self._get_cik(ticker)
        if not cik:
            return {'trades': [], 'error': f'CIK not found for {ticker}'}

        submissions = self._get_submissions(cik)
        if not submissions:
            return {'trades': [], 'error': 'Failed to fetch SEC submissions'}

        recent = submissions.get('filings', {}).get('recent', {})
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        filings = []
        for i, form in enumerate(recent.get('form', [])):
            if form.split('/')[0] != '4':
                continue
            filing_date = recent['filingDate'][i]
            if filing_date < cutoff:
                continue
            if len(filings) >= MAX_INSIDER_FILINGS:
                break

            filings.append({
                'accession': recent['accessionNumber'][i],
                'date': filing_date,
                'doc': recent.get('primaryDocument', [None] * len(recent['form']))[i]
            })

        trades = []
        for filing in filings:
            try:
                _sec_rate_limit()
                cik_url = cik.lstrip('0') or '0'
                acc_no = filing['accession'].replace('-', '')
                doc = filing['doc'].split('/')[-1] if '/' in filing['doc'] else filing['doc']
                
                url = f"https://www.sec.gov/Archives/edgar/data/{cik_url}/{acc_no}/{doc}"
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()

                parsed = self._parse_form4(response.text)
                if parsed:
                    parsed['filing_date'] = filing['date']
                    trades.append(parsed)
            except Exception:
                continue

        return {'trades': trades, 'count': len(trades)}


# =============================================================================
# MAIN FUNCTION FOR N8N
# =============================================================================

def run_stock_research(ticker: str) -> Dict:
    """
    Main entry point for N8N code node.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        Structured dictionary with all research data
    """
    ticker = ticker.strip().upper()
    
    if not ticker or ticker == 'INVALID':
        return {
            'ticker': ticker,
            'error': 'Invalid ticker symbol',
            'articles': [],
            'sentiment': {'overall': 'unknown'},
            'topics': {'topics': []},
            'insider_trades': {'trades': []}
        }

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Initialize components
    scraper = ArticleScraper()
    sentiment_analyzer = SentimentAnalyzer()
    topic_extractor = TopicExtractor()
    insider_tracker = InsiderTradingTracker()

    # Fetch articles
    articles = scraper.get_articles(ticker)
    
    if not articles:
        return {
            'ticker': ticker,
            'error': 'No articles found',
            'articles': [],
            'sentiment': {'overall': 'unknown'},
            'topics': {'topics': []},
            'insider_trades': insider_tracker.get_insider_trades(ticker)
        }

    # Analyze sentiment
    sentiment = sentiment_analyzer.analyze_articles(articles)

    # Extract topics
    topics = topic_extractor.extract_topics(articles)

    # Get insider trades
    insider_trades = insider_tracker.get_insider_trades(ticker)

    # Prepare article summaries (without full text to reduce payload)
    article_summaries = [
        {
            'title': a['title'],
            'published': a['published'],
            'link': a['link'],
            'word_count': a['word_count']
        }
        for a in articles
    ]

    return {
        'ticker': ticker,
        'articles': article_summaries,
        'sentiment': sentiment,
        'topics': topics,
        'insider_trades': insider_trades
    }


# =============================================================================
# N8N CODE NODE ENTRY POINT
# =============================================================================

def n8n_main(_input):
    """
    Entry point for N8N Python code node.
    
    Usage in N8N Code Node:
    -----------------------
    Copy this entire file into the code node, then at the bottom add:
    
        return n8n_main(_input)
    
    Or use the standalone version below.
    """
    ticker = _input.all()[0].json.get('output', '').strip()
    result = run_stock_research(ticker)
    return [{"json": result}]


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    # Test with a sample ticker
    import json
    
    test_ticker = "AAPL"
    print(f"Running stock research for {test_ticker}...")
    
    result = run_stock_research(test_ticker)
    print(json.dumps(result, indent=2, default=str))

