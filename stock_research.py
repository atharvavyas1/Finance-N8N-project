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
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_BASE_URL = "https://data.sec.gov"

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
    """Fetches and parses insider trading data from SEC EDGAR.

    Supports Form 3, 4, and 5 filings with detailed transaction parsing
    including derivative transactions, holdings, and transaction code mapping.
    """

    TRANSACTION_CODES = {
        'P': 'Open Market Purchase',
        'S': 'Open Market Sale',
        'A': 'Grant/Award',
        'D': 'Sale to Issuer',
        'F': 'Payment of Exercise Price or Tax Liability',
        'G': 'Gift',
        'M': 'Exercise of Options',
        'C': 'Conversion',
        'E': 'Expiration',
        'H': 'Held/Withheld',
        'I': 'Discretionary Transaction',
        'J': 'Other',
        'K': 'Equity Swap',
        'L': 'Small Acquisition',
        'U': 'Disposition to Issuer',
        'W': 'Acquisition or Disposition by Will',
        'X': 'Exercise of Out-of-the-Money Options',
        'Z': 'Deposit into or Withdrawal from Voting Trust'
    }

    def __init__(self):
        self.headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate"
        }
        self.data_headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }

    # ---- XML helpers --------------------------------------------------------

    @staticmethod
    def _get_text(element, tag_name: str) -> Optional[str]:
        """Safely get text from an XML element, handling nested <value> tags."""
        if element is None:
            return None
        found = element.find(f'.//{tag_name}')
        if found is None:
            return None
        value_elem = found.find('value')
        if value_elem is not None:
            return value_elem.text
        return found.text

    @staticmethod
    def _parse_address(owner_id) -> Dict:
        """Parse address information from owner ID element."""
        if owner_id is None:
            return {}
        address = owner_id.find('.//reportingOwnerAddress')
        if address is None:
            return {}

        def _txt(tag):
            el = address.find(f'.//{tag}')
            return el.text if el is not None else None

        return {
            'street1': _txt('rptOwnerStreet1'),
            'street2': _txt('rptOwnerStreet2'),
            'city': _txt('rptOwnerCity'),
            'state': _txt('rptOwnerState'),
            'zipCode': _txt('rptOwnerZipCode')
        }

    @classmethod
    def _parse_relationship(cls, relationship) -> Dict:
        """Parse the relationship of the reporting owner to the company."""
        if relationship is None:
            return {}
        return {
            'isDirector': cls._get_text(relationship, 'isDirector') == '1',
            'isOfficer': cls._get_text(relationship, 'isOfficer') == '1',
            'isTenPercentOwner': cls._get_text(relationship, 'isTenPercentOwner') == '1',
            'isOther': cls._get_text(relationship, 'isOther') == '1',
            'officerTitle': cls._get_text(relationship, 'officerTitle')
        }

    @staticmethod
    def _parse_transaction(trans_element, derivative: bool = False) -> Optional[Dict]:
        """Parse a single transaction element from Form 4 XML."""
        if trans_element is None:
            return None

        def _val(parent, tag):
            elem = parent.find(f'.//{tag}')
            if elem is None:
                return None
            value_elem = elem.find('value')
            return value_elem.text if value_elem is not None else elem.text

        transaction = {
            'securityTitle': _val(trans_element, 'securityTitle'),
            'transactionDate': _val(trans_element, 'transactionDate'),
            'deemedExecutionDate': _val(trans_element, 'deemedExecutionDate'),
            'transactionCode': _val(trans_element, 'transactionCode'),
            'equitySwapInvolved': _val(trans_element, 'equitySwapInvolved') == '1',
        }

        shares = _val(trans_element, 'transactionShares')
        price = _val(trans_element, 'transactionPricePerShare')

        transaction.update({
            'shares': float(shares) if shares else None,
            'pricePerShare': float(price) if price else None,
            'acquiredDisposed': _val(trans_element, 'transactionAcquiredDisposedCode'),
            'totalValue': None
        })

        if transaction['shares'] and transaction['pricePerShare']:
            transaction['totalValue'] = transaction['shares'] * transaction['pricePerShare']

        shares_owned = _val(trans_element, 'sharesOwnedFollowingTransaction')
        transaction['sharesOwnedAfter'] = float(shares_owned) if shares_owned else None
        transaction['directIndirect'] = _val(trans_element, 'directOrIndirectOwnership')

        return transaction

    @staticmethod
    def _parse_holdings(root) -> List[Dict]:
        """Parse current non-derivative holdings from Form 4 XML."""
        holdings = []

        def _val(parent, tag):
            elem = parent.find(f'.//{tag}')
            if elem is None:
                return None
            value_elem = elem.find('value')
            return value_elem.text if value_elem is not None else elem.text

        for holding in root.findall('.//nonDerivativeHolding'):
            holdings.append({
                'securityTitle': _val(holding, 'securityTitle'),
                'shares': _val(holding, 'sharesOwnedFollowingTransaction'),
                'directIndirect': _val(holding, 'directOrIndirectOwnership'),
                'natureOfOwnership': _val(holding, 'natureOfOwnership')
            })

        return holdings

    # ---- Form 4 parsing -----------------------------------------------------

    def _parse_form4(self, xml_content: str) -> Optional[Dict]:
        """Parse a Form 4 XML file and extract all relevant information.

        Returns a structured dict with issuer, reportingOwner, transactions,
        derivativeTransactions, and holdings.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return None

        result = {}

        # Issuer
        issuer = root.find('.//issuer')
        if issuer is not None:
            result['issuer'] = {
                'name': self._get_text(issuer, 'issuerName'),
                'cik': self._get_text(issuer, 'issuerCik'),
                'ticker': self._get_text(issuer, 'issuerTradingSymbol')
            }
        else:
            result['issuer'] = None

        # Reporting owner
        owner = root.find('.//reportingOwner')
        if owner is not None:
            owner_id = owner.find('.//reportingOwnerId')
            relationship = owner.find('.//reportingOwnerRelationship')
            result['reportingOwner'] = {
                'name': self._get_text(owner_id, 'rptOwnerName'),
                'cik': self._get_text(owner_id, 'rptOwnerCik'),
                'address': self._parse_address(owner_id),
                'relationship': self._parse_relationship(relationship)
            }
        else:
            result['reportingOwner'] = None

        # Non-derivative transactions
        result['transactions'] = []
        for trans in root.findall('.//nonDerivativeTransaction'):
            parsed = self._parse_transaction(trans, derivative=False)
            if parsed:
                result['transactions'].append(parsed)

        # Derivative transactions (options, warrants, etc.)
        result['derivativeTransactions'] = []
        for trans in root.findall('.//derivativeTransaction'):
            parsed = self._parse_transaction(trans, derivative=True)
            if parsed:
                result['derivativeTransactions'].append(parsed)

        # Current holdings
        result['holdings'] = self._parse_holdings(root)

        return result

    # ---- SEC API helpers ----------------------------------------------------

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Look up a company's CIK by ticker symbol."""
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
        """Get all submission history for a company."""
        _sec_rate_limit()
        cik = cik.zfill(10)
        url = f"{SEC_BASE_URL}/submissions/CIK{cik}.json"

        try:
            response = requests.get(url, headers=self.data_headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _filter_insider_filings(
        self,
        submissions_data: Dict,
        form_types: Optional[List[str]] = None,
        days_back: Optional[int] = None
    ) -> List[Dict]:
        """Filter submissions to insider trading forms (3, 4, 5) within a date range."""
        if form_types is None:
            form_types = ['3', '4', '5']

        recent = submissions_data.get('filings', {}).get('recent', {})

        cutoff_date = None
        if days_back:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        insider_filings = []
        forms = recent.get('form', [])
        num_forms = len(forms)

        for i, form in enumerate(forms):
            base_form = form.split('/')[0]
            if base_form not in form_types:
                continue

            filing_date = recent['filingDate'][i]
            if cutoff_date and filing_date < cutoff_date:
                continue

            if len(insider_filings) >= MAX_INSIDER_FILINGS:
                break

            insider_filings.append({
                'accessionNumber': recent['accessionNumber'][i],
                'filingDate': filing_date,
                'reportDate': recent.get('reportDate', [None] * num_forms)[i],
                'acceptanceDateTime': recent.get('acceptanceDateTime', [None] * num_forms)[i],
                'form': form,
                'primaryDocument': recent.get('primaryDocument', [None] * num_forms)[i],
                'description': recent.get('primaryDocDescription', [None] * num_forms)[i],
                'isAmendment': '/A' in form
            })

        return insider_filings

    def _fetch_form4_xml(self, cik: str, accession_number: str, primary_doc: str) -> str:
        """Download raw XML content of a Form 4 filing.

        Strips XSLT stylesheet paths (e.g. ``xslF345X05/``) so the SEC server
        returns raw XML instead of an HTML-rendered page.
        """
        _sec_rate_limit()
        cik_for_url = cik.lstrip('0') or '0'
        acc_no = accession_number.replace('-', '')

        if '/' in primary_doc:
            primary_doc = primary_doc.split('/')[-1]

        url = f"{EDGAR_ARCHIVES_BASE}/{cik_for_url}/{acc_no}/{primary_doc}"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.text

    # ---- Public API ---------------------------------------------------------

    @classmethod
    def get_transaction_description(cls, code: str) -> str:
        """Get human-readable description of a transaction code."""
        return cls.TRANSACTION_CODES.get(code, f'Unknown ({code})')

    def get_insider_trades(self, ticker: str, days: int = INSIDER_DAYS_BACK) -> Dict:
        """Get recent insider trades for a ticker with full parsed details.

        Returns a dict with ``trades`` (list of comprehensive Form 4 filings)
        and ``count``.
        """
        cik = self._get_cik(ticker)
        if not cik:
            return {'trades': [], 'error': f'CIK not found for {ticker}'}

        submissions = self._get_submissions(cik)
        if not submissions:
            return {'trades': [], 'error': 'Failed to fetch SEC submissions'}

        filings = self._filter_insider_filings(
            submissions, form_types=['4'], days_back=days
        )

        trades = []
        for filing in filings:
            try:
                xml_content = self._fetch_form4_xml(
                    cik,
                    filing['accessionNumber'],
                    filing['primaryDocument']
                )
                parsed = self._parse_form4(xml_content)
                if parsed:
                    parsed['filing_metadata'] = filing
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

