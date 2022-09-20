import re
from collections import Counter
from functools import reduce

import fitz
from nltk.corpus import stopwords
from nltk import word_tokenize
import pymorphy2


def main():
    pdf_path = 'file.pdf'

    text = read_pdf(pdf_path)
    with open('file.txt', 'w', encoding='utf8') as f:
        f.write(text)
    
    articles = split_text_into_articles(text)
    articles = articles_preprocessing(articles)
    lemmatized_articles = lemmatize_articles(articles)

    tf_idfs = calculate_tf_idf(lemmatized_articles)
    result = format_tf_idf(tf_idfs, limit=20)
    with open('tf_idf.txt', 'w', encoding='utf8') as f:
        f.write(result)


def read_pdf(path: str) -> str:
    with fitz.open(path) as document:
        text = ''
        for page in document:
            text += page.get_text()
    return text


def split_text_into_articles(text: str) -> list[str]:
    """First element (non article start of the document) is ignored."""
    splitted_text = re.split(r'УДК \d\d\d\.', text)
    articles = []
    for article in splitted_text[1:]:
        article = re.split(r'СПИСОК ЛИТЕРАТУРЫ', article)[0]
        articles.append(article)
    return articles


def articles_preprocessing(articles: list[str]) -> list[list[str]]:
    """Returns list of words for each article."""
    lowercase_articles = lower_articles(articles)
    tokenized_articles = tokenize_articles(lowercase_articles)
    articles = remove_junk_from_articles(tokenized_articles)
    return articles


def lower_articles(articles: list[str]) -> list[str]:
    lowercase_articles = []
    for article in articles:
        lowercase_articles.append(article.lower())
    return lowercase_articles


def tokenize_articles(articles: list[str]) -> list[list[str]]:
    tokenized_articles = []
    for article in articles:
        tokenized_articles.append(word_tokenize(article))
    return tokenized_articles


def remove_junk_from_articles(articles: list[list[str]]) -> list[list[str]]:
    """articles must be tokenized.
    Junk is stop words and words without alphabetical characters.
    """
    filtered_articles = []
    for article in articles:
        article = remove_stop_words(article)
        article = remove_non_alphabetical_words(article)
        filtered_articles.append(article)
    return filtered_articles


def remove_stop_words(words: list[str]) -> list[str]:
    filtered_words = []
    for word in words:
        if word not in stopwords.words('russian'):
            filtered_words.append(word)
    return filtered_words


def remove_non_alphabetical_words(words: list[str]) -> list[str]:
    """Remove all words containing non-alphabetical (excl. hyphen) characters."""
    filtered_words = []
    for word in words:
        if re.search(r'[a-zA-Zа-яА-Я-]', word) and not re.search(r'[^a-zA-Zа-яА-Я-]', word):
            filtered_words.append(word)
    return filtered_words


def lemmatize_articles(articles: list[list[str]]) -> list[list[str]]:
    """Returns list of words for each article."""
    morph = pymorphy2.MorphAnalyzer()
    lemmatized_articles = []
    for article in articles:
        article_words = []
        for word in article:
            article_words.append(morph.parse(word)[0].normal_form)
        lemmatized_articles.append(article_words)
    return lemmatized_articles


def calculate_tf_idf(lemmatized_articles: list[str]) -> list[dict[str, float]]:
    """Calculate term frequency for each word in all articles by tf–idf metric.
    Returns list of dictionaries of "term: frequency" in article.
    """
    tf_idfs = []
    number_of_articles_with_word = count_number_of_articles_with_all_words(lemmatized_articles)
    number_of_articles = len(lemmatized_articles)
    for article in lemmatized_articles:
        word_frequency = Counter(article)
        number_of_words = len(article)
        article_tf_idfs = {}
        for word in word_frequency:
            tf = word_frequency[word] / number_of_words
            idf = number_of_articles / number_of_articles_with_word[word]
            article_tf_idfs[word] = tf * idf
        tf_idfs.append(article_tf_idfs)
    return sort_tf_idfs(tf_idfs)


def sort_tf_idfs(tf_idfs: list[dict]) -> list[dict[str, float]]:
    """Sort dictionaries by term frequency (descending)."""
    sorted_tf_idfs = []
    for article in tf_idfs:
        sorted_article = dict(sorted(article.items(), key=lambda x: -x[1]))
        sorted_tf_idfs.append(sorted_article)
    return sorted_tf_idfs


def count_number_of_articles_with_all_words(articles: list[list[str]]) -> dict[str, int]:
    """Returns dictionary number of occurrences of terms in articles."""
    word_frequency_in_all_articles = Counter(reduce(lambda x, y: x + y, articles))
    number_of_articles_with_word = {}
    for word in word_frequency_in_all_articles:
        for article in articles:
            if word in article:
                if word in number_of_articles_with_word:
                    number_of_articles_with_word[word] += 1
                else:
                    number_of_articles_with_word[word] = 1
    return number_of_articles_with_word


def format_tf_idf(tf_idfs: list[dict[str, float]], limit: int = 20) -> str:
    """Returns most important terms for each article as string.
    limit — number of terms (default 20) to print.
    """
    result = ''
    for index, article in enumerate(tf_idfs):
        result += f'\nArticle {index + 1}\n'
        terms = list(article.items())
        article_limit = len(terms) if len(terms) < limit else limit
        for word, tf_idf in terms[:article_limit]:
            result += f'{word}: {round(tf_idf, 5)}\n'
    return result


if __name__ == '__main__':
    main()
