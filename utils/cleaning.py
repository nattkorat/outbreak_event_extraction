import re
from bs4 import BeautifulSoup
import justext

class HtmlCleaner:
    """
    A class to clean HTML content by removing tags, Markdown, and unnecessary whitespace.
    """

    def __init__(self):
        pass

    @staticmethod
    def clean_html(html_content: str) -> str:
        """
        Cleans HTML and Markdown from the content.

        Args:
            html_content (str): Raw HTML + Markdown content.

        Returns:
            str: Cleaned plain text.
        """
        # 1. Remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')

        # Optional: remove script and style
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # paragraphs = soup.find_all('p')
        
        # text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
        text = soup.get_text(separator=' ', strip=True)
        
        text = ' '.join(text.split())

        # 2. Remove Markdown syntax
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove images ![alt](url)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)   # Remove links [text](url)
        text = re.sub(r'[#*`_>{}=+-]+', '', text)    # Remove Markdown symbols
        text = re.sub(r'\s{2,}', ' ', text)          # Collapse extra spaces again

        return text.strip()
    
    @staticmethod
    def clean_html_with_justext(html_content: str) -> str:
        """
        Cleans HTML content using justext library to extract meaningful text.

        Args:
            html_content (str): Raw HTML content.

        Returns:
            str: Cleaned plain text.
        """
        paragraphs = justext.justext(html_content, justext.get_stoplist('English'))
        text = ' '.join([p.text for p in paragraphs])
        
        # Normalize whitespace
        return ' '.join(text.split())


class MarkdownCleaner():
    """
    A class to clean Markdown content by removing unnecessary whitespace.
    """

    def __init__(self):
        pass

    @staticmethod
    def clean_markdown(markdown_content: str) -> str:
        """
        Cleans the provided Markdown content by removing extra whitespace.

        Args:
            markdown_content (str): The Markdown content to be cleaned.

        Returns:
            str: The cleaned text content.
        """
        
        # Extract only content from markdown
        markdown_content = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_content)
        markdown_content = re.sub(r'\[.*?\]\(.*?\)', '', markdown_content)  # Remove links
        markdown_content = re.sub(r'[#*`_>{}=+-]+', '', markdown_content)  # Remove Markdown symbols
        markdown_content = re.sub(r'\s{2,}', ' ', markdown_content)  # Collapse extra spaces
        markdown_content = markdown_content.strip()
        
        # Normalize whitespace
        text = markdown_content.replace('\n', ' ').replace('\r', ' ')
        
        return ' '.join(text.split())

class TextCleaner():
    """
    A class to clean plain text content by removing unnecessary special characters and symbol.
    """

    def __init__(self):
        pass

    @staticmethod
    def clean_text(text_content: str) -> str:
        """
        Cleans the provided plain text content by removing unnecessary symbols.

        Args:
            text_content (str): The plain text content to be cleaned.

        Returns:
            str: The cleaned text content.
        """
        
        # Remove unwanted characters and symbols
        cleaned_text = ''.join(
            char for char in text_content
            if char.isalnum() or char.isspace() or char in ['.', ',', '!', '?', '-', '_', "'"]
        )
        # Normalize whitespace
        return ' '.join(cleaned_text.split())
