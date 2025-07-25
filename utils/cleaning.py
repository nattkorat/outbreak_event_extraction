import re
from bs4 import BeautifulSoup
import justext
import unicodedata

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
    
    @staticmethod
    def clean_noisy_content(text_content: str) -> str:
        """Clean content field in a dictionary by removing shortcodes and HTML tags."""
        # Remove shortcodes like [expander_maker ...] and [/expander_maker]
        text_content = re.sub(r'\[/?\w+.*?\]', '', text_content, flags=re.DOTALL)

        # Remove HTML tags like <p>, <div>, etc.
        text_content = re.sub(r'<[^>]+>', '', text_content, flags=re.DOTALL)

        # Remove excessive newlines or spaces
        text_content = re.sub(r'\n+', ' ', text_content)
        text_content = text_content.strip()

        return text_content

    @staticmethod
    def clean_text(text_content: str) -> str:
        """
        Cleans the provided plain text content by removing unnecessary symbols for thai.

        Args:
            text_content (str): The plain text content to be cleaned.

        Returns:
            str: The cleaned text content.
        """
        
        # Remove unwanted characters and symbols
        allowed_punct = {
            '.',
            ',',
            '!',
            '?',
            '-',
            '_',
            "'",
            '“',
            '”',
            '‘',
            '’',
            '…',
            'ๆ',
            'ฯ',
            '៕​',
            '​៖',
        }
        cleaned_text = []

        for char in text_content:
            cat = unicodedata.category(char)
            if (
                cat.startswith('L')   # Letter (Lu, Ll, Lt, Lm, Lo)
                or cat.startswith('M')  # Mark (Mn, Mc, Me): e.g., Thai tone/vowel marks
                or cat.startswith('N')  # Number
                or char.isspace()
                or char in allowed_punct
            ):
                cleaned_text.append(char)

        # Normalize whitespace
        return ' '.join(''.join(cleaned_text).split())

if __name__ == '__main__':
    
    # test clean text 
    text = "[โรคลึกลับระบาดเร็ว ป่วยยกหมู่บ้าน 500 คน อัฟกาฯ เร่งหาต้นตอโรค]"
    
    print("Original Text:", text)
    cleaned_text = TextCleaner.clean_text(text)
    print("Cleaned Text:", cleaned_text)