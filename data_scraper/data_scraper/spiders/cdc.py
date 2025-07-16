import scrapy


class CdcSpider(scrapy.Spider):
    name = "cdc"
    allowed_domains = ["www.cdcmoh.gov.kh"]
    start_urls = ["http://www.cdcmoh.gov.kh/"]

    def parse(self, response):
        """""Parser Response Object

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            articles (Request): Request to the article pages.
            next_page (Request): Request to the next page.
        """
        articles = response.css('.column-1 h2 a::attr(href)').getall()
        for article in articles:
            yield response.follow(article, self.parse_article)

        next_page = response.css('a[title="Next"]::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
    
    def parse_article(self, response):
        """Extracting article content

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            article_content (dict): Content data extracted from web page.
        """
        title = response.css('head title::text').get()
        data = response.css('.item-page p * ::text').getall()
        content = ' '.join(data).strip()

        yield {
            "title": title,
            "date": ''.join(data[:3]).strip(), # a bad way to get date
            "site_name": 'CDCMOH',
            "source": response.url,
            "content": content
        }
