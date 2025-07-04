import scrapy

from scrapy_playwright.page import PageMethod

class EacSpider(scrapy.Spider):
    name = "eac"
    allowed_domains = ["eacnews.asia"]
    start_urls = ["https://eacnews.asia/home/index/26"]
    
    def start_requests(self):
        """Initial Request to the Spider

        Yields:
            Request: Initial request to the start URL.
        """
        yield scrapy.Request(
            url=self.start_urls[0],
            meta={
                'playwright': True,
                "playwright_page_coroutines": [
                   PageMethod("wait_for_timeout", 10000)  # Wait 10 seconds
                ]},
            callback=self.parse)

    def parse(self, response):
        """Parser Response Object

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            articles (Request): Request to the article pages.
            next_page (Request): Request to the next page.
        """
        articles = response.css('.entry__title a::attr(href)').getall()
        print(f"Found {len(articles)} articles on page.")
        
        # for article in articles:
        #     yield response.follow(article, self.parse_article)

        # next_page = response.css('.pagination .next::attr(href)').get()
        # if next_page:
        #     yield response.follow(next_page, callback=self.parse)
    
    def parse_article(self, response):
        """Extracting article content

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            article_content (dict): Content data extracted from web page.
        """
        title = response.css('h1.single-post__entry-title::text').get().strip()
        try:
            post_date = response.css('.entry__meta li.entry__meta-date::text').getall()[1]
        except IndexError:
            post_date = ' '.join(response.css('meta[property="article:published_time"]::attr(content)').getall())

        site_name = response.css('meta[property="og:site_name"]::attr(content)').extract()[-1].strip()
        text = response.css('.entry__article p ::text').getall()
        content = ' '.join(text)

        yield {
            "title": title,
            "date": post_date,
            "site_name": site_name,
            "source": response.url,
            "content": content
        }
