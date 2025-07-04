import scrapy


class DapSpider(scrapy.Spider):
    name = "dap"
    allowed_domains = ["dap-news.com"]
    start_urls = ["https://dap-news.com/category/covid19/"]

    def parse(self, response):
        """Parser Response Object

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            articles (Request): Request to the article pages.
            next_page (Request): Request to the next page.
        """
        articles = response.css('li[class=infinite-post] a::attr(href)').getall()
        for article in articles:
            yield response.follow(article, self.parse_article)

        next_page = response.xpath('//a[text()="Next ›"]/@href').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_article(self, response):
        """Extracting article content

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            article_content (dict): Content data extracted from web page.
        """
        title = response.css('h1.post-title.entry-title.left::text').get()
        post_date = response.css('meta[property="article:published_time"]::attr(content)').get()
        site_name = response.css('meta[property="og:site_name"]::attr(content)').get()
        text = response.css('#content-main p *::text').getall()
        content = ' '.join(text)

        yield {
            "title": title,
            "date": post_date,
            "site_name": site_name,
            "source": response.url,
            "content": content
        }
