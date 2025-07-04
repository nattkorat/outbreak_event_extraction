import scrapy


class VodSpider(scrapy.Spider):
    name = "vod"
    allowed_domains = ["www.vodkhmer.news"]
    start_urls = ["https://www.vodkhmer.news/category/national/social-issues/health/"]

    def parse(self, response):
        """Parser Response Object
        Parameters:
            response: Response fed to the spider for processing.
        Yields:
            articles (Request): Request to the article pages.
            next_page (Request): Request to the next page.
        """
        articles = response.css('.elementor-post__thumbnail__link::attr(href)').extract()
        for article in articles:
            yield response.follow(article, self.parse_article)
        
        next_page = response.css('.elementor-pagination .next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
    
    def parse_article(self, response):
        """
        Parameters:
            response: Response fed to the spider for processing.
        Yields:
            article_content (dict): Content data extracted from web page.
        """
        
        title = response.css('h1.elementor-heading-title::text').get()
        post_date = response.css('meta[property="article:published_time"]::attr(content)').get()
        site_name = response.css('meta[property="og:site_name"]::attr(content)').get()
        text = response.css('.elementor-widget-theme-post-content  p::text').getall()
        content = ' '.join(text)

        yield {
            "title": title,
            "date": post_date,
            "site_name": site_name,
            "source": response.url,
            "content": content
        }
