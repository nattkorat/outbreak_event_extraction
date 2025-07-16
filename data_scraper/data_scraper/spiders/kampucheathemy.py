import scrapy


class KampucheathemySpider(scrapy.Spider):
    name = "kampucheathemy"
    allowed_domains = ["www.kampucheathmey.com"]
    start_urls = ["https://www.kampucheathmey.com/sitemap_index.xml"]
    

    def parse(self, response):
        """Parser Response Object

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            post_stemap (Request): Request to the post sitemap pages.
        """
        response.selector.remove_namespaces()
        xmls = response.xpath('//loc/text()').getall()
        post_sitemaps = [xml for xml in xmls if 'post-sitemap' in xml]
        for xml in post_sitemaps:
            yield scrapy.Request(xml, callback=self.follow_post_sitemap)

    
    def follow_post_sitemap(self, response):
        """Follow post sitemap

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            articles (Request): Request to the article pages.
        """
        response.selector.remove_namespaces()
        xmls = response.xpath('//url/loc/text()').getall()
        health_xmls = [xml for xml in xmls if '/health/' in xml]
        for xml in health_xmls:
            yield scrapy.Request(xml, callback=self.parse_article)
    
    def parse_article(self, response):
        """Extracting article content

        Parameters:
            response: Response fed to the spider for processing.
        
        Yields:
            article_content (dict): Content data extracted from web page.
        """
        title = response.xpath('//h1/text()').get()
        post_date = response.css('meta[property="article:published_time"]::attr(content)').get()
        site_name = response.css('meta[property="og:site_name"]::attr(content)').get()
        texts = response.css('.mvp-post-main-out p ::text').getall()
        content = ' '.join(texts).strip()
        
        yield {
            "title": title,
            "date": post_date,
            "site_name": site_name,
            "source": response.url,
            "content": content
        }