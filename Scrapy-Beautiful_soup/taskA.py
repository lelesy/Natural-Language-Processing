import scrapy

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://tengrinews.kz/']

    def parse(self, response):
        for title in response.css('div.news,div.clearAfter,div.pl,div.mb'):
            a = title.css('a::attr(href)').extract_first()
            if(a!=None):
                print a
