# !/usr/bin/env python
# -*- coding: utf-8 -*-
import scrapy

file = open("output-2.txt","w")
class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['http://krisha.kz/arenda/kvartiry/kaskelen/?das[price][to]=90000']

    def parse(self, response):
        for title in response.css('section.a-list'):
            a = title.css('a.link::text').extract()
            for item in a:
                file.write(item.encode("utf-8")+'\n')
                print item.encode("utf-8")
            next_page = response.xpath('.//a[@class="btn paginator-page-btn"]/@href').extract()
            if next_page:
                next_href = next_page[0]
                next_page_url = 'http://krisha.kz/' + next_href
                request = scrapy.Request(url=next_page_url)
                yield request
