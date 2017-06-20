# !/usr/bin/env python
# -*- coding: utf-8 -*-
import scrapy

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://krisha.kz/arenda/kvartiry/kaskelen/?das[inet.type]=1']

    def parse(self, response):
        for title in response.css('section.a-list'):
            a = title.css('a.link::text').extract()
            for item in a:
                print item.encode("utf-8")
