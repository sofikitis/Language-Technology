import scrapy
import os.path


save_path = 'C:/Users/Vagelas/PycharmProjects/lang_tech/A/html_pages'


class NewsCrawlerSpider(scrapy.Spider):
    name = 'news_crawler'

    custom_settings = {
        'CLOSESPIDER_PAGECOUNT': 2000
    }

    allowed_domains = [#'https://edition.cnn.com/', 'https://www.npr.org/', 'https://www.bbc.com/',
                       #'https://www.theguardian.com/', 'https://abcnews.go.com',
                       #'https://www.bbc.com', 'https://snowboardaddiction', 'https://whitelines.com',
                       #'www.aljazeera.com/'
                       ]

    start_urls = ['https://edition.cnn.com/2020/09/16/africa/amnesty-mozambique-video-killing-investigation-intl'
                  '/index.html ',
                  'https://www.npr.org/2020/09/18/100306972/justice-ruth-bader-ginsburg-champion-of-gender-equality'
                  '-dies-at-87',
                  'https://www.bbc.com/news/world-europe-54211361',
                  'https://www.theguardian.com/world/2020/sep/19/thousands-gather-in-thailand-for-anti-government'
                  '-protest-bangkok.html',
                  'https://abcnews.go.com/Politics/supreme-court-justice-ruth-bader-ginsburg-dies-87/story?id=27200334',
                  'https://www.aljazeera.com/indepth/features/uae-american-drones-china-ramps-sales-200919143746852'
                  '.html',
                  'https://whitelines.com/snowboard-gear/news-previews/jones-2020-2021-snowboard-product-preview.html']

    def parse(self, response):
        filename = response.url.split("/")[-1] + '.html'
        complete_name = os.path.join(save_path, filename)

        with open(complete_name, 'wb') as f:
            f.write(response.body)

        a_selectors = response.css("a").xpath("@href").extract()

        # Loop on each tag
        for selector in a_selectors:

            # Create a new Request object
            request = response.follow(selector, callback=self.parse)
            # Return it thanks to a generator
            yield request
