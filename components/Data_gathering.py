from datetime import datetime
import newspaper
from newspaper import news_pool
import wikipediaapi
import pandas as pd
from tqdm import tqdm
import requests
import time
import gsch
from gsch.agent import Agent
from gsch.paper import Paper
from gsch.option import Option
from fake_useragent import UserAgent


def news_downloader(lists, save_csv=True):
    """
    Download the contents of a newspaper. that is passed as a list of newspaper urls.
    The contents of the papers such as 'title', 'summary', or 'authors' are downloaded from the url with the 'newspaper3k' library.

    :param lists: newspaper urls
    :type lists: list
    :param save_csv: Default: True. If True results are in csv format, if False as pandas dataframe.
    :type save_csv: bool
    :return: news content
    :rtype: csv fprmat or pandas dataframe 
    """
    merged_url_data = []
    news_build = []
    for url in lists:
        news_build.append(newspaper.build(url, memoize_articles=False))
    news_pool.set(news_build)
    news_pool.join()

    for source in news_build:
        for art in source.articles:
            art.download()
            art.parse()
            art.nlp()

            url_text = art.text
            url_summary = art.summary
            url_author = "|".join(art.authors)
            url_top_image = art.top_image
            url_publish_date = art.publish_date
            url_keywords = "|".join(art.keywords)

            filename = datetime.now().strftime("news_%m-%d-%y_%H_%M.csv")

            url_data = {
                "Timestamp": [filename],
                "Date Published": [url_publish_date],
                "Source": [source.brand],
                "Summary": [url_summary],
                "Authors": [url_author],
                "Text": [url_text],
                "Keywords": [url_keywords],
                "Top Image": [url_top_image],
            }
            df = pd.DataFrame(
                url_data,
                columns=[
                    "Timestamp",
                    "Source",
                    "Summary",
                    "Authors",
                    "Text",
                    "Keywords",
                    "Top Image",
                ],
            )
            merged_url_data.append(df)

    print("Created News Downloader:", filename)
    merged_url_data = pd.concat(merged_url_data, axis=0)
    if save_csv:
        merged_url_data.to_csv(filename, index=False)
    return merged_url_data


def wiki(topic, language="en"):
    """
    Downloads Wikipedia page about topic.

    :param topic: Topic of Wikipedia page
    :type topic: string
    :param language: Default: 'en' English
    :type language: string
    :return: Content of page
    :rtype: string
    """

    wiki_lang = wikipediaapi.Wikipedia(language)
    page = wiki_lang.page(topic)
    return page.text


def scholar(keywordlist, yearlow, yearhigh, pages=1):
    """
    Download scientific papers from Google Scholar.

    :param keywordlist: keywords to be tracked
    :type keywordlist: list
    :param yearlow: from year
    :type yearlow: int    
    :param yearhigh: to year
    :type yearhigh: int
    :param pages: number of pages of results to scrape (each page has maximum 10 results). Default: 1
    :type pages: int
    :return: content of paper
    :rtype: pandas dataframe
    """

    def pages_to_start(pages):
        starts = [0]
        for i in range(1, pages):
            starts.append(starts[-1] + 10)
        return starts

    starts_list = pages_to_start(pages)
    dataframes = []

    for start in starts_list:
        yearlow = str(yearlow)
        yearhigh = str(yearhigh)
        agent = Agent()
        option = Option(year_low=yearlow, year_high=yearhigh, start=start)
        papers = agent.search(keywordlist, option)
        collections = []

        for paper in papers:
            row = [
                paper.title,
                paper.authors,
                paper.year,
                paper.url,
                paper.cited_by,
                paper.snippets,
            ]
            collections.append(row)

        df = pd.DataFrame(
            collections,
            columns=["title", "authors", "year", "url", "cited_by", "snippets"],
        )

        dataframes.append(df)

    return pd.concat(dataframes).reset_index().drop(["index"], axis=1)


class getPapers:
    def __init__(self, topic, client="155.246.39.34", driver="phantomjs"):

        """
        This class will extract all the papers that are present in the arxiv website given a keyword;
        You have the option to insert all the information into a MongoDB;
        The class has two methods:
        1. extract_links: This method will generate a list of all the links to get information from given a keyword/topic
        2. extract_information_from_links: After we have the list, we will extract all the information from the links

        :param topic: Keyword to extract papers for; Note: This could be anything even author names.
        :param client: This is the IP to connect to the MongoDB, default: '155.246.39.34'
        :param driver: Options: ['phantomjs', 'chrome']. Make sure you have them in the path and/or directory

        """

        self.t0 = time.time()
        self.records = []
        self.topic = topic
        self.client = client
        self.url = "https://arxiv.org/search/"

        #! Use a fake-useragent
        ua = UserAgent()
        self.curr_ua = ua.random

        #! Try to use the Fake-UserAgent when possible
        if driver == "phantomjs":
            try:
                caps = DesiredCapabilities.PHANTOMJS
                caps["phantomjs.page.settings.userAgent"] = self.curr_ua
                self.driver = webdriver.PhantomJS(
                    "phantomjs", desired_capabilities=caps
                )
            except:
                self.driver = webdriver.PhantomJS("phantomjs")

        elif driver == "chrome":
            try:
                opts = Options()
                opts.add_argument("user-agent={}".format(self.curr_ua))
                self.driver = webdriver.Chrome("chromedriver", chrome_options=opts)
            except:
                self.driver = webdriver.Chrome("chromedriver")
        else:
            print("Not Supported!")

        self.client = pymongo.MongoClient(client)

        # ! Create DB
        self.db = self.client["Papers"]

        # ! Create Collection
        self.col = self.db["arxiv"]

    def extract_links(self):
        """
        This method will generate a list of all the links to get information from given a keyword/topic

        """

        # ! For each topic/tech that is passed: DO;
        print("Tech:", self.topic)

        # ! Start browser and open the link
        self.driver.get(self.url)

        # ! Make a search for the topic
        search = self.driver.find_element_by_xpath('//*[@id="query"]')
        search.send_keys(self.topic.lower())
        self.driver.find_element_by_xpath(
            "/html/body/main/div[2]/form/div[1]/div[3]/button"
        ).click()

        # ! Add a 2 second sleep
        time.sleep(2)
        ittertabs = self.driver.find_elements_by_class_name("arxiv-result")
        pages = self.driver.find_elements_by_xpath("/html/body/main/div[2]/nav[1]/ul")
        curr_url = self.driver.current_url

        page = 0
        load_next = True
        self.master_list_of_links = []

        # ! Until we have completed all the pages extract links in each page
        while load_next:
            self.driver.get(curr_url + "&start=" + str(page))
            try:
                if self.driver.find_element_by_xpath("/html/body/main/div[2]/p"):
                    # print("Found the break")
                    load_next = False
            except Exception as e:
                pass

            page += 50

            ittertabs = self.driver.find_elements_by_class_name("arxiv-result")

            for i in ittertabs:
                self.master_list_of_links.append(
                    i.find_element_by_css_selector("div > p > a").get_attribute("href")
                )

            time.sleep(2.5)

        # ! Keep only the unique links
        self.master_list_of_links = list(set(self.master_list_of_links))
        print("Total papers found:", len(self.master_list_of_links))

    def extract_information_from_links(self, insert=False, output=False):

        """
        :param insert: Boolean: ['True', 'False']; If True will insert the data into MongoDB
        :param output: Boolean: ['True, 'False']; If True will a list of dictionaries
        :return: List

        """

        for link in tqdm(self.master_list_of_links):
            self.driver.get(link)

            # ! Get title of the paper
            title_ = self.driver.find_element_by_xpath('//*[@id="abs"]/h1')

            # ! Get author/s of the paper
            authors_ = self.driver.find_element_by_xpath('//*[@id="abs"]/div[1]/a')

            # ! Get date of the paper
            date_ = self.driver.find_element_by_id("abs")
            date_ = date_.find_elements_by_class_name("dateline")

            # ! Get abstract of the paper
            abstract_ = self.driver.find_element_by_xpath('//*[@id="abs"]/blockquote')

            # ! Get subject of the paper
            table = self.driver.find_element_by_xpath('//*[@id="abs"]/div[3]/table')
            table = table.find_elements_by_tag_name("tr")

            for tr in list(table):
                if "subject" in tr.text.lower():
                    subject_ = tr.text.split(":")[1].strip()

            try:
                subject_
            except:
                subject_ == ""

            pdf = self.driver.find_element_by_xpath(
                '// *[ @ id = "abs"] / div[1] / div[1] / ul / li[1] / a'
            ).get_attribute("href")

            data = {
                "technology": self.topic.lower(),
                "title": title_.text,
                "authors": authors_.text,
                "date": date_[0].text,
                "abstract": abstract_.text,
                "subject": subject_,
                "paper_link": pdf,
                "created_at": str(datetime.now()),
            }

            self.records.append(data)

            if insert == True:
                self.col.insert(data)

            time.sleep(2)
            # pprint.pprint(data, width=1)

        self.t1 = time.time()

        hours, rem = divmod(self.t1 - self.t0, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Total Time: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )

        if output == True:
            return self.records

    def __get_ua__(self):
        return self.curr_ua


# if __name__ == '__main__':
#     ans = news_downloader(['https://www.nytimes.com/'])
#     print(ans)

