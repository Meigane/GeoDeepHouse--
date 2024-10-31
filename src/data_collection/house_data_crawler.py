import pandas as pd  
from selenium import webdriver  
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.common.by import By  
from selenium.webdriver.support.ui import WebDriverWait  
from selenium.webdriver.support import expected_conditions as EC  
import re  
from tqdm import tqdm  
import time
import json
import geopandas as gpd
  
class DataCrawler:  
    CHROMEDRIVER_PATH = r'C:\Users\14531\anaconda3\Scripts\chromedriver.exe'  
  
    def __init__(self, url, infos, file):  
        """  
        初始化数据爬取器  
        :param url: 要爬取的网页URL  
        :param infos: 需要爬取的信息标签和对应名称的字典  
        :param file: 数据保存的文件名  
        """  
        self._url = url  
        self._infos = infos  
        self._df = pd.DataFrame()  
        self._file = file  
        self.driver = webdriver.Chrome(service=Service(self.CHROMEDRIVER_PATH))  
  
    def get_info_df(self):  
        """  
        从网页中获取指定信息，并返回DataFrame  
        """  
        data = {}  
        for label, name in self._infos.items():  
            elements = self.driver.find_elements(By.CLASS_NAME, label)  
            elements_text = [element.text for element in elements]  
            data[name] = elements_text  

        return pd.DataFrame(data)  
  
    def get_page_num(self):  
        """  
        获取网页中的最大页数  
        """  
        wait = WebDriverWait(self.driver, 10)
        page_path = "/html/body/div[5]/div[1]/div[5]/div[2]/div"
        pages = self.driver.find_element(By.XPATH, "/html/body/div[5]/div[1]/div[5]/div[2]/div")  
        match_max_page = re.search(r'\.\.\.(.+?)下一页', pages.text)  
        if match_max_page:  
            max_page = int(match_max_page.group(1).strip())  
        else:  
            max_page = int(input("没有找到最大页数，请输入："))  
        return max_page
  
    def crawl_data(self):  
        """  
        开始爬取数据，并将结果保存到Excel文件中  
        """  
        max_page = self.get_page_num()
  
        all_data = []  
        wait = WebDriverWait(self.driver, 20)  # 设置显式等待时间，20秒  
        for i in tqdm(range(1, max_page + 1), ncols=60):  
            if i > 1:  
                next_page = wait.until(  
                    EC.element_to_be_clickable((By.LINK_TEXT, str(i)))  
                )  
                self.driver.execute_script("arguments[0].click();", next_page)
            df = self.get_info_df()  
            all_data.append(df)  
        self.driver.close()  
        self._df = pd.concat(all_data, ignore_index=True)  
        self._df.to_excel(self._file)  
        print(f"爬取结束，成功写入{self._file}") 
  
if __name__ == "__main__":  
    url = r'https://sh.lianjia.com/chengjiao/chongming/'  
    infos = {"positionInfo": "位置", "houseInfo": "房屋信息", "priceInfo": "价格信息"}  
    file = 'second_hand_house_info.xlsx'  
    second_hand_house = DataCrawler(url, infos, file)  
    second_hand_house.crawl_data()