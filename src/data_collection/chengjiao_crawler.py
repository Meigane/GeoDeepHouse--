import pandas as pd  
from selenium import webdriver  
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.common.by import By  
from selenium.webdriver.support.ui import WebDriverWait  
from selenium.webdriver.support import expected_conditions as EC 
import time  
import re 
from tqdm import tqdm
from house_data_crawler import DataCrawler
from selenium.webdriver.support import expected_conditions as EC

from sql_saver import SqlSaver
from datetime import datetime

class ChengJiao(DataCrawler):
    # DISTRICT_MAP = {
    # "黄浦区": "huangpu",  "徐汇区": "xuhui",  
    # "长宁区": "changning",  "静安区": "jingan",  
    # "普陀区": "putuo",  "虹口区": "hongkou",  
    # "杨浦区": "yangpu",  "闵行区": "minhang",  
    # "宝山区": "baoshan",  "嘉定区": "jiading",  
    # "浦东新区": "pudong",  "金山区": "jinshan",  
    # "松江区": "songjiang",  "青浦区": "qingpu",  
    # "奉贤区": "fengxian"#,  "崇明区": "chongming"  
    # }

    DISTRICT = {"浦东新区":['beicai', 'biyun', 'caolu', 'chuansha', 
                            'datuanzhen', 'gaodong', 'gaohang', 'geqing',
                            'hangtou', 'huamu', 'huinan', 'jinqiao', 'jinyang', 
                            'kangqiao', 'laogangzhen', 'lianyang', 'lingangxincheng',
                            'lujiazui','meiyuan1', 'nanmatou', 'nichengzhen', 'sanlin', 'shibo', 
                            'shuyuanzhen','tangqiao', 'tangzhen', 'waigaoqiao', 
                            'wanxiangzhen', 'weifang', 'xinchang', 'xuanqiao', 
                            'yangdong', 'yangjing', 'yangsiqiantan', 'qiantan', 'yuanshen',
                            'yuqiao1', 'zhangjiang', 'zhoupu', 'zhuqiao'],
                "闵行区":['caohejing', 'chunshen', 'gumei', 'hanghua', 
                            'huacao', 'jinganxincheng', 'jinhongqiao', 'jinhui',
                            'laominxing', 'longbai', 'maqiao', 'meilong', 'minpu', 'pujiang1', 
                            'qibao', 'xinminbieshu', 'wujing', 'xinzhuangbeiguangchang', 
                            'xinzhuangnanguangchang', 'zhuanqiao']
                }


    def __init__(self, url, infos=None, file=None):
        super().__init__(url, infos, file)
        self.db = SqlSaver("GisProject")

    def login(self):
        self.driver.maximize_window()
        self.driver.get(self._url)
        time.sleep(2)
        #输入自己已经注册好的账号
        self.driver.find_element(By.XPATH,"""/html/body/div[1]/div/
                                 div/div/div[1]/div[2]/div/div[2]/
                                 div[2]/form/input[1]""").send_keys('17317268707')
        
        time.sleep(0.5)
        #输入密码
        self.driver.find_element(By.XPATH,"""/html/body/div[1]/div/div/
                                 div/div[1]/div[2]/div/div[2]/div[2]/form/
                                 input[2]""").send_keys('pwd=lianjia')
        time.sleep(0.5)
        
        #点击登录
        self.driver.find_element(By.XPATH,"""/html/body/div[1]/div/div/
                                 div/div[1]/div[2]/div/div[2]/div[2]/
                                 form/button""").click()
        time.sleep(40)# 需要手动完成验证码登录

    def get_url(self):
        data = {}  
        label = 'title'
        elements = self.driver.find_elements(By.CLASS_NAME, label)
        elements = elements[1:] 
        elements_text = [element.text for element in elements if element.text]
        data['name'] = elements_text  
        element_href = [element.find_element(By.TAG_NAME, 'a').get_attribute("href") 
                        for element in elements if element.text]
        data['url'] = element_href

        return pd.DataFrame(data)

    def crawl_data(self):
        self.login()
        for district in self.DISTRICT["浦东新区"]:
            print(f"现在在爬取{district}")
            self.driver.get(self._url + "/" + district)
            max_page = self.get_page_num() 
            if max_page < 100: 
                self.crawl_url(district, max_page)
            else:
                for i in range(1, 7):
                    self.driver.get(self._url + "/" + district + f"/p{i}/")
                    self.crawl_url(district)
            self.crawl_info(district)

    def crawl_url(self, district, max_page=None):
        if not max_page:
            max_page = self.get_page_num() 
            print(f"开始爬取数据，共{max_page}页") 

        wait = WebDriverWait(self.driver, 20)  # 设置显式等待时间，20秒  
        for i in tqdm(range(1, max_page + 1), ncols=60):  
            if i > 1:  
                next_page = wait.until(  
                    EC.element_to_be_clickable((By.LINK_TEXT, str(i)))  
                )  
                self.driver.execute_script("arguments[0].click();", next_page)
            df  = self.get_url() 
            self.db.insert_data(district+"_url", df)

    def crawl_info(self, district):
        df = pd.read_sql_query(f"SELECT distinct url FROM {district}_url;", self.db.engine)
        i = 1
        for url in tqdm(df["url"], ncols=60):
            i  += 1
            if i % 2000 == 0:
                time.sleep(20)
            self.driver.get(url)
            df = self.get_info_df()
            self.db.insert_data(district+"_info", df)
            time.sleep(0.1) 

    def get_info_df(self):  
        """  
        从网页中获取指定信息，并返回DataFrame  
        """  
        info_dic = {"wrapper": "名字", "price":"价格","introContent":"内容"}
        data = {}  
        for label, name in info_dic.items():  
            element = self.driver.find_element(By.CLASS_NAME, label)  
            data[name] = element.text.strip()  
  
        title_parts = data['名字'].split(' ')  
        rst = {  
            '小区': title_parts[0],  
            '房型': title_parts[1] if len(title_parts) > 1 else None,  
        }  
  
        # 尝试解析面积和成交时间
        try:  
            rst['面积'] = float(title_parts[2].split('平米')[0])  
        except:  
            rst['面积'] = None  
        
        try:
            rst['成交时间'] = datetime.strptime(title_parts[2].split('平米')[1].strip('()').split('成交')[1], '%Y.%m.%d').date()  
        except:
            rst['成交时间'] = None  
  
        info_dict = self.extract_info(data['内容'])  
        if info_dict:  
            rst.update(info_dict)  
  
        price_parts = data['价格'].split('万')  
        try:  
            rst['总价'] = float(price_parts[0])
        except:  
            rst['总价'] = None  
        try:  
            rst['单价'] = float(price_parts[1].replace('元/平', '').strip())  
        except:
            rst['单价'] = None  
  
        # 尝试转换日期和建成年代
        try:  
            rst['挂牌时间'] = datetime.strptime(rst.get('挂牌时间', ''), '%Y-%m-%d').date()  
        except:  
            rst['挂牌时间'] = None  
  
        try:  
            rst['建成年代'] = int(rst.get('建成年代', ''))  
        except:  
            rst['建成年代'] = None  
  
        return pd.DataFrame([rst])  

    def extract_info(self, info_str):
        keywords = [  
        '房屋户型', '所在楼层', '户型结构', '建筑类型', 
        '房屋朝向', '建成年代', '装修情况', '建筑结构',  
        '供暖方式', '梯户比例', '配备电梯', '链家编号', '交易权属',  
        '挂牌时间', '房屋用途', '房屋年限', '房权所属'  
        ]
    
        info_dict = {}
        for keyword in keywords:  
            start_index = info_str.find(keyword)  
            if start_index == -1:  
                return None  
            start_index += len(keyword)  
            end_index = start_index  
            while end_index < len(info_str) and not info_str[end_index].isspace():  
                end_index += 1  
            info = info_str[start_index:end_index].strip() 
            if info == '暂无数据':
                info = None
            info_dict[keyword] = info

        return info_dict


if __name__ == "__main__":
    # 以崇明区为例进行测试
    # info_dic = {"wrapper": "名字", "price":"价格","introContent":"内容"}
    # url_dic = {"name": "TEXT PRIMARY KEY", "url": "TEXT"}
    # test = ChengJiao(url='https://sh.lianjia.com/chengjiao', infos=info_dic)
    # test.crawl_data("info", "chongming")

    data_crawler = ChengJiao(url='https://sh.lianjia.com/chengjiao')
    data_crawler.crawl_data()
    