#%%
# access
from selenium import webdriver
import time
import sched
import re
from datetime import datetime

driver=webdriver.Chrome(rf'C:\Users\taiki\AppData\Local\Driver\chromedriver.exe')
driver.get('https://lib02.tmd.ac.jp/webclass/login.php')
#%%
# login
ID=driver.find_element_by_id('username')
ID.send_keys('gpat1981')
password=driver.find_element_by_id('password')
password.send_keys('yuta1202')

login_button=driver.find_element_by_id('LoginBtn')
login_button.click()
