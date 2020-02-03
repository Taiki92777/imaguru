#%%
# access
from selenium import webdriver

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
#%%
M4_link=driver.find_element_by_xpath('//*[@id="%s"]/li[3]/div/div/a')
M4_link.click()
#%%
blocks=driver.find_element_by_xpath('//*[@id="08bd794a925aadd9fc9f8088da892c0a"]/div[2]/section/div/div[1]/h4/a')
blocks.click()
#%%
# only smart phone version(desktop is impossible)
startcourse=driver.find_element_by_xpath('/html/body/div[1]/div[2]/form/div[1]/input')
startcourse.click()
#%%
# pdfがあればタブを開いて元のタブに戻って次へ
# それ以外は次へを押す
import urllib.request
import time
import random
time.sleep(1)
for i in range(73):
    try:
        driver.find_elements_by_class_name('ui-link')
        click_to_pdf=driver.find_element_by_xpath('//*[@id="Textbook"]/p[3]/a')
        click_to_pdf.click()
        time.sleep(random.random()+1)
        handle_array=driver.window_handles
        driver.switch_to_window(handle_array[0])
        time.sleep(random.random()+1)
        gotonext=driver.find_element_by_xpath('/html/body/div[1]/div[3]/span/div/a[3]/span')
        gotonext.click()
    except:
        time.sleep(random.random()+1)
        gotonext=driver.find_element_by_xpath('/html/body/div[1]/div[3]/span/div/a[3]/span')
        gotonext.click()
    