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
#%%
M4_link=driver.find_element_by_xpath('//*[@id="%s"]/li[3]/div/div/a')
M4_link.click()
#%%
Start_time=driver.find_element_by_xpath('//*[@id="d4b34987c8dbe3a3ffddaca452042905"]/div[2]/section/div/div[1]/div[2]/div[2]').text
Start_time=re.sub(r'\D','',Start_time)
Start_time=[int(x) for x in list(Start_time)]
Start_time=list(map(str,Start_time))
Year=int("".join(Start_time[:4]))
Month=int("".join(Start_time[4:6]))
Date=int("".join(Start_time[6:8]))
Hour=int("".join(Start_time[8:10]))
Minute=int("".join(Start_time[10:12]))
Start_time=('%.0f-%.0f-%.0f %.0f:%.0f:05'%(Year,Month,Date,Hour,Minute))
print(Start_time)
#%%
def click_to_test():
    go_to_test=driver.find_element_by_xpath('//*[@id="d4b34987c8dbe3a3ffddaca452042905"]/div[2]/section/div/div[1]/h4')
    go_to_test.click()
scheduler=sched.scheduler(time.time,time.sleep)
run_at=datetime.strptime(Start_time,'%Y-%m-%d %H:%M:%S')
run_at=int(time.mktime(run_at.utctimetuple()))
scheduler.enterabs(run_at,1,click_to_test)
scheduler.run()
#%%
# get questions
question_text=[]
syoumon=18
for i in syoumon: 
    question_text.apprnd(driver.find_element_by_xpath('//*[@id="QF_1"]/div[%.0f]/label/span/span[1]'%i).text)
#自然言語処理で正解をみつける。正解だったら、問題iのチェックボックスをクリック
'''
import correct_or_not
for i in syoumon:
    if correct_or_not.fit_predict(question_text[i])==correct(maybe 0 or 1):
        question_check=driver.find_element_by_xpath('//*[@id="1_%.0f"]'%i)
        question_check.click()

finish_button=driver.find_element_by_xpath('//*[@id="Question"]/div/a[@href="javascript:QuestionClient.finish()"]')
finish_button.click()
'''
