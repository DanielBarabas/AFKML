from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
import time
import test_pages.test_modules as tests
from selenium.webdriver.common.keys import Keys

driver = Chrome()
driver.get("http://localhost:8501")
driver.maximize_window()


##### Homepage #####
tests.test_data_upload(
    driver=driver,
    file_path="C:/Projects/Rajk/prog_2/project/prog_machine_project/data/alcohol_sample.csv",
)

tests.test_wait_css(
    driver=driver,
    css_selector="div.dvn-scroller.glideDataEditor",
)

h2s = tests.find_h2s(driver=driver)

tests.scroll_to_h2 = tests.scroll_to_h2(driver=driver, h2s=h2s, h2_num=0)


### Change data types header

time.sleep(2)


##### Pandas profiling #####
tests.test_change_page(driver=driver, page_name="Pandas profiling")

##### EDA ######

###### Encoding ######

tests.test_change_page(driver=driver, page_name="Encoding")
time.sleep(3)
tests.update_agrid(driver,0,1)
time.sleep(4)
tests.update_agrid(driver,0,2)
time.sleep(4)
tests.update_agrid(driver,0,0)

###### Modelling ######

tests.test_change_page(driver=driver, page_name="Modelling")
#TODO do nice waiting here!
time.sleep(4)
sliders = tests.find_sliders(driver)
print(len(sliders))
tests.move_slider(driver,sliders,1,movement=-95)
body = driver.find_element(By.TAG_NAME, 'body') 
body.send_keys(4*Keys.ARROW_DOWN)
time.sleep(1)
tests.move_slider(driver,sliders,1,movement=5)
time.sleep(3)
body.send_keys(4*Keys.PAGE_DOWN)
button = tests.find_run_button(driver)[0]
button.click()
time.sleep(10)



###### Evaluation ######
tests.test_change_page(driver=driver, page_name="Evaluation")
time.sleep(1)
toggles = tests.find_toggles(driver)
print(len(toggles))
time.sleep(1)
h2s = tests.find_h2s(driver)
print(len(h2s))
body = driver.find_element(By.TAG_NAME, 'body') 
for i in range(len(h2s)):
    toggles = tests.find_toggles(driver)
    h2s = tests.find_h2s(driver)

    toggle = toggles[i]
    print(len(h2s))
    time.sleep(1)
    #tests.scroll_to_h2(driver,h2s,i)
    tests.click_something_from_list(toggles,i)
    time.sleep(6)
    
    body.send_keys(Keys.PAGE_DOWN)
body.send_keys(Keys.PAGE_DOWN)

    

