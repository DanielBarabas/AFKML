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


### Itt scrollolgatni kell majd


##### EDA #####


#time.sleep(10)
##### Encoding #####
tests.test_change_page(driver=driver, page_name="Encoding")

time.sleep(4)
##### Modelling #####
tests.test_change_page(driver=driver, page_name="Modelling")
#TODO do nice waiting here!
time.sleep(4)

# Move the first slider
sliders = tests.find_sliders(driver)
print(len(sliders))
tests.move_slider(driver,sliders,0,movement=-16)
time.sleep(1)
tests.move_slider(driver,sliders,0,movement=20)

# Click on second circle and slide on the appeared slider
circles = tests.find_circles(driver)
tests.test_scroll_to_element(driver,circles[1])
#TODO nicer waiting
time.sleep(0.1)
tests.click_something_from_list(circles,1)

time.sleep(1) # Hogy legyen ideje megjelenni a második slidernek
sliders = tests.find_sliders(driver)
print(len(sliders))
tests.move_slider(driver,sliders,2,movement=5) 
time.sleep(2)
tests.move_slider(driver,sliders,2,movement=-5)
time.sleep(2)
circles = tests.find_circles(driver)
print(len(circles))
tests.test_scroll_to_element(driver,circles[5])
time.sleep(2)
body = driver.find_element(By.TAG_NAME, 'body') 
body.send_keys(3*Keys.ARROW_DOWN)
time.sleep(2)
tests.click_something_from_list(circles,5)
body.send_keys(5*Keys.ARROW_DOWN)
sliders = tests.find_sliders(driver)
print(len(sliders))
time.sleep(2)
tests.move_slider(driver,sliders,3,movement=6) 
time.sleep(2)
tests.move_slider(driver,sliders,3,movement=-2)
time.sleep(2)

tbs = tests.find_tickboxes(driver)
time.sleep(2)
tests.test_scroll_to_element(driver,tbs[1])
body.send_keys(3*Keys.ARROW_DOWN)
time.sleep(2)
tests.click_something_from_list(tbs,1)
##### Evaluation #####


time.sleep(100)
