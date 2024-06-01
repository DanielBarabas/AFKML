from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
import time
import test_pages.test_modules as tests


driver = Chrome()
driver.get("http://localhost:8501")
driver.maximize_window()


##### Homepage #####
tests.test_data_upload(
    driver=driver,
    file_path="C:/Users/aronn/OneDrive/Asztali gép/Rajk/Prog 2/prog_machine_project/data/drinking.csv",
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
tests.test_change_page(driver=driver, page_name="EDA")

# TODO wait until the page is loaded
tests.test_wait_css(driver=driver, css_selector="h2[id='descriptive-table']")
time.sleep(2)

toggles = tests.find_toggles(driver=driver)
h2s = tests.find_h2s(driver=driver)
print(len(h2s))

# desc stats
tests.test_scroll_to_element(driver=driver, elem=h2s[0])
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=0)

# distribution of categoricals
dropdown_css = "div[data-baseweb='select']"
options_css = 'div[data-baseweb="popover"] > div > div > ul > div > div > li'
tests.test_scroll_to_element(driver=driver, elem=h2s[1])
time.sleep(1)
tests.select_dropdown(
    driver=driver, dropdown_css=dropdown_css, options_css=options_css, option_i=0
)
time.sleep(1)
toggles = tests.find_toggles(driver=driver)
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=1)
time.sleep(5)

# Ass charts
multi_class = "stMultiSelect"
options_css = 'div[data-baseweb="popover"] > div > div > ul > div > div > li'
tests.clear_multi(driver=driver, multi_class=multi_class)
time.sleep(1)
tests.fill_multi(
    driver=driver, multi_class=multi_class, options_css=options_css, options_i=[2, 3]
)
toggles = tests.find_toggles(driver=driver)
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=1)
time.slep

time.sleep(10)
##### Encoding #####
tests.test_change_page(driver=driver, page_name="Encoding")


##### Modelling #####
tests.test_change_page(driver=driver, page_name="Modelling")


##### Evaluation #####


time.sleep(5)
