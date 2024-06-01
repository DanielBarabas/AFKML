from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
import time
import test_pages.test_modules as tests


driver = Chrome()
driver.get("http://localhost:8501")
driver.maximize_window()


##### Homepage #####
tests.test_data_upload(
    driver=driver,
    file_path="C:/Users/aronn/OneDrive/Asztali gép/Rajk/Prog 2/prog_machine_project/data/alcohol_sample.csv",
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
time.sleep(20)

### Itt scrollolgatni kell majd


##### EDA #####
tests.test_change_page(driver=driver, page_name="EDA")

# TODO wait until the page is loaded
tests.test_wait_css(driver=driver, css_selector="h2[id='descriptive-table']")
time.sleep(2)

toggles = tests.find_toggles(driver=driver)
h2s = tests.find_h2s(driver=driver)

# desc stats
tests.test_scroll_to_element(driver=driver, elem=h2s[0])
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=0)

# distribution of categoricals
dropdown_css = "div[data-baseweb='select']"
options_css = 'div[data-baseweb="popover"] > div > div > ul > div > div > li'
toggles = tests.find_toggles(driver=driver)
h2s = tests.find_h2s(driver=driver)
tests.test_scroll_to_element(driver=driver, elem=h2s[1])
time.sleep(2)
tests.select_dropdown(
    driver=driver, dropdown_css=dropdown_css, options_css=options_css, option_i=0
)
time.sleep(1)
toggles = tests.find_toggles(driver=driver)
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=1)
toggles = tests.find_toggles(driver=driver)
h2s = tests.find_h2s(driver=driver)
# tests.test_scroll_to_element(driver=driver, elem=h2s[1])
time.sleep(5)

# Ass charts
"""toggles = tests.find_toggles(driver=driver)
h2s = tests.find_h2s(driver=driver)
tests.test_scroll_to_element(driver=driver, elem=h2s[2])
css_delete = 'div[data-baseweb="select"] > div > div > svg'
multi_class = "stMultiSelect"
options_css = 'div[data-baseweb="popover"] > div > div > ul > div > div > li'
tests.clear_multi(driver=driver, css_delete=css_delete, child_i=1)
time.sleep(5)
tests.fill_multi(
    driver=driver, multi_class=multi_class, options_css=options_css, options_i=[2, 3]
)
time.sleep(1)
driver.switch_to.default_content()
driver.find_element(By.CSS_SELECTOR, "h2[id='assocation-figure']").click()
time.sleep(1)
h2s = tests.find_h2s(driver=driver)
tests.test_scroll_to_element(driver=driver, elem=h2s[3])
toggles = tests.find_toggles(driver=driver)
print(f"toggles length {len(toggles)}")
time.sleep(2)
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=2)
time.sleep(5)
"""


# Corr matrix
toggles = tests.find_toggles(driver=driver)
h2s = tests.find_h2s(driver=driver)
tests.test_scroll_to_element(driver=driver, elem=h2s[3])
time.sleep(2)
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=3)


# Missing values
toggles = tests.find_toggles(driver=driver)
time.sleep(1)
# h2s = tests.find_h2s(driver=driver)
# print(len(toggles))
tests.test_scroll_to_element(driver=driver, elem=toggles[4])
time.sleep(1)
toggles = tests.find_toggles(driver=driver)
time.sleep(1)
tests.click_something_from_list(list_of_clickables=toggles, number_of_clickable=4)
time.sleep(1)
toggles = tests.find_toggles(driver=driver)
time.sleep(1)
# tests.test_scroll_to_element(driver=driver, elem=toggles[4])
time.sleep(1)


time.sleep(5)
##### Encoding #####
tests.test_change_page(driver=driver, page_name="Encoding")
time.sleep(5)
tests.select_dropdown(
    driver=driver,
    dropdown_css="div[data-baseweb='select']",
    options_css='div[data-baseweb="popover"] > div > div > ul > div > div > li',
    option_i=1,
)

time.sleep(2)
tests.test_change_page(driver=driver, page_name="Encoding")
time.sleep(3)
tests.update_agrid(driver, 0, 1)
time.sleep(4)
tests.update_agrid(driver, 0, 2)
time.sleep(4)
tests.update_agrid(driver, 0, 0)
time.sleep(3)

##### Modelling #####
tests.test_change_page(driver=driver, page_name="Modelling")
time.sleep(3)

# TODO do nice waiting here!
time.sleep(4)
sliders = tests.find_sliders(driver)
print(len(sliders))
tests.move_slider(driver, sliders, 1, movement=-95)
body = driver.find_element(By.TAG_NAME, "body")
body.send_keys(4 * Keys.ARROW_DOWN)
time.sleep(1)
tests.move_slider(driver, sliders, 1, movement=5)
time.sleep(3)
body.send_keys(4 * Keys.PAGE_DOWN)
button = tests.find_run_button(driver)
print(f"button {len(button)}")
button[0].click()
time.sleep(10)


##### Evaluation #####
tests.test_change_page(driver=driver, page_name="Evaluation")
time.sleep(2)
toggles = tests.find_toggles(driver)
print(len(toggles))
time.sleep(1)
h2s = tests.find_h2s(driver)
print(len(h2s))
body = driver.find_element(By.TAG_NAME, "body")
for i in range(len(h2s)):
    toggles = tests.find_toggles(driver)
    h2s = tests.find_h2s(driver)

    toggle = toggles[i]
    print(len(h2s))
    time.sleep(1)
    # tests.scroll_to_h2(driver,h2s,i)
    tests.click_something_from_list(toggles, i)
    time.sleep(6)

    body.send_keys(Keys.PAGE_DOWN)
body.send_keys(Keys.PAGE_DOWN)


time.sleep(5)
