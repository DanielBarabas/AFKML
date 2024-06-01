from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver


###### General ######


def test_change_page(driver, page_name):
    link_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.LINK_TEXT, page_name))
    )
    if not link_element:
        raise AssertionError(f"Could not find page with name {page_name}")

    link_element.click()


def test_wait_xpath(driver, x_path: str):
    assert WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, x_path))
    ), f"Could not find element by {By.XPATH} with value {x_path}"


def test_wait_css(driver, css_selector: str):
    """Last part of the css selector (class.name) is enough if it is unique"""
    assert WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, css_selector))
    ), f"Could not find element by {By.CLASS_NAME} with value {css_selector}"


# KILL
def test_scroll_to_xpath(driver, x_path: str):
    elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, x_path))
    )
    if not elem:
        raise AssertionError(
            f"Could not find element by {By.XPATH} with value {x_path}"
        )

    driver.execute_script("arguments[0].scrollIntoView();", elem)


def test_scroll_to_tagname(driver, tag_name: str):
    elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, tag_name))
    )
    if not elem:
        raise AssertionError(
            f"Could not find element by {By.XPATH} with value {tag_name}"
        )

    driver.execute_script("arguments[0].scrollIntoView();", elem)


def test_scroll_to_element(driver, elem):
    if not elem:
        raise AssertionError(f"Could not find the given element ({elem})")

    driver.execute_script("arguments[0].scrollIntoView();", elem)
    body = driver.find_element(By.TAG_NAME, "body")
    body.send_keys(Keys.PAGE_UP)


def select_dropdown(driver, dropdown_css: str, options_css: str, option_i: int):
    """!!!option_i: be aware that the currently selected option is OMMITED from the list of options"""
    dropdown = driver.find_element(By.CSS_SELECTOR, dropdown_css)
    dropdown.click()

    time.sleep(1)

    options = driver.find_elements(By.CSS_SELECTOR, options_css)
    options[option_i].click()


def clear_multi(driver, css_multi: str, child_i: int = 1):
    childs = driver.find_elements(By.CSS_SELECTOR, css_multi)
    childs[child_i].click()


def fill_multi(driver, multi_class: str, options_css: str, options_i: list):
    """!!!options_i: be aware that the currently selected option is OMMITED from the list of options"""
    multi = driver.find_element(By.CLASS_NAME, multi_class)
    multi.click()
    options = driver.find_elements(By.CSS_SELECTOR, options_css)
    for i in options_i:
        options[i].click()


def click_something_from_list(list_of_clickables, number_of_clickable):
    try:
        toggle = list_of_clickables[number_of_clickable]
        toggle.click()
    except:
        raise AssertionError(
            f"Could not find the {number_of_clickable}th element on the page"
        )


###### Homepage ########
def test_data_upload(driver, file_path: str):
    file_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )
    if not file_input:
        raise AssertionError("Could not find file input")
    file_input.send_keys(file_path)


###### EDA ########


def scroll_to_h2(driver, h2s, h2_num):
    try:
        h2 = h2s[h2_num]
        driver.execute_script("arguments[0].scrollIntoView();", h2)
    except:
        raise AssertionError(f"Could not find the {h2_num}th header on the page")


def find_h2s(driver):
    try:
        return driver.find_elements(By.TAG_NAME, "h2")
    except:
        raise AssertionError("No h2 headers on the page")


def find_toggles(driver):
    elements = driver.find_elements(
        By.CSS_SELECTOR, "label[data-baseweb='checkbox'] > div"
    )
    del elements[::2]
    return elements


###### Encoding ########


###### Modelling ######
def find_sliders(driver):
    return driver.find_elements(By.CLASS_NAME, "st-emotion-cache-1vzeuhh.ew7r33m3")


def move_slider(driver, sliders, slider_num, movement):
    if movement < 0:
        webdriver.ActionChains(driver).click(sliders[slider_num]).send_keys(
            -movement * Keys.ARROW_DOWN
        ).perform()
    else:
        webdriver.ActionChains(driver).click(sliders[slider_num]).send_keys(
            movement * Keys.ARROW_UP
        ).perform()


def find_circles(driver):
    elements = driver.find_elements(
        By.CSS_SELECTOR, "label[data-baseweb='radio'] > div"
    )
    del elements[::2]
    return elements


def find_stepups(driver):
    return driver.find_elements(By.CLASS_NAME, "step-up")


def find_stepdowns(driver):
    return driver.find_elements(By.CLASS_NAME, "step-down")


def find_tickboxes(driver):
    return driver.find_elements(
        By.CSS_SELECTOR, "label[data-baseweb='checkbox'] > span"
    )


def find_run_button(driver):
    return driver.find_elements(By.CLASS_NAME, "st-emotion-cache-7ym5gk")
