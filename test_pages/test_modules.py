from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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


def test_scroll_to_xpath(driver, x_path: str):
    elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, x_path))
    )
    if not elem:
        raise AssertionError(
            f"Could not find element by {By.XPATH} with value {x_path}"
        )

    driver.execute_script("arguments[0].scrollIntoView();", elem)


###### Homepage ######


def test_data_upload(driver, file_path: str):
    file_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )
    if not file_input:
        raise AssertionError("Could not find file input")
    file_input.send_keys(file_path)
