from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By


driver = Chrome()
driver.get("http://localhost:8501")
driver.maximize_window()
