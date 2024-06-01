from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import time

def change_page(driver,page_name):
     """Go to given page

     Args:
         driver (_type_): driver we use
         page_name (_type_): page to go to
     """
     driver.find_element(By.LINK_TEXT, page_name).click()
