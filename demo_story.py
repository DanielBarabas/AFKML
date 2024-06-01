from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
import subprocess
import time
from test_pages.test_modules import (
    test_data_upload,
    test_wait_css,
    test_scroll_to_xpath,
)

streamlit_process = subprocess.Popen("streamlit run 1_Homepage.py")

driver = Chrome()
driver.get("http://localhost:8501")
driver.maximize_window()


##### Homepage #####
test_data_upload(
    driver=driver,
    file_path="C:/Users/aronn/OneDrive/Asztali gép/Rajk/Prog 2/prog_machine_project/data/drinking.csv",
)

test_wait_css(
    driver=driver,
    css_selector="div.dvn-scroller.glideDataEditor",
)


time.sleep(5)

streamlit_process.terminate()
