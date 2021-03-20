from selenium import webdriver
from os import getcwd
import re
driver = driver = webdriver.PhantomJS(service_args=['--load-images=no'],executable_path=getcwd() + "/phantomjs/bin/phantomjs")
driver.get('https://www.amazon.in/OnePlus-Display-Storage-3700mAH-Battery/product-reviews/B07HGBMJT6/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews')
'''
user = driver.find_element_by_xpath('//span[@class = "a-size-large"]')
print(user.text)
fullreviews = driver.find_element_by_xpath('//a[@class = "a-link-emphasis a-text-bold"]').click()


#rating span class = a-icon-alt
#date span class = a-size-base a-color-secondary review-date
#Top Reviewer Badge span class = a-size-mini a-color-link c7yBadgeAUI c7yTopDownDashedStrike c7y-badge-text a-text-bold
#verified purchase badge span class = a-size-mini a-color-state a-text-bold
#colur //a[class = a-size-mini a-link-normal a-color-secondary]
#helpful span class = a-size-base a-color-tertiary cr-vote-text
'''
main_div = driver.find_element_by_xpath('//div[@class = "a-section a-spacing-none review-views celwidget"]')
print(main_div)


rating =  driver.find_elements_by_xpath('//span[@class = "a-icon-alt"]')
badge_div =  driver.find_elements_by_xpath('//div[@class = "a-row a-spacing-mini review-data review-format-strip"]')
date =  driver.find_elements_by_xpath('//span[@class = "a-size-base a-color-secondary review-date"]')
for i in range (0,10):
    print(rating[i+3].get_attribute('innerHTML'))
    print(date[i+2].get_attribute('innerHTML'))
    x = re.search(r"(class=\"a-size-mini a-color-state a-text-bold\">.*?\</span></a></span>)", badge_div[0].get_attribute('innerHTML'))
    if x:
        y=re.search(r"(\">.*?</)",x.group())
        if y:
            z= y.group()
        else:
            z=None
    
    print(z)
    print(i,' \n\n')


print('\n\n\n')
