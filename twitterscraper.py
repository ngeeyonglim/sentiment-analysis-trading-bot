import time
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
    NoSuchElementException
)
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os
# import openpyxl # pandas typically handles this via its Excel engine

load_dotenv('credentials.env') # Load environment variables from .env file if needed
GOOGLE_USERNAME = os.getenv('GOOGLE_USERNAME')
GOOGLE_PASSWORD = os.getenv('GOOGLE_PASSWORD')

options = Options()
options.add_experimental_option("detach", True)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("window_size=1280,800")
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-save-password-bubble")
# options.add_argument("--headless") # Optional: for running without a visible browser
# options.add_argument("--lang=en-US") # Optional: try to set browser language

def save_to_excel(data_list, filename):
    # Ensure data_list is a list of single tweet texts for a single column DataFrame
    data_to_save = [{"Tweets": tweet} for tweet in data_list]
    df_new = pd.DataFrame(data_to_save)

    if os.path.exists(filename):
        try:
            df_existing = pd.read_excel(filename, engine='openpyxl')
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Optional: Remove duplicates from the combined data based on the "Tweets" column
            df_combined.drop_duplicates(subset=["Tweets"], keep='first', inplace=True)
        except Exception as e:
            print(f"Error reading existing Excel file '{filename}': {e}. Overwriting with new data.")
            df_combined = df_new
    else:
        df_combined = df_new

    try:
        df_combined.to_excel(filename, index=False, engine='openpyxl')
        print(f"Successfully saved/updated {len(df_combined)} unique tweets to {filename}")
    except Exception as e:
        print(f"Error writing to Excel file '{filename}': {e}")


driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20) # Increased default wait time

# --- Google Login ---
print("Attempting Google login...")
# Using a more generic Google sign-in URL, but your specific one might also work
driver.get("https://accounts.google.com/signin")

# Google Credentials
email = GOOGLE_USERNAME
password = GOOGLE_PASSWORD

try:
    email_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="identifierId"]')))
    email_input.send_keys(email)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="identifierNext"]/div/button/span'))).click()
    print("Google email submitted.")

    password_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="password"]/div[1]/div/div[1]/input')))
    password_input.send_keys(password)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="passwordNext"]/div/button/span'))).click()
    print("Google password submitted.")
    # Add a check for successful Google login, e.g., waiting for a known element on a Google page
    # For now, a short sleep to allow redirects/2FA prompts.
    print("Google login attempt finished. Waiting for potential 2FA or redirects...")
    time.sleep(5) # Give time for any post-login Google pages or 2FA
except TimeoutException:
    print("Timeout during Google login. Elements not found or page took too long.")
    driver.quit()
    exit()
except Exception as e:
    print(f"An error occurred during Google login: {e}")
    driver.quit()
    exit()

print("Google login process completed.")

# --- Navigate to Twitter/X and attempt "Sign in with Google" ---
print("\nNavigating to X.com (Twitter)...")
driver.get("https://x.com/login") # Go directly to login page
time.sleep(3) # Allow page to load login options

original_window = driver.current_window_handle

try:
    print("Looking for 'Sign in with Google' button on X.com...")
    # !!! YOU MUST VERIFY/UPDATE THIS XPATH !!!
    # This XPATH is a common guess. Inspect X.com's login page.
    # It might be in an iframe.
    # Option 1: Direct button
    # sign_in_with_google_button_xpath_on_x = "//button[contains(., 'Google')]"
    # Option 2: Button with specific data-testid (often more stable)
    sign_in_with_google_button_xpath_on_x = "//button[@data-testid='google_sign_in_button']"
    # Option 3: Link
    # sign_in_with_google_button_xpath_on_x = "//a[contains(., 'Google')]"
    # Option 4: More generic span inside a clickable div/button
    # sign_in_with_google_button_xpath_on_x = "//*[text()='Sign in with Google']"
    # Check if it's in an iframe
    iframe_xpath = "//iframe[contains(@src, 'google.com/accounts') or contains(@title, 'Google')]" # Common iframe patterns
    try:
        google_iframe = wait.until(EC.presence_of_element_located((By.XPATH, iframe_xpath)))
        print("Google sign-in iframe found. Switching to iframe...")
        driver.switch_to.frame(google_iframe)
        # Now find the button within the iframe
        sign_in_with_google_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@role='button' or @role='link' or @id='button']"))) # Generic inside iframe
        sign_in_with_google_button.click()
        driver.switch_to.default_content() # Switch back from iframe
        print("Clicked 'Sign in with Google' button (inside iframe) on X.com.")
    except TimeoutException:
        print("No iframe found for Google Sign-In, or button not found in iframe. Trying direct button on page.")
        sign_in_with_google_button = wait.until(EC.element_to_be_clickable((By.XPATH, sign_in_with_google_button_xpath_on_x)))
        sign_in_with_google_button.click()
        print("Clicked 'Sign in with Google' button (direct) on X.com.")

    # --- Handle Google Account Chooser Pop-up/New Window ---
    print("Waiting for Google Account Chooser...")
    time.sleep(3) # Give a moment for the chooser to appear

    # Check for new windows
    if len(driver.window_handles) > 1:
        for window_handle in driver.window_handles:
            if window_handle != original_window:
                driver.switch_to.window(window_handle)
                print(f"Switched to new window: {driver.title}")
                break
    else:
        print("Google Account Chooser likely in the same window or a modal.")

    # Now, click the specific account in the Google Account Chooser
    # This is where your 'Sign in as...' XPATH is relevant
    sign_in_as_user_button_xpath = f"//div[contains(text(),'Sign in as {email.split('@')[0]}') or contains(text(),'{email}')]"
    # More generic if the above is too specific:
    # sign_in_as_user_button_xpath = "//div[starts-with(text(),'Sign in as ')]" # Your original one
    
    account_to_select_button = wait.until(EC.element_to_be_clickable((By.XPATH, sign_in_as_user_button_xpath)))
    print(f"Found account chooser option: {account_to_select_button.text}")
    account_to_select_button.click()
    print("Selected account from Google Account Chooser.")

    # Switch back to the original window if we switched for the popup
    if driver.current_window_handle != original_window and original_window in driver.window_handles:
        driver.switch_to.window(original_window)
        print("Switched back to original X.com window.")
    
    # Wait for login to complete and redirect to X.com homepage
    print("Waiting for X.com to load after Google sign-in...")
    # Wait for a known element on the logged-in X.com homepage, like the main search bar or "For You" tab.
    wait.until(EC.presence_of_element_located((By.XPATH, "//input[@data-testid='SearchBox_Search_Input']")))
    print("Successfully logged into X.com using Google.")
    time.sleep(3) # Extra pause for page to fully render

except TimeoutException:
    print("Timeout: Could not complete 'Sign in with Google' flow on X.com or failed to log in.")
    print("Current URL:", driver.current_url)
    # driver.quit() # Decide if you want to quit or try another login method
    # exit()
except Exception as e:
    print(f"An error occurred during X.com 'Sign in with Google': {e}")
    print("Current URL:", driver.current_url)
    # driver.quit()
    # exit()


# --- Scraping Logic ---
hashtags = ['BTC', 'bitcoin', 'crypto']
all_collected_tweets = []
seen_tweet_texts_globally = set()
MAX_CONSECUTIVE_SCROLLS_WITHOUT_NEW_TWEETS = 3 # Stop after 3*N scrolls if no new tweets
TWEETS_TO_SCRAPE_PER_HASHTAG = 100 # Optional: limit per hashtag

for hashtag_query in hashtags:
    print(f"\n--- Processing hashtag: {hashtag_query} ---")
    driver.get("https://x.com/explore")
    print(f"Navigated to explore page for '{hashtag_query}'")

    try:
        # This is the search bar on the EXPLORE page.
        explore_search_input_xpath = "//input[@data-testid='SearchBox_Search_Input']"
        # old '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div/div[1]/div[2]/div/div/div/form/div[1]/div/div/div/label/div[2]/div/input'

        search_input = wait.until(EC.visibility_of_element_located((By.XPATH, explore_search_input_xpath)))
        search_input.clear()
        search_input.send_keys(hashtag_query + " lang:en")
        search_input.send_keys(Keys.ENTER)
        print(f"Searched for '{hashtag_query} lang:en'")
        time.sleep(5) # Allow search results to load and "Latest" tab to appear

        # Optional: Click on "Latest" tab for chronological tweets
        # try:
        #     latest_tab_xpath = "//a[@href='/explore/tabs/latest']//span[contains(text(),'Latest')]"
        #     wait.until(EC.element_to_be_clickable((By.XPATH, latest_tab_xpath))).click()
        #     print("Clicked on 'Latest' tab.")
        #     time.sleep(3) # Allow latest tweets to load
        # except TimeoutException:
        #     print("'Latest' tab not found or not clickable. Proceeding with default (Top) tweets.")

        actions = ActionChains(driver)
        consecutive_scrolls_without_new = 0
        TWEETS_TO_SCRAPE_PER_HASHTAG = 10
        
        # Keep track of tweets seen in this specific hashtag search to avoid reprocessing during scrolls
        tweets_seen_this_hashtag_search = set()

        while True:
            new_tweets_found_this_batch = 0
            # Scroll a few times
            scroll_attempts = 5
            print(f"Scrolling down {scroll_attempts} times...")
            for _ in range(scroll_attempts):
                actions.send_keys(Keys.PAGE_DOWN).perform()
                time.sleep(1.5) # Shorter sleep, adjust if content loads slower

            # Fetch tweet data
            # This XPATH targets the text content of tweets.
            tweet_text_elements_xpath = "//article[@data-testid='tweet']//div[@data-testid='tweetText']"

            tweet_elements_found = driver.find_elements(By.XPATH, tweet_text_elements_xpath)
            print(f"Found {len(tweet_elements_found)} potential tweet text elements on page.")

            if not tweet_elements_found and consecutive_scrolls_without_new == 0: # First check after scrolls
                 print("No tweet elements found with the current XPATH. Check XPATH or if search yielded results.")

            for tweet_element in tweet_elements_found:
                try:
                    tweet_text = tweet_element.text.strip()
                    if tweet_text and tweet_text not in seen_tweet_texts_globally and tweet_text not in tweets_seen_this_hashtag_search:
                        all_collected_tweets.append(tweet_text)
                        seen_tweet_texts_globally.add(tweet_text)
                        tweets_seen_this_hashtag_search.add(tweet_text) # Add to set for this hashtag's scroll session
                        print(f"  New unique tweet: {tweet_text[:80]}...")
                        new_tweets_found_this_batch += 1
                except StaleElementReferenceException:
                    print("StaleElementReferenceException caught while reading tweet text. Will re-fetch on next scroll.")
                    break # Break from processing this batch of elements, will scroll and re-fetch
                except Exception as e_inner:
                    print(f"Error processing a tweet element: {e_inner}")
            
            if new_tweets_found_this_batch > 0:
                consecutive_scrolls_without_new = 0 # Reset counter
                # Save intermediate results
                save_to_excel(list(all_collected_tweets), "tweets.xlsx")
            else:
                consecutive_scrolls_without_new += 1
                print(f"No new unique tweets found in this scroll batch. Consecutive count: {consecutive_scrolls_without_new}")

            if consecutive_scrolls_without_new >= MAX_CONSECUTIVE_SCROLLS_WITHOUT_NEW_TWEETS:
                print(f"No new unique tweets after {MAX_CONSECUTIVE_SCROLLS_WITHOUT_NEW_TWEETS} scroll cycles. Moving to next hashtag or finishing.")
                break
            
            # Safety break if too many tweets are collected for one hashtag (e.g., > 500)
            if len(tweets_seen_this_hashtag_search) > TWEETS_TO_SCRAPE_PER_HASHTAG:
                print(f"Collected over 500 tweets for '{hashtag_query}'. Moving to next hashtag.")
                break

            


    except TimeoutException:
        print(f"Timeout while searching or loading content for hashtag: '{hashtag_query}'. Skipping to next.")
    except NoSuchElementException as e:
        print(f"NoSuchElementException for hashtag '{hashtag_query}': {e}. XPATH might be wrong for explore search or tweet content.")
        print("Current URL:", driver.current_url)
    except Exception as e_outer:
        print(f"An unexpected error occurred for hashtag '{hashtag_query}': {e_outer}")
        print("Current URL:", driver.current_url)

# Final save after all hashtags are processed
if all_collected_tweets:
    save_to_excel(list(all_collected_tweets), "tweets.xlsx")

print("\n--- Scraping Finished ---")
print(f"Total unique tweets collected from all hashtags: {len(all_collected_tweets)}")
# for text in all_collected_tweets:
#     print(text[:100] + "...") # Print snippets

# driver.quit() # Uncomment to close the browser automatically when done
print("Script finished. Browser window remains open if detach option is active.")