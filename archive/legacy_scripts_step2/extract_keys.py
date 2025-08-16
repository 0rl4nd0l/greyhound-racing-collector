
import asyncio
import re
from playwright.async_api import async_playwright

async def main():
    dog_id = "890320106"
    url = f"https://fasttrack.grv.org.au/Dog/Form/{dog_id}"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)

        # Get the page's script content
        scripts = await page.query_selector_all('script')
        for script in scripts:
            content = await script.inner_text()
            if 'formcontent' in content.lower():
                # Extract the key and sharedKey using regex
                key_match = re.search(r'key:\s*"([a-f0-9\-]+)"', content)
                shared_key_match = re.search(r'sharedKey:\s*"([a-zA-Z0-9%=-]+)"', content)

                if key_match and shared_key_match:
                    key = key_match.group(1)
                    shared_key = shared_key_match.group(1)
                    print(f"Found key: {key}")
                    print(f"Found sharedKey: {shared_key}")
                    break
        else:
            print("Could not find key and sharedKey in page scripts.")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

