
import asyncio
import json
from playwright.async_api import async_playwright

async def main():
    """
    Main function to initialize Playwright, navigate to the FastTrack website,
    and intercept the FormContent request to get the full URL.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        dog_id = "890320106"
        dog_profile_url = f"https://fasttrack.grv.org.au/Dog/Form/{dog_id}"

        # Use a request handler to intercept the FormContent request
        async def handle_request(route, request):
            if "FormContent" in request.url:
                print(f"Intercepted FormContent request: {request.url}")
                # Save the full URL to a file
                with open("samples/fasttrack_raw/form_content_url.txt", "w") as f:
                    f.write(request.url)
                print("Saved FormContent URL to samples/fasttrack_raw/form_content_url.txt")
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", handle_request)

        try:
            await page.goto(dog_profile_url, wait_until="networkidle")
        except Exception as e:
            # The request is aborted after we find the URL, so an error is expected.
            # We can ignore it.
            pass

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

