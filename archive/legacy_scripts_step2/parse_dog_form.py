
import json
from bs4 import BeautifulSoup
import pandas as pd

def main():
    """
    Parses the saved HTML from the Dog FormContent endpoint and extracts the race data into a CSV file.
    """
    dog_id = "890320106"
    html_file = f"samples/fasttrack_raw/dog_{dog_id}_form_content.html"

    with open(html_file, "r") as f:
        # The file contains a JSON string, so we need to load it first
        json_string = f.read()
        json_data = json.loads(json_string)  # Parse the outer JSON string
        if isinstance(json_data, str):
            # It's a nested JSON string, parse it again
            json_data = json.loads(json_data)
        html_content = json_data['content']

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table with the dog's runs
    runs_table = soup.find("table", {"id": "dogRuns"})

    if not runs_table:
        print("Could not find the dog runs table.")
        return

    # Extract the table headers
    headers = [th.text for th in runs_table.find("thead").find_all("th")]

    # Extract the table rows
    rows = []
    for tr in runs_table.find("tbody").find_all("tr"):
        rows.append([td.text.strip() for td in tr.find_all("td")])

    # Create a Pandas DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Save the DataFrame to a CSV file
    csv_file = f"samples/fasttrack_raw/dog_{dog_id}_races.csv"
    df.to_csv(csv_file, index=False)

    print(f"Saved dog race data to {csv_file}")

if __name__ == "__main__":
    main()

