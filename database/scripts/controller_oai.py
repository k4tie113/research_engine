from multiprocessing import Pool
from subprocess import run

def fetch_year(year):
    print(f"🚀 Starting {year}")
    run(["python3", "fetch_oai.py", str(year)], check=True)
    print(f"✅ Finished {year}")

if __name__ == "__main__":
    years = list(range(2010, 2026))
    with Pool(processes=4) as pool:  # adjust for your CPU
        pool.map(fetch_year, years)
